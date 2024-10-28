from utils_2016312761 import utils
from torch import nn
import torch.nn.functional as F
from math import sqrt
from itertools import product as product
import torchvision
import torch 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

        self.load_pretrained_layers()

    def forward(self, image):
        x = F.relu(self.conv1_1(image))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool3(x)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        conv4_3_features = x
        x = self.pool4(x)

        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.pool5(x)

        out = F.relu(self.conv6(x))
        conv7_features = F.relu(self.conv7(out))

        return conv4_3_features, conv7_features

    def load_pretrained_layers(self):
        state_dict = self.state_dict()
        parameters = list(state_dict.keys())

        pretrained_state_dict = torchvision.models.vgg16(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())

        for i, param in enumerate(parameters[:-4]):
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        fc6_weight = pretrained_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)
        fc6_bias = pretrained_state_dict['classifier.0.bias']
        state_dict['conv6.weight'] = utils.decimation(fc6_weight, m=[4, None, 3, 3])
        state_dict['conv6.bias'] = utils.decimation(fc6_bias, m=[4])
        
        fc7_weight = pretrained_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)
        fc7_bias = pretrained_state_dict['classifier.3.bias']
        state_dict['conv7.weight'] = utils.decimation(fc7_weight, m=[4, 4, None, None])
        state_dict['conv7.bias'] = utils.decimation(fc7_bias, m=[4])

        self.load_state_dict(state_dict)

class AuxConv(nn.Module):
    def __init__(self):
        super(AuxConv, self).__init__()

        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, padding=0)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)
        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)

        self.init_conv2d()

    def init_conv2d(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv7_features):
        x = F.relu(self.conv8_1(conv7_features))
        x = F.relu(self.conv8_2(x))
        conv8_features = x

        x = F.relu(self.conv9_1(x))
        x = F.relu(self.conv9_2(x))
        conv9_features = x

        x = F.relu(self.conv10_1(x))
        x = F.relu(self.conv10_2(x))
        conv10_features = x

        out = F.relu(self.conv11_1(x))
        conv11_features = F.relu(self.conv11_2(out))

        return conv8_features, conv9_features, conv10_features, conv11_features


class PredConv(nn.Module):

    def __init__(self, n_classes):
        super(PredConv, self).__init__()

        self.n_classes = n_classes
        
        self.loc_conv4_3 = nn.Conv2d(512, 4 * 4, kernel_size=3, padding=1)
        self.loc_conv7 = nn.Conv2d(1024, 6 * 4, kernel_size=3, padding=1)
        self.loc_conv8 = nn.Conv2d(512, 6 * 4, kernel_size=3, padding=1)
        self.loc_conv9 = nn.Conv2d(256, 6 * 4, kernel_size=3, padding=1)
        self.loc_conv10 = nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1)
        self.loc_conv11 = nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1)

        self.class_conv4_3 = nn.Conv2d(512, 4 * n_classes, kernel_size=3, padding=1)
        self.class_conv7 = nn.Conv2d(1024, 6 * n_classes, kernel_size=3, padding=1)
        self.class_conv8 = nn.Conv2d(512, 6 * n_classes, kernel_size=3, padding=1)
        self.class_conv9 = nn.Conv2d(256, 6 * n_classes, kernel_size=3, padding=1)
        self.class_conv10 = nn.Conv2d(256, 4 * n_classes, kernel_size=3, padding=1)
        self.class_conv11 = nn.Conv2d(256, 4 * n_classes, kernel_size=3, padding=1)

        self.init_conv2d()

    def init_conv2d(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv4_3_features, conv7_features, conv8_features, conv9_features, conv10_features, conv11_features):
        batch_size = conv4_3_features.size(0)

        loc_conv4_3 = self.loc_conv4_3(conv4_3_features)
        loc_conv4_3 = loc_conv4_3.permute(0, 2, 3, 1).contiguous()
        loc_conv4_3 = loc_conv4_3.view(batch_size, -1, 4)

        class_conv4_3 = self.class_conv4_3(conv4_3_features)
        class_conv4_3 = class_conv4_3.permute(0, 2, 3, 1).contiguous()
        class_conv4_3 = class_conv4_3.view(batch_size, -1, self.n_classes)

        loc_conv7 = self.loc_conv7(conv7_features)
        loc_conv7 = loc_conv7.permute(0, 2, 3, 1).contiguous()
        loc_conv7 = loc_conv7.view(batch_size, -1, 4)

        loc_conv8 = self.loc_conv8(conv8_features)
        loc_conv8 = loc_conv8.permute(0, 2, 3, 1).contiguous()
        loc_conv8 = loc_conv8.view(batch_size, -1, 4)

        loc_conv9 = self.loc_conv9(conv9_features)
        loc_conv9 = loc_conv9.permute(0, 2, 3, 1).contiguous()
        loc_conv9 = loc_conv9.view(batch_size, -1, 4)

        loc_conv10 = self.loc_conv10(conv10_features)
        loc_conv10 = loc_conv10.permute(0, 2, 3, 1).contiguous()
        loc_conv10 = loc_conv10.view(batch_size, -1, 4)

        loc_conv11 = self.loc_conv11(conv11_features)
        loc_conv11 = loc_conv11.permute(0, 2, 3, 1).contiguous()
        loc_conv11 = loc_conv11.view(batch_size, -1, 4)

        class_conv7 = self.class_conv7(conv7_features)
        class_conv7 = class_conv7.permute(0, 2, 3, 1).contiguous()
        class_conv7 = class_conv7.view(batch_size, -1, self.n_classes)

        class_conv8 = self.class_conv8(conv8_features)
        class_conv8 = class_conv8.permute(0, 2, 3, 1).contiguous()
        class_conv8 = class_conv8.view(batch_size, -1, self.n_classes)

        class_conv9 = self.class_conv9(conv9_features)
        class_conv9 = class_conv9.permute(0, 2, 3, 1).contiguous()
        class_conv9 = class_conv9.view(batch_size, -1, self.n_classes)

        class_conv10 = self.class_conv10(conv10_features)
        class_conv10 = class_conv10.permute(0, 2, 3, 1).contiguous()
        class_conv10 = class_conv10.view(batch_size, -1, self.n_classes)

        class_conv11 = self.class_conv11(conv11_features)
        class_conv11 = class_conv11.permute(0, 2, 3, 1).contiguous()
        class_conv11 = class_conv11.view(batch_size, -1, self.n_classes)

        locs = torch.cat([loc_conv4_3, loc_conv7, loc_conv8, loc_conv9, loc_conv10, loc_conv11], dim=1)
        classes_scores = torch.cat([class_conv4_3, class_conv7, class_conv8, class_conv9, class_conv10, class_conv11], dim=1)

        return locs, classes_scores


class SSD300(nn.Module):
    def __init__(self, n_labeles):
        super(SSD300, self).__init__()

        self.n_labeles = n_labeles

        self.base = VGG16()
        self.aux_convs = AuxConv()
        self.pred_convs = PredConv(n_labeles)

        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))
        nn.init.constant_(self.rescale_factors, 20)

        self.priors_cxcy = self.create_prior_cxcy()

    def forward(self, image):
        conv4_3_features, conv7_features = self.base(image)

        norm = conv4_3_features.pow(2).sum(dim=1, keepdim=True).sqrt()
        conv4_3_features = conv4_3_features / norm
        conv4_3_features = conv4_3_features * self.rescale_factors
        conv8_features, conv9_features, conv10_features, conv11_features = \
        self.aux_convs(conv7_features)

        locs, classes_scores = self.pred_convs(conv4_3_features, conv7_features, conv8_features, conv9_features,
                                               conv10_features, conv11_features)

        return locs, classes_scores

    def create_prior_cxcy(self):
        fmap_dim = {'conv4_3': 38,
                     'conv7': 19,
                     'conv8': 10,
                     'conv9': 5,
                     'conv10': 3,
                     'conv11': 1}
        object_scale = {'conv4_3': 0.1,
                      'conv7': 0.2,
                      'conv8': 0.375,
                      'conv9': 0.55,
                      'conv10': 0.725,
                      'conv11': 0.9}
        aspect_ratio = {'conv4_3': [1., 2., 0.5],
                         'conv7': [1., 2., 3., 0.5, .333],
                         'conv8': [1., 2., 3., 0.5, .333],
                         'conv9': [1., 2., 3., 0.5, .333],
                         'conv10': [1., 2., 0.5],
                         'conv11': [1., 2., 0.5]}

        fmaps = list(fmap_dim.keys())
        prior_cxcys = []

        for midx, fmap in enumerate(fmaps):
            for i in range(fmap_dim[fmap]):
                for j in range(fmap_dim[fmap]):
                    cx = (j + 0.5) / fmap_dim[fmap]
                    cy = (i + 0.5) / fmap_dim[fmap]
                    for ratio in aspect_ratio[fmap]:
                        prior_cxcys.append([cx, cy, object_scale[fmap] * sqrt(ratio), object_scale[fmap] / sqrt(ratio)])
                        if ratio == 1.0:
                            try:
                                add_scale = sqrt(object_scale[fmap] * object_scale[fmaps[midx + 1]])
                            except IndexError:
                                add_scale = 1.0
                            prior_cxcys.append([cx, cy, add_scale, add_scale])

        prior_cxcys = torch.FloatTensor(prior_cxcys).to(device)
        prior_cxcys.clamp_(0, 1)

        return prior_cxcys

    def detect_objects(self, pred_locs, pred_scores, min_score, max_overlap, top_k):
        batch_size = pred_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        pred_scores = F.softmax(pred_scores, dim=2)

        all_images_boxes = []
        all_images_labels = []
        all_images_scores = []

        #assert n_priors == pred_locs.size(1) == pred_scores.size(1)

        for i in range(batch_size):
            cpoint = utils.gcxgcy_to_cxcy(pred_locs[i], self.priors_cxcy)
            point = torch.cat([cpoint[:, :2] - (cpoint[:, 2:] / 2), cpoint[:, :2] + (cpoint[:, 2:] / 2)], 1)
            decoded_locs = point

            image_boxes = []
            image_labels = []
            image_scores = []

            max_scores, best_label = pred_scores[i].max(dim=1)

            for c in range(1, self.n_labeles):
                class_scores = pred_scores[i][:, c]
                score_above_min_score = class_scores > min_score
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]
                class_decoded_locs = decoded_locs[score_above_min_score]

                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)
                class_decoded_locs = class_decoded_locs[sort_ind]

                overlap = utils.find_jaccard_overlap(class_decoded_locs, class_decoded_locs)

                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)

                for box in range(class_decoded_locs.size(0)):
                    if suppress[box] == 1:
                        continue

                    suppress = torch.max(suppress, overlap[box] > max_overlap)
                    suppress[box] = 0

                image_boxes.append(class_decoded_locs[1 - suppress])
                image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
                image_scores.append(class_scores[1 - suppress])

            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

            image_boxes = torch.cat(image_boxes, dim=0)
            image_labels = torch.cat(image_labels, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            n_objects = image_scores.size(0)

            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]
                image_boxes = image_boxes[sort_ind][:top_k]
                image_labels = image_labels[sort_ind][:top_k]

            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores

class MultiBoxLoss(nn.Module):
    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        point = torch.cat([priors_cxcy[:, :2] - (priors_cxcy[:, 2:] / 2),
         priors_cxcy[:, :2] + (priors_cxcy[:, 2:] / 2)], 1)
        self.priors_xy = point
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)

    def forward(self, pred_locs, pred_scores, boxes, labels):
        batch_size = pred_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = pred_scores.size(2)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)

        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = utils.find_jaccard_overlap(boxes[i], self.priors_xy)
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)

            _, prior_for_each_object = overlap.max(dim=1)

            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)
            overlap_for_each_prior[prior_for_each_object] = 1.

            label_for_each_prior = labels[i][object_for_each_prior]
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0

            true_classes[i] = label_for_each_prior

            point = boxes[i][object_for_each_prior]
            cpoint = torch.cat([(point[:, 2:] + point[:, :2]) / 2, point[:, 2:] - point[:, :2]], 1)
            true_locs[i] = utils.cxcy_to_gcxgcy(cpoint, self.priors_cxcy)

        positive_priors = true_classes != 0

        loc_loss = self.smooth_l1(pred_locs[positive_priors], true_locs[positive_priors])

        n_positives = positive_priors.sum(dim=1)
        n_hard_negatives = self.neg_pos_ratio * n_positives

        conf_loss_all = self.cross_entropy(pred_scores.view(-1, n_classes), true_classes.view(-1))
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)

        conf_loss_pos = conf_loss_all[positive_priors]
        conf_loss_neg = conf_loss_all.clone()
        conf_loss_neg[positive_priors] = 0.
        conf_loss_neg, idx = conf_loss_neg.sort(dim=1, descending=True)
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]

        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()

        return conf_loss + self.alpha * loc_loss
