import torch
import json
import os
import random
import torchvision.transforms.functional as FT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

labels_vocab = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
label_map = {k: v + 1 for v, k in enumerate(labels_vocab)}
label_map['background'] = 0
reverse_label_map = {v: k for k, v in label_map.items()}

def decimation(tensor, m):
    length = len(m)
    if tensor.dim() != length:
      print("decimation error!!")
    
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(dim=d, index=torch.arange(start=0, end=tensor.size(d), step=m[d]).long())

    return tensor

def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],
                      torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)

def save_model(epoch, model, optimizer):
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer}
    filename = 'model.pth.tar'
    torch.save(state, filename)

def calculate_mAP(detect_boxes, detect_labels, detect_scores, output_boxes, output_labels, output_difficulties):
    n_labeles = len(label_map)

    output_images = []
    for i in range(len(output_labels)):
        output_images.extend([i] * output_labels[i].size(0))
    output_images = torch.LongTensor(output_images).to(device)
    output_boxes = torch.cat(output_boxes, dim=0)
    output_labels = torch.cat(output_labels, dim=0)
    output_difficulties = torch.cat(output_difficulties, dim=0)

    if output_images.size(0) != output_boxes.size(0) or output_boxes.size(0) != output_labels.size(0):
      print("true data error!!")

    detect_images = []
    for i in range(len(detect_labels)):
        detect_images.extend([i] * detect_labels[i].size(0))
    detect_images = torch.LongTensor(detect_images).to(device)
    detect_boxes = torch.cat(detect_boxes, dim=0)
    detect_labels = torch.cat(detect_labels, dim=0)
    detect_scores = torch.cat(detect_scores, dim=0)

    ap = torch.zeros((n_labeles - 1), dtype=torch.float)
    for c in range(1, n_labeles):
        output_labeled_images = output_images[output_labels == c]
        output_labeled_boxes = output_boxes[output_labels == c]
        output_labeled_difficulties = output_difficulties[output_labels == c]
        n_easy_labeled_objects = (1 - output_labeled_difficulties).sum().item()

        output_labeled_boxes_detected = torch.zeros((output_labeled_difficulties.size(0)), dtype=torch.uint8).to(device)

        detect_labeled_images = detect_images[detect_labels == c]
        detect_labeled_boxes = detect_boxes[detect_labels == c]
        detect_labeled_scores = detect_scores[detect_labels == c]
        n_labeled_detections = detect_labeled_boxes.size(0)
        if n_labeled_detections == 0:
            continue

        detect_labeled_scores, sort_ind = torch.sort(detect_labeled_scores, dim=0, descending=True)
        detect_labeled_images = detect_labeled_images[sort_ind]
        detect_labeled_boxes = detect_labeled_boxes[sort_ind]

        true_positives = torch.zeros((n_labeled_detections), dtype=torch.float).to(device)
        false_positives = torch.zeros((n_labeled_detections), dtype=torch.float).to(device)
        for d in range(n_labeled_detections):
            this_detection_box = detect_labeled_boxes[d].unsqueeze(0)
            this_image = detect_labeled_images[d]

            object_boxes = output_labeled_boxes[output_labeled_images == this_image]
            object_difficulties = output_labeled_difficulties[output_labeled_images == this_image]
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue

            overlaps = find_jaccard_overlap(this_detection_box, object_boxes)
            max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)

            original_ind = torch.LongTensor(range(output_labeled_boxes.size(0)))[output_labeled_images == this_image][ind]

            if max_overlap.item() > 0.5:
                if object_difficulties[ind] == 0:
                    if output_labeled_boxes_detected[original_ind] == 0:
                        true_positives[d] = 1
                        output_labeled_boxes_detected[original_ind] = 1
                    else:
                        false_positives[d] = 1
            else:
                false_positives[d] = 1

        cumul_true_positives = torch.cumsum(true_positives, dim=0)
        cumul_false_positives = torch.cumsum(false_positives, dim=0)
        cumul_precision = cumul_true_positives / (cumul_true_positives + cumul_false_positives + 1e-10)
        cumul_recall = cumul_true_positives / n_easy_labeled_objects

        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)
        for i, thres in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= thres
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        ap[c - 1] = precisions.mean()

    mean_average_precision = ap.mean().item()
    ap = {reverse_label_map[c + 1]: v for c, v in enumerate(ap.tolist())}

    return ap, mean_average_precision

def find_intersection(set1, set2):
    min_bound = torch.max(set1[:, :2].unsqueeze(1), set2[:, :2].unsqueeze(0))
    max_bound = torch.min(set1[:, 2:].unsqueeze(1), set2[:, 2:].unsqueeze(0))

    intersection = torch.clamp(max_bound - min_bound, min=0)
    return intersection[:, :, 0] * intersection[:, :, 1]

def find_jaccard_overlap(set1, set2):
    intersection = find_intersection(set1, set2)

    area1 = (set1[:, 2] - set1[:, 0]) * (set1[:, 3] - set1[:, 1])
    area2 = (set2[:, 2] - set2[:, 0]) * (set2[:, 3] - set2[:, 1])
    union = area1.unsqueeze(1) + area2.unsqueeze(0)
    union = union - intersection

    return intersection / union

def expand(fill, image, boxes):
    height = image.size(1)
    width = image.size(2)
    filler = torch.FloatTensor(fill)
    
    rand_scale = random.uniform(1, 4)
    scaled_height = int(rand_scale * height)
    scaled_width = int(rand_scale * width)

    new_image = torch.ones((3, scaled_height, scaled_width), dtype=torch.float) * filler.unsqueeze(1).unsqueeze(1)

    left = random.randint(0, scaled_width - width)
    top = random.randint(0, scaled_height - height)
    new_image[:, top:top + height, left:left + width] = image
    new_boxes = boxes + torch.FloatTensor([left, top, left, top]).unsqueeze(0)

    return new_image, new_boxes

def randomCrop(image, boxes, labels, difficulties):
    height = image.size(1)
    width = image.size(2)

    while True:
        min_overlap = random.choice([0., .1, .3, .5, .7, .9, None])

        if min_overlap is None:
            return image, boxes, labels, difficulties

        for trial in range(50):
            scaled_height = random.uniform(0.3, 1)
            scaled_height = int(scaled_height * height)
            scaled_width = random.uniform(0.3, 1)
            scaled_width = int(scaled_width * width)

            if scaled_height / scaled_width <= 0.5 or scaled_height / scaled_width >= 2:
                continue

            left = random.randint(0, width - scaled_width)
            right = left + scaled_width
            top = random.randint(0, height - scaled_height)
            bottom = top + scaled_height
            crop = torch.FloatTensor([left, top, right, bottom])

            overlap = find_jaccard_overlap(crop.unsqueeze(0), boxes).squeeze(0)
            if overlap.max().item() < min_overlap:
                continue

            cropped_image = image[:, top:bottom, left:right]
            center_point = (boxes[:, :2] + boxes[:, 2:]) / 2.0
            center_crop = (center_point[:, 0] > left) * (center_point[:, 0] < right) * (center_point[:, 1] > top) * (center_point[:, 1] < bottom)

            if not center_crop.any():
                continue

            cropped_boxes = boxes[center_crop, :]
            cropped_boxes[:, :2] = torch.max(cropped_boxes[:, :2], crop[:2])
            cropped_boxes[:, :2] -= crop[:2]
            cropped_boxes[:, 2:] = torch.min(cropped_boxes[:, 2:], crop[2:])
            cropped_boxes[:, 2:] -= crop[:2]
            cropped_labels = labels[center_crop]
            cropped_difficulties = difficulties[center_crop]

            return cropped_image, cropped_boxes, cropped_labels, cropped_difficulties

def horizontalflip(image, boxes):
    boxes_cp = boxes
    boxes_cp[:, 0] = image.width - boxes[:, 0] - 1
    boxes_cp[:, 2] = image.width - boxes[:, 2] - 1
    boxes = boxes_cp[:, [2, 1, 0, 3]]

    return FT.hflip(image), boxes

def resize(dims, image, boxes, return_percent_coords=True):
    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    resized_boxes = boxes / old_dims

    if not return_percent_coords:
        height = dims[0]
        width = dims[1]
        new_dims = torch.FloatTensor([width, height, width, height]).unsqueeze(0)
        resized_boxes = resized_boxes * new_dims

    return FT.resize(image, dims), resized_boxes

def photometric_distortion(image):
    distortions = [FT.adjust_brightness,
                   FT.adjust_contrast,
                   FT.adjust_hue,
                   FT.adjust_saturation]

    random.shuffle(distortions)

    for distort in distortions:
        if random.random() < 0.5:
            if distort.__name__ == 'adjust_hue':
                image = distort(image, random.uniform(-18 / 255.0, 18 / 255.0))
            else:
                image = distort(image, random.uniform(0.5, 1.5))

    return image