import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from utils_2016312761 import dataset
from utils_2016312761 import utils
from utils_2016312761 import model as md
import os
import xml.etree.ElementTree as ET
import json

cudnn.benchmark = True

labels_vocab = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
label_map = {k: v + 1 for v, k in enumerate(labels_vocab)}
label_map['background'] = 0

n_labeles = len(label_map)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#################### EDIT HERE ####################
batch_size = 32
iterations = 60000
print_every = 100
learning_rate = 0.001
momentum = 0.9
weight_decay = 5e-4
###################################################

def parse_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    minmax = []
    labels = []
    difficulties = []

    info_dic = {}

    for object in root.iter('object'):
        difficult = int(object.find('difficult').text == '1')

        label = object.find('name').text.lower().strip()
        if label not in label_map:
            continue

        bndbox = object.find('bndbox')
        xmin = int(bndbox.find('xmin').text) - 1
        ymin = int(bndbox.find('ymin').text) - 1
        xmax = int(bndbox.find('xmax').text) - 1
        ymax = int(bndbox.find('ymax').text) - 1

        minmax.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])
        difficulties.append(difficult)

    info_dic['minmax'] = minmax
    info_dic['labels'] = labels
    info_dic['difficulties'] = difficulties

    return info_dic

def create_data_lists(train_path, test_path, output_folder):

    train_images = []
    train_info = []

    with open(os.path.join(train_path, 'ImageSets/Main/trainval.txt')) as f:
        filenames = f.read().splitlines()

    for filename in filenames:
        info = parse_annotation(os.path.join(train_path, 'Annotations', filename + '.xml'))
        if len(info['minmax']) == 0:
            print(filename)
            continue
        train_info.append(info)
        train_images.append(os.path.join(train_path, 'JPEGImages', filename + '.jpg'))

    file_list = os.listdir(train_path + '/Annotations')
    new_file_list = []
    for i in range(len(file_list)):
      if file_list[i][:4] == '2007':
        new_file_list.append(file_list[i][:11])
    del file_list
    train_filename = []

    for filename in new_file_list:
        info = parse_annotation(os.path.join(train_path, 'Annotations', filename + '.xml'))
        if len(info['minmax']) == 0:
            continue
        train_info.append(info)
        train_images.append(os.path.join(train_path, 'JPEGImages', filename + '.jpg'))
        train_filename.append(filename)

    with open(os.path.join(output_folder, 'train_images.json'), 'w') as trainfile:
        json.dump(train_images, trainfile)
    with open(os.path.join(output_folder, 'train_info.json'), 'w') as trainfile:
        json.dump(train_info, trainfile)

    test_images = []
    test_info = []

    with open(os.path.join(test_path, 'ImageSets/Main/test.txt')) as f:
        filenames = f.read().splitlines()

    #n = 0
    for filename in filenames:
        if '2007_'+filename in train_filename:
            continue
        info = parse_annotation(os.path.join(test_path, 'Annotations', filename + '.xml'))
        if len(info) == 0:
            continue
        test_info.append(info)
        test_images.append(os.path.join(test_path, 'JPEGImages', filename + '.jpg'))
        #n+=1
        #if n==192:
        #  break

    with open(os.path.join(output_folder, 'test_images.json'), 'w') as testfile:
        json.dump(test_images, testfile)
    with open(os.path.join(output_folder, 'test_info.json'), 'w') as testfile:
        json.dump(test_info, testfile)


def main():
    create_data_lists(train_path='VOCtrainval/VOCdevkit/VOC2012',
                      test_path='VOCtest/VOCdevkit/VOC2007',
                      output_folder='./')
    
    decay_lr1 = int(iterations*0.75)
    decay_lr2 = int(iterations*0.85)
    model = md.SSD300(n_labeles=n_labeles)
    
    bias = []
    parameters = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if name.endswith('.bias'):
                bias.append(parameter)
            else:
                parameters.append(parameter)
    optimizer = torch.optim.SGD(params=[{'params': bias, 'lr': 2 * learning_rate}, {'params': parameters}],
                                lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    model = model.to(device)
    criterion = md.MultiBoxLoss(priors_cxcy = model.priors_cxcy).to(device)
    train_dataset = dataset.VOCDataset(split='train')
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=2,
                                               pin_memory=True)

    epochs = int(iterations / int(len(train_dataset) / batch_size))
    print("epochs:", epochs)
    decay_lr1 = int(decay_lr1 / int(len(train_dataset) / batch_size))
    decay_lr2 = int(decay_lr2 / int(len(train_dataset) / batch_size))
    print("decay_lr points:", decay_lr1, decay_lr2)

    for epoch in range(epochs):
        if epoch == decay_lr1:
            for parameters in optimizer.param_groups:
                parameters['lr'] = parameters['lr'] * 0.9

        train(trainloader=trainloader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)

        utils.save_model(epoch, model, optimizer)
    
    torch.save(model, './model_2016312761.pt')

def train(trainloader, model, criterion, optimizer, epoch):
    model.train()

    start = time.time()

    for i, (images, boxes, labels, difficulties) in enumerate(trainloader):
        images = images.to(device)
        boxes = [box.to(device) for box in boxes]
        labels = [label.to(device) for label in labels]

        predicted_locs, predicted_scores = model(images)

        loss = criterion(predicted_locs, predicted_scores, boxes, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % print_every == 0:
            print('Epoch: {0}, [{1}/{2}]\t Time: {3}s \tLoss: {4}'
            .format(epoch, i, len(trainloader), time.time()-start, loss.item()))

if __name__ == '__main__':
    main()