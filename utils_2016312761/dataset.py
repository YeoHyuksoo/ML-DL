import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils_2016312761 import utils

import random
import torchvision.transforms.functional as FT

def transform(image, boxes, labels, difficulties, split):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    image_trans = image
    boxes_trans = boxes
    labels_trans = labels
    difficulties_trans = difficulties

    if split == 'train':
        image_trans = utils.photometric_distortion(image_trans)
        image_trans = FT.to_tensor(image_trans)

        if random.random() < 0.5:
            image_trans, boxes_trans = utils.expand(mean, image_trans, boxes)

        image_trans, boxes_trans, labels_trans, difficulties_trans = utils.randomCrop(image_trans, boxes_trans, labels_trans, difficulties_trans)
        image_trans = FT.to_pil_image(image_trans)

        if random.random() < 0.5:
            image_trans, boxes_trans = utils.horizontalflip(image_trans, boxes_trans)
    
    image_trans, boxes_trans = utils.resize((300, 300), image_trans, boxes_trans)
    image_trans = FT.to_tensor(image_trans)
    image_trans = FT.normalize(image_trans, mean=mean, std=std)

    return image_trans, boxes_trans, labels_trans, difficulties_trans

class VOCDataset(Dataset):
    def __init__(self, split): 
        self.split = split

        with open(os.path.join('./', split + '_images.json'), 'r') as filename:
            self.images = json.load(filename)
        with open(os.path.join('./', split + '_info.json'), 'r') as filename:
            self.objects = json.load(filename)

    def __getitem__(self, i):
        img = Image.open(self.images[i], mode='r')
        img = img.convert('RGB')

        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['minmax'])
        labels = torch.LongTensor(objects['labels'])
        difficulties = torch.ByteTensor(objects['difficulties'])
        img, boxes, labels, difficulties = transform(img, boxes, labels, difficulties, self.split)

        return img, boxes, labels, difficulties

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        images = []
        boxes = []
        labels = []
        difficulties = []

        for comp in batch:
            images.append(comp[0])
            boxes.append(comp[1])
            labels.append(comp[2])
            difficulties.append(comp[3])

        return torch.stack(images, dim=0), boxes, labels, difficulties
