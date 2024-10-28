from utils_2016312761 import utils
from utils_2016312761 import dataset
from tqdm import tqdm
from pprint import PrettyPrinter
import torch

import numpy as np
from sklearn.metrics import average_precision_score

batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_model = './model.pth.tar'

save_model = torch.load(save_model)
model = save_model['model']
model = model.to(device)

test_dataset = dataset.VOCDataset(split='test')
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=test_dataset.collate_fn, num_workers=2, pin_memory=True)

def evaluate(test_loader, model):
    model.eval()

    detect_boxes = []
    detect_labels = []
    detect_scores = []
    output_boxes = []
    output_labels = []
    output_difficulties = []

    with torch.no_grad():
        for i, (images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)
            pred_locations, pred_scores = model(images)

            batch_boxes, batch_labels, batch_scores = model.detect_objects(pred_locations, pred_scores,
                                                                                       min_score=0.01, max_overlap=0.45,
                                                                                       top_k=200)
            boxes = [box.to(device) for box in boxes]
            labels = [label.to(device) for label in labels]
            difficulties = [difficult.to(device) for difficult in difficulties]

            detect_boxes.extend(batch_boxes)
            detect_labels.extend(batch_labels)
            detect_scores.extend(batch_scores)
            output_boxes.extend(boxes)
            output_labels.extend(labels)
            output_difficulties.extend(difficulties)

        APs, mAP = utils.calculate_mAP(detect_boxes, detect_labels, detect_scores, output_boxes, output_labels, output_difficulties)
    
    pp = PrettyPrinter()
    pp.pprint(APs)
    print('\nmAP: %.3f' % mAP)

if __name__ == '__main__':
    evaluate(test_loader, model)
