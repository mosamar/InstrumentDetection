import os
import shutil

import torch
from sklearn.model_selection import train_test_split
from ultralytics import YOLO


def train():
    torch.cuda.empty_cache()

    model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)

    results = model.train(data='vessel_model.yaml', device=0, batch=10,
                          epochs=1000, imgsz=640, half=False, amp=False, patience=0
                          )
    model.val()  # It'll automatically evaluate the data you trained.
    print(results)
    success = model.export(format='onnx')


if __name__ == '__main__':
    train()
