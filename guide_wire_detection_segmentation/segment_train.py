import torch
from ultralytics import YOLO


def train():
    torch.cuda.empty_cache()

    model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)

    results = model.train(data='wire_data.yaml',
                          workers=16, device=0, batch=12,
                          epochs=500, imgsz=640, half=False, amp=False, patience=0,
                          )
    model.val()  # It'll automatically evaluate the data you trained.
    model.test()  # It'll automatically evaluate the data you trained.
    success = model.export(format='onnx')


if __name__ == '__main__':
    # set_data()
    train()
