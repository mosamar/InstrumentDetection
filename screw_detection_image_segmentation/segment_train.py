import torch
from ultralytics import YOLO


def train():
    torch.cuda.empty_cache()

    model = YOLO('runs/segment/train4/weights/best.pt')  # load a pretrained model (recommended for training)

    # results = model.train(data='segmentation_data.yaml',
    #                       workers=16, device=0, batch=16,
    #                       epochs=1000, imgsz=640, half=False, amp=False, patience=0,
    #                       )
    model.val()  # It'll automatically evaluate the data you trained.
    # print(results)
    success = model.export(format='onnx')


if __name__ == '__main__':
    # set_data()
    train()
