from ultralytics import YOLO

# Create a new YOLO model from scratch
if __name__ == '__main__':

    # with open('./screw_detection_phantom_data.yaml', 'r') as f:
    #     print(f.read())

    model = YOLO('yolov8n.yaml')

    # Load a pretrained YOLO model (recommended for training)
    # model = YOLO('yolov8n.pt')

    results = model.train(data='screw_detection_phantom_data.yaml', workers=16, device=0, batch=16,
                          epochs=100, imgsz=640, half=False, amp=False)
    model.val()  # It'll automatically evaluate the data you trained.
    print(results)
    success = model.export(format='onnx')
