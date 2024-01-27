import torch
import torchvision
from ultralytics import YOLO

# An instance of your model.
model = YOLO('  runs/segment/train/weights/best.pt').to(device=0)

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)