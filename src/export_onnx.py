import torch
import torch.nn as nn
from torchvision import models

# 1. Re-initialize the model architecture (Must match training exactly)
model = models.resnet50(weights=None) # We don't need to download weights, we have our own
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# 2. Load your trained weights
# map_location='cpu' ensures it loads even if you trained on GPU/MPS
model.load_state_dict(torch.load("../models/defect_model.pth", map_location='cpu'))
model.eval() # Set to evaluation mode (turns off Dropout, Batchnorm updates)

# 3. Create dummy input
# Shape: (Batch_Size, Channels, Height, Width) -> (1, 3, 256, 256)
dummy_input = torch.randn(1, 3, 256, 256)

# 4. Export
output_file = "../models/defect_detector.onnx"
torch.onnx.export(model,
                  dummy_input,
                  output_file,
                  input_names=['input'],
                  output_names=['output'],
                  opset_version=11) # Version 11 is standard and stable

print(f"Success! Model exported to {output_file}")