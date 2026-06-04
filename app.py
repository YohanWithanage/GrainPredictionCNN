import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

class GrainCNN(nn.Module):
    def __init__(self, img_size=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.regressor(x)
        return x.squeeze(1)

# Load model
device = torch.device('cpu')
model = GrainCNN(img_size=64)
model.load_state_dict(torch.load('grain_model_weights.pth', map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

def predict(image):
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        prediction = model(tensor).item()
    return f"{prediction:.2f} µm"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Microstructure Image"),
    outputs=gr.Text(label="Predicted Grain Size"),
    title="GrainCNN — Grain Size Predictor",
    description="Upload a microstructure image to predict mean grain size in µm.",
)

demo.launch()
