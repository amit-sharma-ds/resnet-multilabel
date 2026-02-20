# src/app.py
import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to model (relative to src/)
model_path = "multilabel-resnet/model/multilabel_model.pth"

# Load ResNet18 model
model = models.resnet18(pretrained=False)  # do not load ImageNet weights, just architecture
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 4)  # 4 attributes
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Prediction function
def predict_attributes(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
    
    attributes = [f"Attr{i+1}" for i, val in enumerate(preds[0]) if val == 1]
    return ", ".join(attributes) if attributes else "No attributes detected"

# Gradio interface
iface = gr.Interface(
    fn=predict_attributes,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="Predicted Attributes"),
    title="Multilabel Attribute Prediction",
    description="Upload an image and get the predicted attributes using ResNet18."
)

if __name__ == "__main__":
    iface.launch()