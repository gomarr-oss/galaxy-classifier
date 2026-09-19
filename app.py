import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


CLASS_NAMES = [
    "Disturbed", "Merging", "Round Smooth",
    "In-between Smooth", "Cigar Smooth", "Barred Spiral",
    "Unbarred Tight Spiral", " Unbarred Loose Spiral",
    "Edge-on without Bulge", "Edge-on with Bulge"
]

@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load("models/best_resnet_model.pt", 
                                     map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# STREAMLIT
# App title
st.title("🌌 Galaxy Morphology Classifier ✨")
st.write("Upload a galaxy image, and the model will classify its morphological type!")

# File uploader
uploaded_file = st.file_uploader("Choose a galaxy image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Galaxy", width=300)

    # Classify button
    if st.button("Classify Galaxy 🔭"):
        with st.spinner("Analyzing galaxy..."):
            # Preprocess and predict
            tensor = preprocess_image(image)

            with torch.no_grad():
                outputs = model(tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                predicted_class = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_class].item() * 100

            # Display results
            st.success(f"**{CLASS_NAMES[predicted_class]}** ({confidence:.1f}% confidence)")

            # Bar chart
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.barh(CLASS_NAMES, probabilities.detach().numpy() * 100)
            ax.set_xlabel("Confidence (%)")
            ax.set_title("Prediction Confidence by Galaxy Type")
            st.pyplot(fig)
