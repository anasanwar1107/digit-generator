import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ===== Model Class =====

class CVAE(nn.Module):
    def __init__(self, input_dim=784, label_dim=10, latent_dim=20):
        super(CVAE, self).__init__()
        self.fc1 = nn.Linear(input_dim + label_dim, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim + label_dim, 400)
        self.fc4 = nn.Linear(400, input_dim)

    def encode(self, x, y):
        h1 = F.relu(self.fc1(torch.cat([x, y], dim=1)))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        h3 = F.relu(self.fc3(torch.cat([z, y], dim=1)))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

# ===== Utils =====

def one_hot(label, num_classes=10):
    return torch.eye(num_classes)[label]

@st.cache_resource
def load_model():
    model = CVAE()
    model.load_state_dict(torch.load("cvae_mnist.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

def generate_images(model, digit, num_images=5):
    y = one_hot(torch.tensor([digit] * num_images)).float()
    z = torch.randn(num_images, 20)
    with torch.no_grad():
        generated = model.decode(z, y).view(-1, 28, 28).numpy()
    return generated

# ===== Streamlit UI =====

st.set_page_config(page_title="Digit Generator", layout="centered")

# ---- Title ----
st.markdown("<h1 style='text-align: center;'>Handwritten Digit Image Generator</h1>", unsafe_allow_html=True)

# ---- Instructions ----
st.markdown("### Generate synthetic MNIST-like images using your trained model.")

# ---- Input ----
st.markdown("Choose a digit to generate (0–9):")
digit = st.selectbox("", list(range(10)), index=0)

# ---- Generate Button ----
if st.button("Generate Images"):
    model = load_model()
    imgs = generate_images(model, digit)

    st.markdown(f"### Generated images of digit {digit}")
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.image(imgs[i], width=100, caption=f"Sample {i+1}", clamp=True)
