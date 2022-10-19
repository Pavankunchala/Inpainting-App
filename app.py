from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from torchvision.transforms import ToTensor
import torch
import numpy as np
import cv2
import aotgan.model.aotgan as net

@st.cache
def load_model(model_name):
    model = net.InpaintGenerator.from_pretrained(model_name)
    return model

def postprocess(image):
    image = torch.clamp(image, -1., 1.)
    image = (image + 1) / 2.0 * 255.0
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy().astype(np.uint8)
    return image

def infer(img, mask):
    with torch.no_grad():
        img_cv = cv2.resize(np.array(img), (512, 512))  # Fixing everything to 512 x 512 for this demo.
        img_tensor = (ToTensor()(img_cv) * 2.0 - 1.0).unsqueeze(0)
        mask_tensor = (ToTensor()(mask.astype(np.uint8))).unsqueeze(0)
        masked_tensor = (img_tensor * (1 - mask_tensor).float()) + mask_tensor
        pred_tensor = model(masked_tensor, mask_tensor)
        comp_tensor = (pred_tensor * mask_tensor + img_tensor * (1 - mask_tensor))
        comp_np = postprocess(comp_tensor[0])

        return comp_np

st.title("AOTGAN Image Inpainting")

st.sidebar.title('Inpainting ')

stroke_width = 8
stroke_color = "#FFF"
bg_color = "#000"
bg_image = st.sidebar.file_uploader("Image:", type=["png", "jpg", "jpeg"])

drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "rect", "circle")
)

model_name = st.sidebar.selectbox(
    "Select model:", ("NimaBoscarino/aot-gan-celebahq", "NimaBoscarino/aot-gan-places2")
)
model = load_model(model_name)

bg_image = Image.open(bg_image) if bg_image else Image.open('c1.jpeg')

st.subheader("Draw on the image to erase features. The inpainted result will be generated and displayed below.")
canvas_result = st_canvas(
    fill_color="rgb(255, 255, 255)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    background_image=bg_image,
    update_streamlit=True,
    height=512,
    width=512,
    drawing_mode=drawing_mode,
    key="canvas",
)
    
if canvas_result.image_data is not None and bg_image and len(canvas_result.json_data["objects"]) > 0:
    result = infer(bg_image, canvas_result.image_data[:, :, 3])
    st.image(result)

