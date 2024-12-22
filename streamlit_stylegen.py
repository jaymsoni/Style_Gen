import streamlit as st
from PIL import Image
from diffusers import StableDiffusionPipeline
from diffusers import DiffusionPipeline
import requests
import io
import torch
from tempfile import NamedTemporaryFile

API_URL = "https://api-inference.huggingface.co/models/jaysoni/sd_xl_lora"
headers = {"Authorization": "Bearer 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"}

def query(payload):
  response = requests.post(API_URL, headers=headers, json=payload)
  return response.content

@st.cache_resource()
def load_pipeline():
  return DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-0.9",torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to("cuda")

if "pipeline" not in st.session_state.keys():
  st.session_state["pipeline"] = load_pipeline()

pipe = st.session_state["pipeline"]

def generate_image(prompt, num_inference_steps=5):
  print("world")
  with NamedTemporaryFile(suffix=".jpg") as temp_file:  # Create temporary file
    image_bytes = query({
      "inputs": prompt,
    })
    temp_file.write(image_bytes)
    temp_file.flush()  # Ensure data is written to disk

    image = Image.open(temp_file.name)  # Open from temporary file path

  images = pipe(prompt=prompt, image=image).images[0]
  print(images)
  # images
  return images


# Text input and image size selection
text_prompt = st.text_area("Describe the image you want:")
image_size_str = st.selectbox("Image size", ("256x256", "512x512", "1024x1024"))
image_size = tuple(int(x) for x in image_size_str.split('x'))  # Parse image size

if st.button("Generate Image"):
  try:
    image = generate_image(text_prompt, num_inference_steps=20)  # Adjust steps as needed
    st.image(image)
    st.download_button("Download Image", image.tobytes(), f"image_{text_prompt.replace(' ','_')}.png")  # Download option with byte conversion
  except Exception as e:
    st.error(f"An error occurred: {e}")
