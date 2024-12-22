import streamlit as st
from transformers import AutoTokenizer, AutoModelForImageCaptioning
from PIL import Image
from PIL import Image
from diffusers import StableDiffusionPipeline
from diffusers import DiffusionPipeline
import requests
import io
import torch
from tempfile import NamedTemporaryFile


# Load the model and tokenizer
model_name = "MODEL_NAME_FROM_HUGGING_FACE"  # Specify the model name from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForImageCaptioning.from_pretrained(model_name)

# Function to generate image from text
def generate_image(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
    outputs = model.generate(**inputs)
    image = Image.open(outputs[0])
    return image

# Streamlit UI
st.title("Text to Image Generation")

text_input = st.text_input("Enter a description for the image")
if text_input:
    st.write("Generating image...")
    generated_image = generate_image(text_input)
    st.image(generated_image, caption="Generated Image")
