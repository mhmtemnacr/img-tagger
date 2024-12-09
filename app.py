import streamlit as st
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration
from keybert import KeyBERT


try:
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
except Exception as e:
    st.error(f"Error loading model: {e}")

kw_model = KeyBERT()

st.title("Image Tagger")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image.", use_container_width=True)

    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(
        **inputs,
        max_length=50,
        num_beams=5
    )
    caption = processor.decode(out[0], skip_special_tokens=True)

    keywords = kw_model.extract_keywords(caption, top_n=20)

    st.subheader("Generated Caption", divider="gray")
    st.write(caption)
    st.subheader("Keywords", divider="gray")
    st.write(", ".join([kw[0] for kw in keywords]))
