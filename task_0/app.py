import io
from PIL import Image
import streamlit as st

bg_url = "https://images.unsplash.com/photo-1477132394330-d2376dc4c091?q=80&w=1611&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
st.markdown(
    f"""
    <style>
    .stApp {{
        background: url("{bg_url}") center/cover no-repeat fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)
st.write("# Lab 1 - Image Uploader")

uploaded_file = st.file_uploader(
    "Pick a file", type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
  try:
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    st.image(image, caption="Uploaded Image", width="stretch")
  except Exception as e:
    st.error(f"Could not open image: {e}")
else:
  st.info("Please upload an image file.")