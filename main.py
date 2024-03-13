import streamlit as st
from PIL import Image, ImageDraw

from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(
    page_title="Get Pixel Position by Mouse Clicks",
    layout="wide",
)

st.title("Get Pixel Position by Mouse Clicks")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Click on the image to get pixel position")

    value = streamlit_image_coordinates(image, key="pil")

    if value is not None:
        x, y = value["x"], value["y"]

        # Draw a small circle at the clicked position
        draw = ImageDraw.Draw(image)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="red", outline="red")

        st.image(image, caption="Uploaded Image with Clicked Position", use_column_width=True)
        st.write(f"Pixel position: ({x}, {y})")
