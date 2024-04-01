import numpy as np
from PIL import Image
import streamlit as st
import matplotlib

def load_and_display_mask_with_pil(mask_file):
    """
    Displays a mask using PIL, with normalization if necessary.

    Parameters:
    - mask_file: The file-like object containing the mask in .npy format.

    Returns:
    None
    """
    # Load the mask from the .npy file
    mask = np.load(mask_file, allow_pickle=True)

    # Normalize the mask to span the full 0-255 range if it doesn't already
    if mask.max() > 0:  # Avoid division by zero
        mask = (mask / mask.max()) * 255.0
    mask = mask.astype(np.uint8)

    # Check the shape
    if len(mask.shape) == 3 and mask.shape[2] == 1:
        mask = np.squeeze(mask)  # Remove the singleton dimension

    # Convert the numpy array to a PIL Image
    mask_image = Image.fromarray(mask)

    return mask_image

def mask_to_colored_pil(mask, colormap='viridis'):
    # load the mask, convert it into np format and change dtype to int32
    mask = np.load(mask, allow_pickle=True).astype(np.int32)

    # Apply colormap
    normed_data = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))

    mapped_data = matplotlib.colormaps[colormap](normed_data)

    # Convert to PIL Image
    img_data = (mapped_data[:, :, :3] * 255).astype(np.uint8)
    img = Image.fromarray(img_data)
    return img

if __name__=="__main__":
    st.set_page_config(layout="wide", page_title="PixelMaster")
    st.title("Visualize Masks")

    mask_uploaded = st.file_uploader("Choose the mask", type=["npy"])
    if mask_uploaded:
        mask_image_generated = mask_to_colored_pil(mask_uploaded)

        st.image(mask_image_generated)