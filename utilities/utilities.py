from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
import matplotlib
import numpy as np
import cv2
import io
from utilities.image_processing import TrashIm

def process_image_format(file):
    image = np.load(file, allow_pickle=True)

    # convert images to 8 bit
    image = image / (2 ** 10)
    image = image * 2 ** 8
    image = np.floor(image)
    image = np.uint8(image)

    # Convert BAYER to BGR for displaying
    image = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2BGR)

    return image


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)




def init_sam():
    sam_checkpoint = "model/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    return predictor


def generate_masks(predictor,input_point, input_label):
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    return masks, scores, logits

def image_with_point(image, X, y, diameter=15, color=[255,0,0], thickness=-1):
    """Renders image with point on it"""
    img_copy = np.copy(image)  # Create a copy of the image
    img_with_circle = cv2.circle(img_copy, (X, y), diameter, color, thickness)  # Draw on the copy
    img_pil = Image.fromarray(img_with_circle)
    return img_pil

def generate_mask_image(mask, image, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]

    # Reshape the mask to include all four color channels
    mask_np = np.repeat(mask[..., np.newaxis], 4, axis=-1)

    # Convert the mask array to the same data type as the color array
    mask_np = mask_np.astype(color.dtype)

    # Apply the color to the mask
    mask_np *= color

    # Set alpha channel of background pixels to 0
    mask_np[..., 3] = np.where(mask_np[..., 3] == 0, 0, mask_np[..., 3])

    # Convert the modified NumPy array back to a PIL Image object
    mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))

    # convert the input image to a PIL Image object
    image_pil = Image.fromarray(image)

    # paste the mask onto the image
    image_pil.paste(mask_pil, mask=mask_pil)

    return image_pil

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

def get_mask_npy(mask):
    # Convert the mask to a byte stream

    # Convert the mask to np.int32 type
    mask_int32 = mask.astype(np.int32)
    buffer = io.BytesIO()
    np.save(buffer, mask_int32, allow_pickle=True)
    buffer.seek(0)  # Rewind the buffer to the beginning so it's ready for reading
    return buffer

# This function can be used in streamlit
def generate_image_mosiac(input_file, light, dark, correction, calCurve0):
    # flat field correction
    ffieldIm = TrashIm.ffield(input_file,light,dark)
    bands = [609.0, 625.6, 648.0, 666.3, 683.9, 700.8, 718.9, 736.6, 754.1, 770.1, 786.2, 802.4, 818.3, 833.1, 849.4]
    # interpolate
    wb = TrashIm.interp(ffieldIm)
    # calibration and correction
    imageMosaic = wb.copy()
    test = np.dot(imageMosaic,correction.T)
    imageMosaic = test/calCurve0
    return imageMosaic

# this function needs to be added to streamlit as well
def combine_mosiac_and_therm(image_mosiac, therm,H):
    therm = np.reshape(therm, (therm.shape[0], therm.shape[1],1))
    dstMS, dstTH,dstC = TrashIm.undistort(image_mosiac,therm,color=0)
    imwarp = TrashIm.applyHom(dstTH,H,dstMS.shape[1],dstMS.shape[0])
    fullIm = TrashIm.stack(dstMS,imwarp)
    return fullIm