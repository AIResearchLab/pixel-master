import streamlit as st
import numpy as np
from utilities.image_processing import TrashIm
import matplotlib.pyplot as plt
from utilities.utilities import generate_image_mosiac, combine_mosiac_and_therm
import io
def load_files_for_image_processing():
    # Load 'dark' image array only if not in session state
    if 'dark' not in st.session_state:
        dark_files = [
            'utilities\\image_processing\\utilImages\\darkmosaic.npy',
            'utilities\\image_processing\\utilImages\\darkmosaic1.npy',
            'utilities\\image_processing\\utilImages\\darkmosaic2.npy'
        ]
        st.session_state.dark = sum(np.load(file) for file in dark_files) / 3.0

    # Load 'light' image array only if not in session state
    if 'light' not in st.session_state:
        st.session_state.light = np.load('utilities\\image_processing\\utilImages\\12_07_dfield.npy')

    # Load calibration curve only if not in session state
    if 'calCurve0' not in st.session_state:
        st.session_state.calCurve0 = np.load('utilities\\image_processing\\utilImages\\calCurve.npy')

    # Load correction arrays only if not in session state
    if 'correction' not in st.session_state:
        c0 = np.load('utilities\\image_processing\\utilImages\\correction0.npy')
        c1 = np.load('utilities\\image_processing\\utilImages\\correction1.npy')
        st.session_state.correction = (c0 + c1) / 2.0

    # Load homography matrices only if not in session state
    if 'H' not in st.session_state:
        st.session_state.H = np.load('utilities\\image_processing\\CamMatrices\\HomographyTherm2.npy')
    if 'H1' not in st.session_state:
        st.session_state.H1 = np.load('utilities\\image_processing\\CamMatrices\\HomographyColor.npy')


def img_he(img):
    histo = TrashIm.image_histogram_equalization(img, 30)
    hist = histo[0]
    return hist
def plot_channels(img, nrows, ncols):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 12))

    for i in range(nrows):
        for j in range(ncols):
            # Calculate the index of the current channel
            channel_index = i * ncols + j
            if channel_index < img.shape[2]:
                ax = axes[i, j]
                # Plotting the channel data
                ax.imshow(img[:, :, channel_index])
                ax.axis('off')  # Turn off axis for cleaner visualization
                ax.set_title(f'Channel {channel_index + 1}')

    plt.tight_layout()
    return fig

def process_image(input_file, therm):
    st.session_state['submitted'] = True

    im_mosiac = generate_image_mosiac(input_file, st.session_state['light'], st.session_state['dark'],
                                      st.session_state['correction'], st.session_state['calCurve0'])
    # this file needs to be downloaded
    fullIm = combine_mosiac_and_therm(im_mosiac, therm, st.session_state['H'])
    return fullIm

# Function to convert array to bytes for download
def get_image_download_link(img_array):
    buffer = io.BytesIO()
    np.save(buffer, img_array)
    buffer.seek(0)
    return buffer

# Function to reset session state except for the uploaders
def reset_session_state():
    keys_to_preserve = {'uploaded_raw_img', 'uploaded_thermal_img'}  # Preserve uploader states
    for key in list(st.session_state.keys()):
        if key not in keys_to_preserve:
            del st.session_state[key]


if __name__=="__main__":
    st.set_page_config(layout="wide", page_title="PixelMaster")
    st.title("Image Processing")
    # Upload the raw image
    uploaded_raw_img = st.file_uploader("Upload Raw Image", key="uploaded_raw_img")
    if uploaded_raw_img is not None and uploaded_raw_img != st.session_state.get('uploaded_raw_img_last', None):
        st.session_state.uploaded_raw_img_last = uploaded_raw_img
        reset_session_state()  # Reset session state when a new raw image is uploaded

    # Upload the thermal image
    uploaded_thermal_img = st.file_uploader("Upload Thermal Image", key="uploaded_thermal_img")
    if uploaded_thermal_img is not None and uploaded_thermal_img != st.session_state.get('uploaded_thermal_img_last',
                                                                                         None):
        st.session_state.uploaded_thermal_img_last = uploaded_thermal_img
        reset_session_state()  # Reset session state when a new thermal image is uploaded
    # load all the required items
    load_files_for_image_processing()

    if "processed_image" not in st.session_state:
        st.session_state["processed_image"] = None



    if uploaded_raw_img is not None and uploaded_thermal_img is not None:
        # input raw_image
        raw_img = np.load(uploaded_raw_img)
        therm = np.load(uploaded_thermal_img)

        display = img_he(raw_img)

        prev_col1, prev_col2 = st.columns(2)
        with prev_col1:
            st.subheader("Histogram Equalization for the MultiSpectral")
            fig1, ax1 = plt.subplots()
            ax1.imshow(display, 'gray')
            ax1.axis('off')
            st.pyplot(fig1)

        with prev_col2:
            st.subheader("Thermal Image")
            fig1, ax1 = plt.subplots()
            ax1.imshow(therm)
            ax1.axis('off')
            st.pyplot(fig1)

        if st.button("Preprocess Image"):
            st.session_state.processed_image = process_image(raw_img, therm)

        if st.session_state["processed_image"] is not None:
            with st.expander("Preview Processed Image"):
                fig, ax = plt.subplots()
                # Specify the number of rows and columns for the plot
                nrows = 4
                ncols = 4

                # Display the plot in Streamlit
                fig = plot_channels(st.session_state["processed_image"], nrows, ncols)
                st.pyplot(fig)

            # Get bytes buffer from the numpy array (image data)
            downloadable_data_cube = get_image_download_link(st.session_state["processed_image"])

            # Create a link for downloading
            st.download_button(
                label="Download Processed Image",
                data=downloadable_data_cube,
                file_name="processed_data_cube.npy",
                mime="application/octet-stream"
            )
