import streamlit as st
from utilities.utilities import *
from streamlit_image_coordinates import streamlit_image_coordinates

import io


def display_mask(mask):
    """Converts a single-channel mask to a 3-channel image for display."""
    if len(mask.shape) == 2:  # If the mask is single-channel, convert it to 3-channel
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    return mask


# Add this function to convert the mask to a BytesIO object and return it
def get_mask_npy(mask):
    # Convert the mask to a byte stream
    buffer = io.BytesIO()
    np.save(buffer, mask, allow_pickle=True)
    buffer.seek(0)  # Rewind the buffer to the beginning so it's ready for reading
    return buffer


if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="PixelMaster")
    st.title("Image Segmentation Using SAM")
    mask_image_1= None
    mask_image_2= None
    mask_image_3= None

    # Upload the file
    uploaded_file = st.file_uploader("Choose an image")

    if 'sam_predictor' not in st.session_state:
    # sam predictor =
        sam_predictor = init_sam()
        st.session_state['sam_predictor'] = sam_predictor


    if uploaded_file is not None:
        # Process the image
        image = process_image_format(uploaded_file)





        st.write("Please click on this image to select the pixel coordinates")
        value = streamlit_image_coordinates(image)

        # Use session_state to keep track of the submission state and input details
        if 'submitted' not in st.session_state:
            st.session_state['submitted'] = False

        if 'masks' not in st.session_state:
            st.session_state['masks'] = []

        st.session_state['original_image'] = image

        if value:
            # Check if 'prev_value' exists and if it's different from the current 'value'
            if 'prev_value' in st.session_state and (st.session_state['prev_value'] != value):
                st.session_state['masks'] = []  # Reset the masks list if 'value' has changed

            # Update 'prev_value' in session_state to the current 'value'
            st.session_state['prev_value'] = value

            st.session_state['submitted'] = True
            st.session_state['input_point'] = np.array([[value["x"], value["y"]]])
            st.session_state['input_label'] = np.array([1])

        if st.session_state['submitted']:
            # This block now uses session_state to preserve the form's submission state and inputs
            input_point = st.session_state['input_point']
            input_label = st.session_state['input_label']
            #
            image_with_marker = image_with_point(st.session_state['original_image'], st.session_state['input_point'][0,0], st.session_state['input_point'][0,1])


            st.divider()

            st.subheader("Preview Section")
            # for the original image with pointer

            if image_with_marker:
                preview = st.image(image_with_marker, caption='Preview')
                col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
                with col1:
                    with st.expander("Image with Marker", expanded=True):
                        st.image(image_with_marker)
                        if st.button("Generate Masks"):
                            # set predictor to image
                            st.session_state['sam_predictor'].set_image(st.session_state['original_image'])
                            # Assuming sam_predictor is available here
                            masks, scores, logits = generate_masks(st.session_state['sam_predictor'], input_point, input_label)

                            st.session_state['masks'] = masks
                            # Now, this should work as expected
                            print(scores)
                            cols = st.columns(3)  # Create three columns

                        if st.button("Preview", key="preview1"):
                            preview.image(image_with_marker)

                if len(st.session_state['masks'])>0:
                    mask_image_1 = generate_mask_image(st.session_state['masks'][0],  st.session_state['original_image'])
                    mask_image_2 = generate_mask_image(st.session_state['masks'][1],  st.session_state['original_image'])
                    mask_image_3 = generate_mask_image(st.session_state['masks'][2],  st.session_state['original_image'])

                if mask_image_1:
                    with col2:
                        with st.expander("Mask 1", expanded=True):
                            st.image(mask_image_1)
                            if st.button("Preview", key="mask1"):
                                preview.image(mask_image_1)

                            mask_1_download = get_mask_npy(st.session_state['masks'][0])  # Get the mask as a BytesIO object
                            st.download_button(
                                label="Download Mask 1",
                                data=mask_1_download,
                                file_name=f"mask_1.npy",
                                mime="application/octet-stream"
                            )

                if mask_image_2:
                    with col3:
                        with st.expander("Mask 2", expanded=True):
                            st.image(mask_image_2)
                            if st.button("Preview", key="mask2"):
                                preview.image(mask_image_2)

                            mask_2_download = get_mask_npy(
                                st.session_state['masks'][1])  # Get the mask as a BytesIO object
                            st.download_button(
                                label="Download Mask 2",
                                data=mask_2_download,
                                file_name=f"mask_2.npy",
                                mime="application/octet-stream"
                            )

                if mask_image_3:
                    with col4:
                        with st.expander("Mask 3", expanded=True):
                            st.image(mask_image_3)
                            if st.button("Preview", key="mask3"):
                                preview.image(mask_image_3)

                            mask_3_download = get_mask_npy(
                                st.session_state['masks'][2])  # Get the mask as a BytesIO object
                            st.download_button(
                                label="Download Mask 3",
                                data=mask_3_download,
                                file_name=f"mask_3.npy",
                                mime="application/octet-stream"
                            )