import streamlit as st
from utilities.utilities import *
from streamlit_image_coordinates import streamlit_image_coordinates

import matplotlib






def mask_to_colored_pil(mask, colormap='viridis'):
    # Apply colormap
    normed_data = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))

    mapped_data = matplotlib.colormaps[colormap](normed_data)

    # Convert to PIL Image
    img_data = (mapped_data[:, :, :3] * 255).astype(np.uint8)
    img = Image.fromarray(img_data)
    return img
def combine_masks(masks):
    # Ensure the combined_mask starts as an integer array of zeros with the same shape as the first mask
    combined_mask = np.zeros_like(masks[0], dtype=np.int32)

    for i, mask in enumerate(masks):
        # Convert the mask to integer if it's not already, to avoid the UFuncTypeError
        addMask = mask.astype(np.int32) * (i + 1)  # Scale by index (i + 1) for differentiation
        combined_mask += addMask  # Aggregate into combined_mask

    return combined_mask


def combine_masks_with_labels(masks_with_labels):
    if not masks_with_labels:
        return None  # or an appropriate default value or error handling

    #  initialize combined_mask as zeros
    shape_of_first_mask = masks_with_labels[0]['mask'].shape
    combined_mask = np.zeros(shape_of_first_mask, dtype=np.int32)

    for mask_with_label in masks_with_labels:
        mask = mask_with_label['mask']
        label = mask_with_label['Label']

        # Ensure mask is in integer format, then multiply by its label
        # This assumes label is an integer or can be cast to one.
        addMask = mask.astype(np.int32) * label
        combined_mask += addMask

    return combined_mask
if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="PixelMaster")
    st.title("Multi Segmentation")
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

        file_info = f"{uploaded_file.name}-{uploaded_file.size}"

        if "file_info" not in st.session_state or st.session_state["file_info"] != file_info:
            keys_to_reset = ["submitted", "masks", "prev_value", "input_point", "input_label", "selected_masks"]
            for key in keys_to_reset:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state["file_info"] = file_info

        # Process the image
        image = process_image_format(uploaded_file)

        st.write("Please click on this image to select the pixel coordinates")
        value = streamlit_image_coordinates(image)

        # Use session_state to keep track of the submission state and input details
        if 'submitted' not in st.session_state:
            st.session_state['submitted'] = False

        if 'masks' not in st.session_state:
            st.session_state['masks'] = []

        if 'selected_masks' not in st.session_state:
            st.session_state['selected_masks'] = []

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

                            masks, scores, logits = generate_masks(st.session_state['sam_predictor'], input_point, input_label)

                            st.session_state['masks'] = masks

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
                            if st.button("Mark as best",  key="select_best_1"):
                                st.session_state["selected_masks"].append({
                                    "mask": st.session_state['masks'][0],
                                    "Label": 0  # Set initial label to 0
                                })

                if mask_image_2:
                    with col3:
                        with st.expander("Mask 2", expanded=True):
                            st.image(mask_image_2)
                            if st.button("Preview", key="mask2"):
                                preview.image(mask_image_2)

                            if st.button("Mark as best", key="select_best_2"):
                                st.session_state["selected_masks"].append({
                                    "mask": st.session_state['masks'][1],
                                    "Label": 0  # Set initial label to 0
                                })

                if mask_image_3:
                    with col4:
                        with st.expander("Mask 3", expanded=True):
                            st.image(mask_image_3)
                            if st.button("Preview", key="mask3"):
                                preview.image(mask_image_3)

                            if st.button("Mark as best",  key="select_best_3"):
                                st.session_state["selected_masks"].append({
                                    "mask": st.session_state['masks'][2],
                                    "Label": 0  # Set initial label to 0
                                })

                if len(st.session_state["selected_masks"]) > 0:
                    st.header("Selected Masks")

                    # Calculate how many rows we will need
                    rows = (len(st.session_state["selected_masks"]) + 2) // 3

                    # Temporary list to hold updated masks
                    updated_masks = []

                    # Iterate over the rows
                    for row in range(rows):
                        # Dynamically create columns for the row; each row has up to 3 masks
                        cols_selected = st.columns(3)

                        # Iterate over the columns/masks in this row
                        for i in range(3):
                            mask_index = row * 3 + i
                            if mask_index < len(st.session_state["selected_masks"]):
                                mask_item = st.session_state["selected_masks"][mask_index]
                                with cols_selected[i]:
                                    st.write(f"Mask {mask_index + 1}")
                                    selected_mask_image = generate_mask_image(mask_item['mask'],
                                                                              st.session_state["original_image"])
                                    st.image(selected_mask_image)

                                    # Allow the user to update the label for this mask
                                    updated_label = st.number_input("Label", value=mask_item["Label"],
                                                                    key=f"mask_label_{mask_index}")

                                    # Prepare and store the updated mask information
                                    updated_masks.append({"mask": mask_item["mask"], "Label": updated_label})

                    # Update the session state with the updated masks and labels after user interaction
                    if st.button("Update Labels"):
                        st.session_state["selected_masks"] = updated_masks
                        st.success("Labels updated successfully.")

                    # Combine Masks Section
                    if st.button("Combine Masks", key="combine_masks"):
                        combined_mask = combine_masks_with_labels(st.session_state["selected_masks"])

                        # converting for rendering
                        mask_image_combined = mask_to_colored_pil(combined_mask)

                        # Display the combined mask
                        st.image(mask_image_combined, caption="Combined Mask Visualization", use_column_width=True)

                        mask_npy_bytes = get_mask_npy(combined_mask)

                        # Add a download button for the combined mask
                        st.download_button(
                            label="Download Combined Mask",
                            data=mask_npy_bytes,
                            file_name="combined_mask.npy",
                            mime="application/octet-stream"
                        )