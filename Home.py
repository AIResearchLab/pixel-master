import streamlit as st
from PIL import Image




def main():
    st.set_page_config(layout="wide", page_title="PixelMaster")
    st.image('assets\modified_logo.png', width=250)
    footer = """
        ---
            University of Canberra, Bruce ACT 2617 Australia
            Phone: +61 2 6201 5111
            ABN: 81 633 873 422
            CRICOS: 00212K
            TEQSA Provider ID: PRV12003 (Australian University)
            © 2024 UC's Human Centred Technology Research Centre. All rights reserved.
        """
    st.markdown(footer)

if __name__ == '__main__':
    main()
