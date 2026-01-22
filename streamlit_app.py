import streamlit as st
import numpy as np
import cv2
import imutils
import pytesseract
import pandas as pd
import time
import os
from PIL import Image
import numpy as np



# This hides the 'Made with Streamlit' footer and the GitHub icon/menu
hide_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
st.markdown(hide_style, unsafe_allow_html=True)
# --- APP CONFIG ---
st.set_page_config(page_title="Ethiopian LPR System", layout="centered")
st.title("🚗 License Plate Recognition")
st.write("Upload a vehicle image to extract the plate number (Amharic/English).")

# --- UI SIDEBAR ---
st.sidebar.header("Settings")
save_to_csv = st.sidebar.checkbox("Save result to CSV", value=True)

# --- IMAGE UPLOADER ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Display Original
    st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=True)
    
    # --- PROCESSING ---
    with st.spinner('Processing...'):
        image_resized = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(gray, 170, 200)

        # Contour Detection
        cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]

        NumberPlateCnt = None
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                NumberPlateCnt = approx
                break

        if NumberPlateCnt is not None:
            # Crop ROI
            x, y, w, h = cv2.boundingRect(NumberPlateCnt)
            roi = gray[y:y+h, x:x+w]
            _, roi_binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # OCR
            config = ('-l eng+amh --oem 1 --psm 6')
            text = pytesseract.image_to_string(roi_binary, config=config)
            clean_text = "".join([c for c in text if c.isalnum() or c in ['-', ' ']]).strip()

            # --- DISPLAY RESULTS ---
            st.success(f"**Detected Plate:** {clean_text}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(roi_binary, caption="Cropped Plate (OCR Input)")
            
            # --- SAVE DATA ---
            if save_to_csv:
                data = {
                    'Date': [time.strftime("%Y-%m-%d %H:%M:%S")],
                    'Plate_Number': [clean_text]
                }
                df = pd.DataFrame(data)
                file_path = 'data.csv'
                header_needed = not os.path.exists(file_path)
                df.to_csv(file_path, mode='a', header=header_needed, index=False)
                st.info("Result saved to data.csv")
        else:
            st.error("Could not detect a license plate. Try a clearer image or adjust lighting.")
