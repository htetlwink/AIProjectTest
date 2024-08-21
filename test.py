import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
#import model
import pandas as pd
import tensorflow as tf
import pickle

st.set_page_config(page_title="Skin Color Checker", page_icon="📷")
st.markdown("<h1 style='text-align: center;'>Skin Color Checker?</h1>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center;'>နောက်ခံပြောင်မှာ မျက်နှာ ကို ရှင်းရှင်းလင်းလင်းပေါ်ရင်‌ပိုကောင်းပါတယ်</h6>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center;'>နေရောင်ရှိတဲ့ဘက်မျက်နှာမူပြီးရရိုက်ရင် ပိုကောင်းပါတယ်</h6>", unsafe_allow_html=True)
st.markdown("---")
st.markdown(f"""<div style="background-color:red; padding: 10px; border-radius: 1px;">
        <p style="color:white; text-align:center;">Upload photo က 2MB ထပ်ကျော်ရင် server down ပါတယ် </p></div>""", unsafe_allow_html=True)
st.markdown("---")
image_file = st.file_uploader("Upload Your Selfie to Color Check", type=["jpg", "png", "jpeg"])
st.markdown("<h6 style='text-align: center;'>Disclaimer: We do not store personal data and your picture only valid in this particular instance</h6>", unsafe_allow_html=True)

def UseTheModel(SkinType,R,G,B):

    #Load the Model File
    with open("model/Trained_model.pkl", 'rb') as file:
        model = pickle.load(file)
        print("Model Loaded Successfully !")

    #Load the Encoder File
    with open('model/Skin_Type_encoder.pkl', 'rb') as file:
        le = pickle.load(file)

    UserData = np.array([[SkinType,R,G,B]])
    prediction = model.predict(UserData)
    makeuptype = le.inverse_transform(prediction)
    return print(f"You Should Use {makeuptype}")



def get_skin_color_from_face(image_path):
    # Load the image
    image = cv2.imread(image_path)
    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load the Haar cascade XML file
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    # Convert the image to grayscale for face detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        raise ValueError("No face detected in the image.")

    # Assume the first detected face is the one we want
    x, y, w, h = faces[0]
    
    # Extract the face region from the image
    face_region = image_rgb[y:y+h, x:x+w]
    
    # Convert the face region to HSV color space
    face_hsv = cv2.cvtColor(face_region, cv2.COLOR_RGB2HSV)
    
    # Define skin color range in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Create a mask for skin color
    skin_mask = cv2.inRange(face_hsv, lower_skin, upper_skin)
    
    # Apply the mask to the face region
    skin = cv2.bitwise_and(face_region, face_region, mask=skin_mask)
    
    # Find the average color of the skin
    mean_color = cv2.mean(face_region, mask=skin_mask)[:3]
    
    # Calculate the most frequent (mode) color in the skin region
    # Reshape the skin region to a 2D array of pixels
    pixels = skin[skin_mask > 0].reshape(-1, 3)
    
    # Find the mode of the pixels (most frequent color)
    mode_color = np.array([np.bincount(pixels[:, i]).argmax() for i in range(3)])
    
    # Find the midpoint between mean and mode colors
    midpoint_color = ((mean_color + mode_color) / 2).astype(int)
    
    # MODE Color Convert RGB to HEX
    hex_color = '#{:02x}{:02x}{:02x}'.format(mode_color[0], mode_color[1], mode_color[2])

    # MIDPOINT COLOR Convert RGB to HEX
    #hex_color = '#{:02x}{:02x}{:02x}'.format(midpoint_color[0], midpoint_color[1], midpoint_color[2])
    
    # MEAN COLOR Convert RGB to HEX
    #hex_color = '#{:02x}{:02x}{:02x}'.format(int(mean_color[0]), int(mean_color[1]), int(mean_color[2]))

    #Assign RGB Color code
    rgbcolor = mode_color

    return hex_color, image_rgb, (x, y, w, h), face_region, skin_mask, skin, rgbcolor

if image_file is not None:

    userskintype = st.selectbox("Select your skin type", options=("---","Oily","Dry","Sensitive","Combination"))
    print(userskintype)
    
    if userskintype != "---":
            
        # Save the uploaded image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(image_file.read())
            temp_file_path = temp_file.name
                    
        submitbtm = st.button("Start Analyze")
        if submitbtm:

            clearbtm = st.button("Clear Data")        
            try:
                skin_hex, image_rgb, face_rect, face_region, skin_mask, skin_region, rgbcolor = get_skin_color_from_face(temp_file_path)
                st.markdown(f"""<div style="background-color:{skin_hex}; padding: 20px; border-radius: 5px;">
            <p style="color:white; text-align:center;">This is your color tone {skin_hex}!</p></div>""", unsafe_allow_html=True)
                st.markdown("---")
                # Display images
                st.image(image_rgb, caption="Original Image")
                st.markdown("---")
                x, y, w, h = face_rect
                cv2.rectangle(image_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
                st.image(image_rgb, caption="Detected Face")
                st.markdown("---")

                st.image(face_region, caption="Face Region")
                st.markdown("---")
                st.image(skin_mask, caption="Skin Mask")
                st.markdown("---")
                st.image(skin_region, caption="Skin Region")
                st.markdown(f"""<div style="background-color:{skin_hex}; padding: 20px; border-radius: 5px;">
            <p style="color:white; text-align:center;">{skin_hex}</p></div>""", unsafe_allow_html=True)
                st.markdown(f"""<div style="background-color:{skin_hex}; padding: 20px; border-radius: 5px;">
            <p style="color:white; text-align:center;">R:{rgbcolor[0]}, G:{rgbcolor[1]}, B:{rgbcolor[2]}</p></div>""", unsafe_allow_html=True)

                makeupshade = UseTheModel(userskintype,rgbcolor[0],rgbcolor[1],rgbcolor[2])
                st.markdown(f"""<div style="background-color:{skin_hex}; padding: 20px; border-radius: 5px;">
            <p style="color:white; text-align:center;">Your Should Use {makeupshade}</p></div>""", unsafe_allow_html=True)
                
            except ValueError as e:
                st.error(str(e))
            finally:
                os.remove(temp_file_path)
            if clearbtm:
                image_file is None
