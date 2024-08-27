import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
import cv2
import tempfile
import os
import gspread
from google.oauth2.service_account import Credentials

def db_access(Skin_Type,Skin_Col):
    # Define the scope
    scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

    CRE_dict = {
  "type": "service_account",
  "project_id": "aiprojectdatabase-433803",
  "private_key_id": "24a66978b0cf51999f68af9c3604d867374754ef",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC1FeUgxYf5VGfG\nLJWd7r8h8hZbB0dU8+gFnN224NUUATM0JuHQcPL2Ci0wcMTZ3gWavBqP5vFpoghp\nh0xT5PWmGDG+4hN8VbS6N++T/Pmoe368GTaA5OQKRdDW84nNehIYlk1njCgN2ZZb\nf+YY2W8USsaEWQhGuqihF3Lm1nwzpxFBetBlsIA1axz07N9wQAe8SWp8RFYBCSDn\n/Kuq3nvnNPIRv+ALU+X8M/AVP0y0HxaVACejznI8VkmyESbaKU+v+M5ASnETsCMw\nl/m6lKdpwdyQA90jwjwGPtY2VXNP2Y+600Wh31YJErI8/8XKbx/kgDRG/wl1qXJa\n9aRbx4fXAgMBAAECggEAIzE4oNhStyVsr4pdn15VPeMe7hzpg3yNVH3qZs6mCme7\nTEDcNNammSNcKeRYlWC9JRe2b835j8ZiLSQStOEzzk44aLjmAY1kfKY/RLruyAwM\nEsExovYYzVhJIGUfHFRDbQzUyTFnXV2yh2DBVoX3PPHVR8ZHwfsnp3r3pR388EqN\nPESXMavvXA3ciXdplszrQcFii5t7xHMD/FW6u56a1PJfuCj1mysG4XYSpkJXKfDx\n1r/87FF5kOcSPpEUVypSlKXtB2cIdMRVDLA/+nKc/QfVnzOMwZcTcHiV4RfIK0PL\nKauSbLE7XT7qmyPtOJzAds81fiORvUIrGaqmIiPYCQKBgQDnUTk/dpgBLkes5KfD\nK2ID2SYygKV/u/S6RpvNnOyTQdOUiTM3KLa+mAme4G+6L2N1BiXOf1KVDquTuZ0f\nj0vSNjqK8bsZ1nXk44TE32ojpBfi6NdksP+KyCatyOw4FqYwnu6vCS6Xhnj9TWa4\ntD1/D4pMdzQb4xKJ3h38cTb5uQKBgQDIaIQlr8n/A1XLti8exvZlHbmy/kh4liBH\n+KWjIAm3xcRSrcpmkt2XsttK/7UkjBt7Ii1gbClWPc3cvXbQbOQHFCtPc1dV9PFy\nJdq+Nq2gXwIcALHDOfje26AYZO+QAMqktwmzUtxhKFEhM69awuGPO7yqXrNNofKQ\n0OeJ6WEWDwKBgHMHUx6aDPDZYM87TamiUzVysKoAi0w/3W0cW7IdzQ9Vdq+wooVV\ne7q/xFj7ZtQBaMXy7q4HZru09eGaNeZRzfSU/vvFRbONkEboVUfJifB7U12FSEdM\nNWeALKvS9JTXvoEDJ9JnEIJNXrEn4mMLTmF3CuEHjiQoAToJ+INmkV4RAoGBAKVC\noq5dLqpO+sH44xRzJ54sjASRcfuWeNpArX4+HiVgPUucqoo5U+gTgohvItYXf1Xj\n0h1wNAo8/vSnfEHVeZhoxmpHB98HFM93bdFrT3QuxJOI8w21UYec/oD/QxmxvWlk\n0ugATWEFGRnTAChNCinOLf8kBqHfCSLoUfbE7917AoGAEGWpOm9J8JC45gd3K0Kj\nqhk2s6MgM+Ru9p1LK9L+V3hW2+yIDRgVRdJY7mmgIP1uG9JMAHomP1VoqqM018Bt\nqIocJX+JmaghpR+e6pbl7GlENw08l5f7YEP4Lm6R6RprPpogCwMnJG1XptATHOnZ\nd4hOzsRIupHXY4tdLgJmyd4=\n-----END PRIVATE KEY-----\n",
  "client_email": "sheetaccess@aiprojectdatabase-433803.iam.gserviceaccount.com",
  "client_id": "104019459891593823430",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/sheetaccess%40aiprojectdatabase-433803.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}


    # Provide the path to your service account key
    creds = Credentials.from_service_account_info(CRE_dict, scopes=scope)

    # Authorize the client
    client = gspread.authorize(creds)

    # Open the Google Sheet
    sheet = client.open("database").sheet1  # Use .worksheet('sheet_name') for specific sheets

    # Update a specific cell
    sheet.update_cell(2,1,Skin_Type)
    print("Skin Type Import to DB")
    sheet.update_cell(2,2,Skin_Col)
    print("Skin Color Import to DB")


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

def user_input_features(r,g,b,SkinType):
                      
    #island = st.sidebar.selectbox('Island',('Biscoe','Dream','Torgersen'))
    #sex = st.sidebar.selectbox('Sex',('male','female'))
    #bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1,59.6,43.9)
    #bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1,21.5,17.2)
    #flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0,231.0,201.0)
    #body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0,6300.0,4207.0)
    data = {'R': r,
            'G': g,
            'B': b,
            'Core_Skin_Type': SkinType}
    features = pd.DataFrame(data, index=[0])
    return features


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

if image_file is not None:

    userskintype = st.selectbox("Select your skin type", options=("---","Oily","Dry","Sensitive","Combination"))
    print(userskintype)

    if userskintype != "---":

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
                st.markdown("---")

                st.markdown(f"""<div style="background-color:{skin_hex}; padding: 20px; border-radius: 5px;">
            <p style="color:white; text-align:center;">{skin_hex}</p></div>""", unsafe_allow_html=True)
                st.markdown(f"""<div style="background-color:{skin_hex}; padding: 20px; border-radius: 5px;">
            <p style="color:white; text-align:center;">R:{rgbcolor[0]}, G:{rgbcolor[1]}, B:{rgbcolor[2]}</p></div>""", unsafe_allow_html=True)
                
                red = rgbcolor[0]
                green = rgbcolor[1]
                blue = rgbcolor[2]
                input_df = user_input_features(red,green,blue,userskintype)

                #Model Releated Section
                makeupdataset_clean = pd.read_csv("https://raw.githubusercontent.com/htetlwink/AIProjectTest/main/dataset/makeupdatasetclean.csv")
                makeupdataset = makeupdataset_clean.drop(columns=['Core_Color'])
                df = pd.concat([input_df,makeupdataset],axis=0)
                
                #Encode the skintype
                encode = ['Core_Skin_Type']
                for col in encode:
                    dummy = pd.get_dummies(df[col], prefix=col)
                    df = pd.concat([df,dummy], axis=1)
                    del df[col]
                df = df[:1]

                #Model Loaded
                load_clf = pickle.load(open('model/TrainedModel.pkl', 'rb'))

                #Model Prediction
                prediction = load_clf.predict(df)

                st.subheader('You Should Use This MakeUp Color')
                CoreColor_Suggest = np.array(['Pale','Natural','Golden','Mocha'])
                st.write(CoreColor_Suggest[prediction][0])

                #Save the predict Color
                ColorPredit = CoreColor_Suggest[prediction][0]

                #Import Data into sheet
                db_access(userskintype, ColorPredit)
                
            except ValueError as e:
                st.error(str(e))
            finally:
                os.remove(temp_file_path)
            if clearbtm:
                image_file is None