import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
import cv2
import tempfile
import os
import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="AI Makeup Foundation Assistant",page_icon="💋")

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

def db_access(Skin_Type,Skin_Col,CRE_dict):
    # Define the scope
    scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

    # Provide the path to your service account key
    creds = Credentials.from_service_account_info(CRE_dict, scopes=scope)

    # Authorize the client
    client = gspread.authorize(creds)

    # Open the Google Sheet
    sheet = client.open("database").worksheet("SQL_test")  # Use .worksheet('sheet_name') for specific sheets

    # Update a specific cell
    sheet.update_cell(2,1,Skin_Type)
    print("Skin Type Import to DB")
    sheet.update_cell(2,2,Skin_Col)
    print("Skin Color Import to DB")

def Foundation_Access(CRE_dict):
    # Define the scope
    scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

    # Provide the path to your service account key
    creds = Credentials.from_service_account_info(CRE_dict, scopes=scope)

    # Authorize the client
    client = gspread.authorize(creds)

    # Open the Google Sheet
    sheet = client.open("database").worksheet("SQL_test")  # Use .worksheet('sheet_name') for specific sheets

    # REad a specific cell
    
    #color_name = sheet.acell("Y4").value
    #print(f"Color cell name is {color_name}")
    #color_hex = sheet.acell("Y4").value
    #print(f"Color Hex Value is {color_hex}")

    #st.markdown("Foundation Suggestion Color")
    #st.markdown(f"""<div style="background-color:{color_hex}; padding: 20px; border-radius: 5px;">
    #<p style="color:white; text-align:center;">{color_name}</p></div>""", unsafe_allow_html=True)

    Foundation_Color = sheet.get("Q4:Q6")
    Foundation_Hex = sheet.get("Y4:Y6")
    flattened_color_data = [item for sublist in Foundation_Color for item in sublist]
    flattened_hex_data = [item for sublist in Foundation_Hex for item in sublist]
    for i,j in zip(flattened_color_data,flattened_hex_data):
        print(i)
        print(j)
        st.markdown(f"""<div style="background-color:{j}; padding: 20px; border-radius: 5px;">
        <p style="color:white; text-align:center;">{i}</p></div>""", unsafe_allow_html=True)    

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
                      
    data = {'R': r,
            'G': g,
            'B': b,
            'Core_Skin_Type': SkinType}
    features = pd.DataFrame(data, index=[0])
    return features


with st.sidebar:
    selected = option_menu(
        menu_title="AI Makeup Foundation Assistant",
        options=["Home","Skin Type","How it's work?"],
        icons=["house","book","lightbulb"],
        menu_icon="emoji-smile",
        default_index=0,
        #orientation="horizontal"
)

if selected == "Home":
    #st.title(f"You have selected {selected}")
    st.markdown("<h1 style='text-align: center;'>AI Makeup Foundation Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center;'>Ensure the face is well-lit by natural light with no shadows.</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center;'>Use a clear background for best results.</h6>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"""<div style="background-color:#42f5cb; padding: 5px; border-radius: 1px;">
            <p style="color:black; text-align:center;">Please ensure your photo is less than 2MB to avoid service issues.</p></div>""", unsafe_allow_html=True)
    st.markdown("---")
    
    image_file = st.file_uploader("Upload Your Selfie to Color Check", type=["jpg", "png", "jpeg"])
    
    st.markdown("<h6 style='text-align: center;'>Disclaimer: Your photo is only valid for this instance only. We do not store or collect your personal data.</h6>", unsafe_allow_html=True)
    
    if image_file is not None:
        if 'show_help' not in st.session_state:
            st.session_state.show_help = False
        col1, col2 = st.columns([1, 1])

        #if not st.session_state.show_help:
        with col1:
            userskintype = st.selectbox("Select your skin type", options=("---","Oily","Dry","Sensitive","Combination"))
            print(userskintype)

        with col2:
            if st.button('Need Help With Skin Type?'):
                st.session_state.show_help = not st.session_state.show_help

        if st.session_state.show_help:
            st.markdown("<h1 style='text-align: center;'>What is Your Skin Type ?</h1>", unsafe_allow_html=True)
            st.subheader("Oily Skin")
            st.markdown("""Oily skin is a skin type characterized by excess production of sebum, the natural oil produced by sebaceous glands in the skin. This excess oil can lead to a shiny or greasy appearance, enlarged pores, and an increased likelihood of acne and blackheads. People with oily skin often need to use specific skincare products to manage oil production and keep their skin balanced.""")
            st.image("pic/OilySkin2.jpg","Oily Skin Sample")

            st.subheader("Dry Skin")
            st.markdown("""Dry skin is a skin type characterized by a lack of moisture in the outer layer of the skin. This can lead to a rough, flaky, or scaly texture, a tight or uncomfortable feeling, and sometimes itching or irritation. Dry skin can be caused by environmental factors like cold weather, low humidity, or harsh soaps, as well as by underlying health conditions or aging. People with dry skin typically need to use moisturizing products to help restore and maintain hydration.""")
            st.image("pic/DrySkin.jpg","Dry Skin Sample")

            st.subheader("Sensitive Skin")
            st.markdown("""Sensitive skin is a skin type that reacts easily to various products, environmental factors, or even touch. It can become red, itchy, or irritated when exposed to things like harsh chemicals, fragrances, or extreme temperatures. People with sensitive skin need to be careful with the products they use to avoid triggering these reactions.""")
            st.image("pic/SenSkin.jpg","Sensitive Skin Sample")

            st.subheader("Combination Skin")
            st.markdown("""Combination skin is characterized by having different skin types in various areas of the face. Typically, the T-zone, which includes the forehead, nose, and chin, is oilier and may have larger pores. This area often experiences excess shine, blackheads, or acne due to the increased oil production. In contrast, the cheeks and sometimes other parts of the face, like the jawline or around the eyes, may be drier or normal. These areas can feel tight, flaky, or less oily. Managing combination skin often requires using different products or skincare routines for each area: a mattifying treatment for the oily T-zone and a hydrating product for the drier regions.""")
            st.image("pic/CombineSkin1.jpg","Combination Skin Sample")
            
            # Show close help button
            if st.button('Close Help'):
                st.session_state.show_help = False
                
        #userskintype = st.selectbox("Select your skin type", options=("---","Oily","Dry","Sensitive","Combination"))

        if userskintype != "---":

            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(image_file.read())
                temp_file_path = temp_file.name

            submitbtm = st.button("Start Analyze")
            
            if submitbtm:

                clearbtm = st.button("Clear Data")

                try:
                    global skin_region
                    global skin_mask
                    global image_rgb
                    global face_region
                    global face_rect
                    global skin_hex
                    global rgbcolor
                    skin_hex, image_rgb, face_rect, face_region, skin_mask, skin_region, rgbcolor = get_skin_color_from_face(temp_file_path)
                    st.markdown(f"""<div style="background-color:{skin_hex}; padding: 20px; border-radius: 5px;">
                <p style="color:white; text-align:center;">This is your color tone {skin_hex}!</p></div>""", unsafe_allow_html=True)
                    st.markdown("---")
                    # Display images
                    st.session_state.skin_region = skin_region
                    st.session_state.skin_mask  = skin_mask
                    st.session_state.image_rgb = image_rgb
                    st.session_state.face_region = face_region
                    st.session_state.face_rect = face_rect
                    st.session_state.skin_hex = skin_hex
                    st.session_state.rgbcolor = rgbcolor

                    with st.expander("Wanna see how your photo process ?",icon="🔎"):
                        st.markdown("<h6 style='text-align: center;'>This is your Original Photo.</h6>", unsafe_allow_html=True)
                        st.image(image_rgb, caption="Original Image")
                        st.markdown("---")

                        x, y, w, h = face_rect
                        cv2.rectangle(image_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        st.markdown("<h6 style='text-align: center;'>We use haarcascade to detect your face.</h6>", unsafe_allow_html=True)
                        st.image(image_rgb, caption="Detected Face")
                        st.markdown("---")
                        st.markdown("<h6 style='text-align: center;'>We select your face region.</h6>", unsafe_allow_html=True)
                        st.image(face_region, caption="Face Region")
                        st.markdown("---")
                        st.markdown("<h6 style='text-align: center;'>We use HSV values to detect the your skin and mask it to seperate skin and non-skin.</h6>", unsafe_allow_html=True)
                        st.image(skin_mask, caption="Skin Mask")
                        st.markdown("---")
                        st.markdown("<h6 style='text-align: center;'>Finally, this is your skin region.</h6>", unsafe_allow_html=True)
                        st.image(skin_region, caption="Skin Region")
                        st.markdown("---")
                        st.markdown("<h6 style='text-align: center;'>We use MODE Color which is most frequent color from your skin region.</h6>", unsafe_allow_html=True)
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

                    st.subheader('Prediction...')
                    CoreColor_Suggest = np.array(['Pale','Natural','Golden','Mocha'])
                    #st.write(CoreColor_Suggest[prediction][0])
                    
                    st.markdown(F"<h6 style='text-align: center;'>{CoreColor_Suggest[prediction][0]} is the best foundation makeup color for you !!!</h6>", unsafe_allow_html=True)

                    #Save the predict Color
                    ColorPredit = CoreColor_Suggest[prediction][0]

                    if ColorPredit == "Pale":
                        st.markdown(f"""<div style="background-color:#FAF9DE; padding: 20px; border-radius: 5px;">
                    <p style="color:white; text-align:center;">Pale</p></div>""", unsafe_allow_html=True)
                    if ColorPredit == "Natural":
                        st.markdown(f"""<div style="background-color:#F1ECE8; padding: 20px; border-radius: 5px;">
                    <p style="color:white; text-align:center;"></p>Natural</div>""", unsafe_allow_html=True)
                    if ColorPredit == "Golden":
                        st.markdown(f"""<div style="background-color:#FFD700; padding: 20px; border-radius: 5px;">
                    <p style="color:white; text-align:center;">Golden</p></div>""", unsafe_allow_html=True)
                    if ColorPredit == "Mocha":
                        st.markdown(f"""<div style="background-color:#6D3B07; padding: 20px; border-radius: 5px;">
                    <p style="color:white; text-align:center;">Mocha</p></div>""", unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.markdown("---")
                    st.markdown("Our suggested eye shadow base color...")

                    #Import Data into sheet
                    db_access(userskintype, ColorPredit,CRE_dict)

                    #Display Foundation Color Suggestion
                    Foundation_Access(CRE_dict)
                    
                except ValueError as e:
                    st.error(str(e))
                finally:
                    os.remove(temp_file_path)
                if clearbtm:
                    image_file is None



if selected == "Skin Type":
    #st.title(f"You have selected {selected}")
    st.markdown("<h1 style='text-align: center;'>What is Your Skin Type ?</h1>", unsafe_allow_html=True)
    st.subheader("Oily Skin")
    st.markdown("""Oily skin is a skin type characterized by excess production of sebum, the natural oil produced by sebaceous glands in the skin. This excess oil can lead to a shiny or greasy appearance, enlarged pores, and an increased likelihood of acne and blackheads. People with oily skin often need to use specific skincare products to manage oil production and keep their skin balanced.""")
    st.image("pic/OilySkin2.jpg","Oily Skin Sample")

    st.subheader("Dry Skin")
    st.markdown("""Dry skin is a skin type characterized by a lack of moisture in the outer layer of the skin. This can lead to a rough, flaky, or scaly texture, a tight or uncomfortable feeling, and sometimes itching or irritation. Dry skin can be caused by environmental factors like cold weather, low humidity, or harsh soaps, as well as by underlying health conditions or aging. People with dry skin typically need to use moisturizing products to help restore and maintain hydration.""")
    st.image("pic/DrySkin.jpg","Dry Skin Sample")

    st.subheader("Sensitive Skin")
    st.markdown("""Sensitive skin is a skin type that reacts easily to various products, environmental factors, or even touch. It can become red, itchy, or irritated when exposed to things like harsh chemicals, fragrances, or extreme temperatures. People with sensitive skin need to be careful with the products they use to avoid triggering these reactions.""")
    st.image("pic/SenSkin.jpg","Sensitive Skin Sample")

    st.subheader("Combination Skin")
    st.markdown("""Combination skin is characterized by having different skin types in various areas of the face. Typically, the T-zone, which includes the forehead, nose, and chin, is oilier and may have larger pores. This area often experiences excess shine, blackheads, or acne due to the increased oil production. In contrast, the cheeks and sometimes other parts of the face, like the jawline or around the eyes, may be drier or normal. These areas can feel tight, flaky, or less oily. Managing combination skin often requires using different products or skincare routines for each area: a mattifying treatment for the oily T-zone and a hydrating product for the drier regions.""")
    st.image("pic/CombineSkin1.jpg","Combination Skin Sample")


if selected == "How it's work?":

    st.markdown("This will be our github and contact.")