import streamlit as st
# Streamlit ko import karte hain, jisse tum interactive web app bana pao 
import tensorflow as tf
# TensorFlow import karte hain taaki trained CNN/DNN model ko load aur use kar sako  
import numpy as np
# NumPy import karte hain image ko numerical array me convert karne ke liye aur prediction handle karne ke liye  

# Tensorflow Model Prediction
def model_prediction(test_image):
    # ✅ Step 1: Load the trained CNN model
    model = tf.keras.models.load_model('trained_model.keras')

    # ✅ Step 2: Load the test image and resize it to match model input size (128x128)
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))

    # ✅ Step 3: Convert image into array (numerical representation)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)

    # ✅ Step 4: Convert single image into a batch (since model expects batch input)
    input_arr = np.array([input_arr])

    # ✅ Step 5: Make prediction using trained model
    prediction = model.predict(input_arr)

    # ✅ Step 6: Get the index of class with highest probability
    result_index = np.argmax(prediction)

    # ✅ Step 7: Return predicted class index
    return result_index

# ✅ Sidebar Title
st.sidebar.title("Dashboard")
# ✅ Dropdown Menu for Page Navigation
app_mode = st.sidebar.selectbox(
    "Select Page",              # Label for dropdown
    ["Home","About","Disease Recognition"]     # Options in dropdown
)

# ✅ Agar user "Home" option select kare sidebar me
if(app_mode == "Home"):
    # Page ka header
    st.header("🌱 PLANT DISEASE RECOGNITION SYSTEM")

    # Ek background / banner image show karna (Home page image)
    image_path = "Home_page.jpg"   # Ye image tumhari project folder me saved honi chahiye
    st.image(image_path, use_container_width=True)

    # Thoda project introduction text bhi dal diya taaki page acha lage
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

# ✅ Agar user "About" option select kare sidebar me
elif(app_mode == "About"):
    # Page ka heading
    st.header("📖 About the Project")

    # Project description
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                """)

# ✅ Agar user "Disease Recognition" page select kare to ye block chalega
elif(app_mode=="Disease Recognition"):
    st.header("🌿 Plant Disease Recognition")

    # 🔹 Step 1: Image Upload
    test_image = st.file_uploader("Choose an Image:")
    # 🔹 Step 2: Agar user "Show Image" button dabaye to uploaded image dikhado
    if(st.button("Show Image")):
        st.image(test_image,use_container_width=True)
    
    # 🔹 Step 3: Predict Button
    if(st.button("Predict")):
        st.snow()    # ❄️ Animation effect for fun
        st.write("🔎 Our Prediction:")
        # Model se prediction nikalna
        result_index = model_prediction(test_image)

        # 🔹 Step 4: Class Names (dataset ke 38 categories)
        class_name = ['Apple___Apple_scab',
                    'Apple___Black_rot',
                    'Apple___Cedar_apple_rust',
                    'Apple___healthy',
                    'Blueberry___healthy',
                    'Cherry_(including_sour)___Powdery_mildew',
                    'Cherry_(including_sour)___healthy',
                    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                    'Corn_(maize)___Common_rust_',
                    'Corn_(maize)___Northern_Leaf_Blight',
                    'Corn_(maize)___healthy',
                    'Grape___Black_rot',
                    'Grape___Esca_(Black_Measles)',
                    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                    'Grape___healthy',
                    'Orange___Haunglongbing_(Citrus_greening)',
                    'Peach___Bacterial_spot',
                    'Peach___healthy',
                    'Pepper,_bell___Bacterial_spot',
                    'Pepper,_bell___healthy',
                    'Potato___Early_blight',
                    'Potato___Late_blight',
                    'Potato___healthy',
                    'Raspberry___healthy',
                    'Soybean___healthy',
                    'Squash___Powdery_mildew',
                    'Strawberry___Leaf_scorch',
                    'Strawberry___healthy',
                    'Tomato___Bacterial_spot',
                    'Tomato___Early_blight',
                    'Tomato___Late_blight',
                    'Tomato___Leaf_Mold',
                    'Tomato___Septoria_leaf_spot',
                    'Tomato___Spider_mites Two-spotted_spider_mite',
                    'Tomato___Target_Spot',
                    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                    'Tomato___Tomato_mosaic_virus',
                    'Tomato___healthy']
        # 🔹 Step 5: Final Result Show karo
        st.success(f"✅ Model Prediction: **{class_name[result_index]}**")
    
    


