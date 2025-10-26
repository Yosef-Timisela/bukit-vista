import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
import os
import time # Import the time module

# --- PAGE CONFIG ---
st.set_page_config(page_title="🏡 BukitVista Property Price Predictor (CNN)", layout="wide")
st.title("🏡 BukitVista Property Price Prediction App")
st.markdown("Upload a property image to predict its price category and see similar BukitVista listings.")

# Define hardcoded local paths
DATASET_PATH = "bukitvista_analyst.xlsx"
MODEL_PATH = "price_image_classifier.h5"

# --- LOAD DATASET (Cached) ---
@st.cache_data
def load_dataset(path):
    # Load dataset assuming it's in the same directory
    try:
        df = pd.read_excel(path, sheet_name="Sheet1")
        return df
    except FileNotFoundError:
        st.error(f"❌ Dataset file not found: **{path}**.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Failed to load dataset: {e}")
        st.stop()

df = load_dataset(DATASET_PATH)

# --- LOAD MODEL (Cached Resource) ---
@st.cache_resource
def load_model_from_disk(path):
    try:
        return tf.keras.models.load_model(path)
    except FileNotFoundError:
        st.error(f"❌ Model file not found: **{path}**. Please ensure it's in the same folder.")
        st.stop()
    except Exception as e:
        st.error(f"❌ An error occurred loading the model: {e}")
        st.stop()

model = load_model_from_disk(MODEL_PATH)
st.sidebar.success(f"✅ Model loaded successfully: {MODEL_PATH}")
st.sidebar.success(f"✅ Dataset loaded successfully: {DATASET_PATH}")


# --- LABELS AND PRICE RANGES ---
class_labels = ['Low', 'Medium', 'High']
price_ranges = {
    "Low": "$30–70 per night",
    "Medium": "$70–150 per night",
    "High": "$150+ per night"
}

# Preprocess the uploaded image
def preprocess_image(img, target_size=(128, 128)):
    # The model's input shape determines the required size
    img_resized = img.resize(target_size)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Define price categorization function (used for filtering recommendations)
def categorize_price(price):
    if pd.isna(price):
        return 'Unknown'
    price = float(price)
    if price < 70:
        return 'Low'
    elif 70 <= price <= 150:
        return 'Medium'
    else:
        return 'High'

# --- UPLOAD IMAGE FOR PREDICTION ---
st.header("📤 Upload Property Image for Prediction")
# Updated file uploader to accept more common image formats
image_formats = ["jpg", "jpeg", "png", "webp", "tiff", "tif", "bmp", "gif"]
uploaded_img = st.file_uploader("Choose a property image...", type=image_formats)

if uploaded_img is not None:
    # Get image input and target size
    img = Image.open(uploaded_img).convert("RGB")
    
    # Determine target size from model input shape (assuming channels last, e.g., (None, 128, 128, 3))
    input_shape = model.input_shape
    target_size = (input_shape[1], input_shape[2]) if len(input_shape) == 4 else (128, 128)

    st.image(img, caption=f"🖼️ Uploaded Property Image (Resized to {target_size[0]}x{target_size[1]})")

    # Preprocess
    img_array = preprocess_image(img, target_size)

    # Predict
    pred = model.predict(img_array)[0]
    predicted_class_index = np.argmax(pred)
    predicted_class = class_labels[predicted_class_index]
    confidence = np.max(pred) * 100

    # --- Prediction Result (Same text output as Colab print) ---
    st.subheader("🎯 Prediction Result")
    
    st.markdown(f"**🏷️ Predicted class:** `{predicted_class}`")
    st.markdown(f"**💵 Estimated price range:** `{price_ranges.get(predicted_class, 'N/A')}`")
    st.markdown(f"**🔢 Confidence:** `{confidence:.2f}%`")
    st.markdown(f"**🔍 Probabilities:** `{dict(zip(class_labels, [round(p*100,2) for p in pred]))}`")


    # --- Property Recommendations from Dataset ---
    st.header("🏘️ Similar Property Recommendations")

    if 'price_per_night' not in df.columns or 'picture_url' not in df.columns:
        st.error("❌ Dataset must contain 'price_per_night' and 'picture_url' columns for recommendations.")
    else:
        # Prepare recommendation data
        df['price_per_night'] = pd.to_numeric(df['price_per_night'], errors='coerce')
        df['price_class'] = df['price_per_night'].apply(categorize_price)
        
        # Filter properties by the predicted class
        recos_df = df[df['price_class'] == predicted_class]
        sample_size = min(3, len(recos_df))

        if recos_df.empty or sample_size == 0:
            st.warning(f"No similar properties found in the dataset for the '{predicted_class}' category.")
        else:
            # --- FIX: Use controlled random sampling for dynamic results ---
            # Use the current system time as the random seed to ensure different results on each run
            recos_df = recos_df.sample(
                n=sample_size, 
                random_state=int(time.time())
            )
            # -------------------------------------------------------------

            st.markdown(f"🏡 Recommended Properties Similar to Uploaded Image (**{predicted_class}** range):")

            # Create columns for the 3 recommendations
            cols = st.columns(sample_size)
            headers = {"User-Agent": "Mozilla/5.0 (compatible; StreamlitApp/1.0)"}

            for i, row in recos_df.iterrows():
                name = row.get("name", f"Property #{i+1}")
                price = row["price_per_night"]
                img_url = row["picture_url"]

                with cols[recos_df.index.get_loc(i)]:
                    st.markdown(f"**{name}**")
                    st.markdown(f"💵 **${price:.0f}/night**")

                    try:
                        response = requests.get(img_url, headers=headers, timeout=5)
                        response.raise_for_status()
                        rec_img = Image.open(BytesIO(response.content))
                        st.image(rec_img, caption=f"{predicted_class} Category")
                    except Exception:
                        st.warning("⚠️ Image load error.")
                        st.image("https://placehold.co/300x200/cccccc/333333?text=Image+Unavailable")

else:
    st.info("⬆️ Upload an image to start the property price prediction and view recommendations.")
