import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
import time
import os
import tempfile

# --- CONFIGURATION ---
DATASET_PATH = "bukitvista_analyst.xlsx"

# --- PAGE CONFIG ---
st.set_page_config(page_title="🏡 BukitVista Property Price Predictor (CNN)", layout="wide")
st.title("🏡 BukitVista Property Price Prediction App")
st.markdown("Upload your trained model, then upload a property image to predict its price category and see similar BukitVista listings.")

# --- LOAD DATASET (Cached) ---
@st.cache_data
def load_dataset(path):
    try:
        df = pd.read_excel(path, sheet_name="Sheet1")
        return df
    except FileNotFoundError:
        st.error(f"❌ Dataset file not found: **{path}**. Please ensure it is in the same folder.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Failed to load dataset: {e}")
        st.stop()

df = load_dataset(DATASET_PATH)
st.sidebar.success(f"✅ Dataset loaded successfully: {DATASET_PATH}")
st.sidebar.write("Columns detected:", df.columns.tolist())

# --- LOAD MODEL VIA UPLOAD ---
st.sidebar.header("🧠 Model Loader")
model_file = st.sidebar.file_uploader("Upload your trained model (.h5)", type=["h5"])

model = None
if model_file is not None:
    model_bytes = BytesIO(model_file.read())

    @st.cache_resource
    def load_uploaded_model(model_data):
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as temp_file:
                model_data.seek(0)
                temp_file.write(model_data.read())
                temp_file_path = temp_file.name

            model = tf.keras.models.load_model(temp_file_path)
            return model
        except Exception as e:
            st.error(f"❌ An error occurred loading the uploaded model: {e}")
            st.stop()
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    model = load_uploaded_model(model_bytes)

    if model:
        st.sidebar.success("✅ Model loaded successfully from upload.")

# Stop execution until model is uploaded
if model is None:
    st.stop()

# --- LABELS AND PRICE RANGES ---
class_labels = ["Low", "Medium", "High"]
price_ranges = {
    "Low": "$30–70 per night",
    "Medium": "$70–150 per night",
    "High": "$150+ per night",
}

# --- IMAGE PREPROCESSING ---
def preprocess_image(img, target_size=(128, 128)):
    img_resized = img.resize(target_size)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- PRICE CATEGORY ---
def categorize_price(price):
    if pd.isna(price):
        return "Unknown"
    price = float(price)
    if price < 70:
        return "Low"
    elif 70 <= price <= 150:
        return "Medium"
    else:
        return "High"

# --- UPLOAD IMAGE FOR PREDICTION ---
st.header("📤 Upload Property Image for Prediction")

image_formats = ["jpg", "jpeg", "png", "webp", "tiff", "tif", "bmp", "gif"]
uploaded_img = st.file_uploader("Choose a property image...", type=image_formats)

if uploaded_img is not None:
    img = Image.open(uploaded_img).convert("RGB")

    input_shape = model.input_shape
    target_size = (input_shape[1], input_shape[2]) if len(input_shape) == 4 else (128, 128)

    st.image(img, caption=f"🖼️ Uploaded Property Image (Resized to {target_size[0]}x{target_size[1]})")

    img_array = preprocess_image(img, target_size)

    # --- PREDICTION ---
    pred = model.predict(img_array)[0]
    predicted_class_index = np.argmax(pred)
    predicted_class = class_labels[predicted_class_index]
    confidence = np.max(pred) * 100

    # --- PREDICTION RESULT ---
    st.subheader("🎯 Prediction Result")
    st.markdown(f"**🏷️ Predicted class:** {predicted_class}")
    st.markdown(f"**💵 Estimated price range:** {price_ranges.get(predicted_class, 'N/A')}")
    st.markdown(f"**🔢 Confidence:** {confidence:.2f}%")
    st.markdown(f"**🔍 Probabilities:** {dict(zip(class_labels, [round(p * 100, 2) for p in pred]))}")

    # --- RECOMMENDATIONS ---
    st.header("🏘️ Similar Property Recommendations")

    if "price_per_night" not in df.columns or "picture_url" not in df.columns:
        st.error("❌ Dataset must contain 'price_per_night' and 'picture_url' columns for recommendations.")
    else:
        df["price_per_night"] = pd.to_numeric(df["price_per_night"], errors="coerce")
        df["price_class"] = df["price_per_night"].apply(categorize_price)

        recos_df = df[df["price_class"] == predicted_class]
        sample_size = min(3, len(recos_df))

        if recos_df.empty or sample_size == 0:
            st.warning(f"No similar properties found in the dataset for the '{predicted_class}' category.")
        else:
            recos_df = recos_df.sample(n=sample_size, random_state=int(time.time()))

            st.markdown(f"🏡 Recommended Properties Similar to Uploaded Image (**{predicted_class}** range):")

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
    st.info("⬆️ Please upload an image to start the property price prediction and view recommendations.")
