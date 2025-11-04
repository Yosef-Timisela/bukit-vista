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
import gzip

# ==============================
# 🏡 BukitVista Property Price Predictor (CNN - TFLite Version)
# ==============================

# --- CONFIGURATION ---
DATASET_PATH = "bukitvista_analyst.xlsx"
DEFAULT_MODEL_PATH = "models/price_image_classifier_quant.tflite.gz"

# --- PAGE CONFIG ---
st.set_page_config(page_title="🏡 BukitVista Property Price Predictor (CNN)", layout="wide")
st.title("🏡 BukitVista Property Price Prediction App")
st.markdown("Upload your trained model (TFLite), then upload a property image to predict its price category and see similar BukitVista listings.")

# ==============================
# 📊 LOAD DATASET (CACHED)
# ==============================
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

# ==============================
# 🧠 LOAD MODEL (TFLITE + GZIP)
# ==============================
st.sidebar.header("🧠 Model Loader")
model_file = st.sidebar.file_uploader("Upload your trained model (.tflite or .tflite.gz)", type=["tflite", "gz"])

@st.cache_resource
def load_tflite_model(model_bytes):
    """Load a TensorFlow Lite model from bytes (optionally gzipped)."""
    try:
        if model_bytes[:2] == b'\x1f\x8b':  # gzip magic bytes
            with gzip.GzipFile(fileobj=BytesIO(model_bytes), mode='rb') as gz:
                model_content = gz.read()
        else:
            model_content = model_bytes

        interpreter = tf.lite.Interpreter(model_content=model_content)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"❌ Failed to load TFLite model: {e}")
        st.stop()

# Try to load uploaded model, else default model
interpreter = None
if model_file:
    interpreter = load_tflite_model(model_file.read())
    st.sidebar.success("✅ Uploaded model loaded successfully.")
elif os.path.exists(DEFAULT_MODEL_PATH):
    with open(DEFAULT_MODEL_PATH, "rb") as f:
        interpreter = load_tflite_model(f.read())
    st.sidebar.success("✅ Default model loaded successfully from repository.")
else:
    st.warning("⚠️ Please upload or include a TFLite model file.")
    st.stop()

# ==============================
# 🧮 PREDICTION HELPER
# ==============================
def predict_tflite(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# ==============================
# 🏷️ LABELS & PRICE RANGES
# ==============================
class_labels = ["Low", "Medium", "High"]
price_ranges = {
    "Low": "$30–70 per night",
    "Medium": "$70–150 per night",
    "High": "$150+ per night",
}

# ==============================
# 🖼️ IMAGE PREPROCESSING
# ==============================
def preprocess_image(img, target_size=(128, 128)):
    img_resized = img.resize(target_size)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array

# ==============================
# 💰 PRICE CATEGORIZATION
# ==============================
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

# ==============================
# 📤 IMAGE UPLOAD & PREDICTION
# ==============================
st.header("📤 Upload Property Image for Prediction")

image_formats = ["jpg", "jpeg", "png", "webp", "tiff", "tif", "bmp", "gif"]
uploaded_img = st.file_uploader("Choose a property image...", type=image_formats)

if uploaded_img is not None:
    img = Image.open(uploaded_img).convert("RGB")

    # Try to infer input size from model (or fallback)
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    target_size = (input_shape[1], input_shape[2]) if len(input_shape) == 4 else (128, 128)

    st.image(img, caption=f"🖼️ Uploaded Property Image (Resized to {target_size[0]}x{target_size[1]})")

    img_array = preprocess_image(img, target_size)

    # --- PREDICTION ---
    pred = predict_tflite(interpreter, img_array)[0]
    predicted_class_index = np.argmax(pred)
    predicted_class = class_labels[predicted_class_index]
    confidence = np.max(pred) * 100

    # --- DISPLAY RESULT ---
    st.subheader("🎯 Prediction Result")
    st.markdown(f"**🏷️ Predicted class:** {predicted_class}")
    st.markdown(f"**💵 Estimated price range:** {price_ranges.get(predicted_class, 'N/A')}")
    st.markdown(f"**🔢 Confidence:** {confidence:.2f}%")
    st.markdown(f"**🔍 Probabilities:** {dict(zip(class_labels, [round(p * 100, 2) for p in pred]))}")

    # ==============================
    # 🏘️ SIMILAR PROPERTY RECOMMENDATIONS
    # ==============================
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
