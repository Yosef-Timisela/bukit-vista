import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import requests
from io import BytesIO

# --- PAGE CONFIG ---
st.set_page_config(page_title="🏡 BukitVista Property Price Predictor", layout="wide")
st.title("🏡 BukitVista Property Price Prediction App")
st.markdown("Upload a property image to predict its price category and see similar BukitVista listings.")

# --- LOAD DATASET AUTOMATICALLY ---
@st.cache_data
def load_dataset():
    df = pd.read_excel("bukitvista_analyst.xlsx", sheet_name="Sheet1")
    return df

try:
    df = load_dataset()
    st.sidebar.success("✅ Dataset loaded successfully: bukitvista_analyst.xlsx")
    st.sidebar.write("Columns detected:", df.columns.tolist())
except Exception as e:
    st.error(f"❌ Failed to load dataset: {e}")
    st.stop()

# --- LOAD MODEL ---
st.sidebar.header("🧠 Model Loader")
model_file = st.sidebar.file_uploader("Upload your trained model (.h5)", type=["h5"])

if model_file is not None:
    @st.cache_resource
    def load_model():
        return tf.keras.models.load_model(model_file)
    model = load_model()
    st.sidebar.success("✅ Model loaded successfully.")
else:
    st.sidebar.warning("Please upload your trained model (.h5) file to continue.")
    st.stop()

# --- LABELS AND PRICE RANGES ---
class_labels = ['Low', 'Medium', 'High']
price_ranges = {
    "Low": "$30–70 per night",
    "Medium": "$70–150 per night",
    "High": "$150+ per night"
}

# --- UPLOAD IMAGE FOR PREDICTION ---
st.header("📤 Upload Property Image for Prediction")
uploaded_img = st.file_uploader("Choose a property image...", type=["jpg", "jpeg", "png"])

if uploaded_img is not None:
    img = Image.open(uploaded_img).convert("RGB")
    st.image(img, caption="🖼️ Uploaded Property Image", use_container_width=True)

    # Preprocess
    img_resized = img.resize((128, 128))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array)[0]
    predicted_class = class_labels[np.argmax(pred)]
    confidence = np.max(pred) * 100

    st.subheader("🎯 Prediction Result")
    st.write(f"**Predicted Category:** {predicted_class}")
    st.write(f"**Estimated Price Range:** {price_ranges[predicted_class]}")
    st.write(f"**Confidence:** {confidence:.2f}%")
    st.write("**Class Probabilities:**", dict(zip(class_labels, [f'{p*100:.2f}%' for p in pred])))

    # --- PROPERTY RECOMMENDATIONS ---
    st.header("🏘️ Similar Property Recommendations")

    if 'price_per_night' not in df.columns or 'picture_url' not in df.columns:
        st.error("❌ Dataset must contain 'price_per_night' and 'picture_url' columns.")
    else:
        def categorize_price(price):
            if price < 70:
                return 'Low'
            elif 70 <= price <= 150:
                return 'Medium'
            else:
                return 'High'

        df['price_class'] = df['price_per_night'].apply(categorize_price)
        recos_df = df[df['price_class'] == predicted_class]

        if recos_df.empty:
            st.warning("No similar properties found for this category.")
        else:
            recos_df = recos_df.sample(n=min(3, len(recos_df)))
            for i, row in recos_df.iterrows():
                name = row.get("name", f"Property #{i+1}")
                price = row["price_per_night"]
                img_url = row["picture_url"]

                col1, col2 = st.columns([1, 2])
                with col1:
                    try:
                        headers = {"User-Agent": "Mozilla/5.0"}
                        response = requests.get(img_url, headers=headers, timeout=10)
                        response.raise_for_status()
                        rec_img = Image.open(BytesIO(response.content))
                        st.image(rec_img, use_container_width=True)
                    except Exception as e:
                        st.warning(f"⚠️ Could not load image for {name}")

                with col2:
                    st.markdown(f"### {name}")
                    st.markdown(f"💵 **${price}/night**")
                    st.markdown(f"🏷️ **Category:** {predicted_class}")
else:
    st.info("Please upload an image to start prediction.")
