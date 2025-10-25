import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import requests
from io import BytesIO

# --- Page Setup ---
st.set_page_config(page_title="Property Price Predictor", layout="wide")
st.title("🏠 Property Price Prediction App")
st.markdown("Upload a property image to predict its price category and get similar recommendations.")

# --- Load Dataset ---
st.sidebar.header("📂 Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Dataset loaded successfully!")
    st.sidebar.write("Columns detected:", df.columns.tolist())
else:
    st.warning("Please upload a dataset to continue.")
    st.stop()

# --- Load Model ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("price_image_classifier.h5")

model = load_model()

# --- Class Labels & Price Ranges ---
class_labels = ['Low', 'Medium', 'High']
price_ranges = {
    "Low": "$30–70 per night",
    "Medium": "$70–150 per night",
    "High": "$150+ per night"
}

# --- Image Upload Section ---
st.header("📤 Upload Property Image for Prediction")
uploaded_img = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_img is not None:
    img = Image.open(uploaded_img).convert("RGB")
    st.image(img, caption="🖼️ Uploaded Image", use_column_width=True)

    # --- Preprocess Image ---
    img_resized = img.resize((128, 128))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # --- Prediction ---
    pred = model.predict(img_array)[0]
    predicted_class = class_labels[np.argmax(pred)]
    confidence = np.max(pred) * 100

    st.subheader("🎯 Prediction Result")
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Estimated Price Range:** {price_ranges[predicted_class]}")
    st.write(f"**Confidence:** {confidence:.2f}%")
    st.write("**Class Probabilities:**", dict(zip(class_labels, [f"{p*100:.2f}%" for p in pred])))

    # --- Recommendation Section ---
    st.header("🏡 Recommended Properties")

    if 'price_per_night' not in df.columns or 'picture_url' not in df.columns:
        st.error("❌ Dataset must contain 'price_per_night' and 'picture_url' columns.")
    else:
        # Categorize price into Low/Medium/High
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
            st.warning("No recommendations available for this price category.")
        else:
            recos_df = recos_df.sample(n=min(3, len(recos_df)))
            for i, row in recos_df.iterrows():
                name = row.get("name", f"Property #{i+1}")
                price = row["price_per_night"]
                img_url = row["picture_url"]

                col1, col2 = st.columns([1, 3])
                with col1:
                    try:
                        headers = {"User-Agent": "Mozilla/5.0"}
                        response = requests.get(img_url, headers=headers, timeout=10)
                        response.raise_for_status()
                        rec_img = Image.open(BytesIO(response.content))
                        st.image(rec_img, use_container_width=True)
                    except Exception as e:
                        st.warning(f"⚠️ Image unavailable for {name}")
                with col2:
                    st.markdown(f"### {name}")
                    st.markdown(f"💵 **${price}/night**")
                    st.markdown(f"🏷️ **Category:** {predicted_class}")

else:
    st.info("Please upload an image to start prediction.")
