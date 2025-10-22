# app.py
import streamlit as st
import pandas as pd
import os, requests, shutil, math, io, time
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import StandardScaler
import joblib

# ---------------------------
# CONFIG
# ---------------------------
DATA_XLSX = "bukitvista_analyst.xlsx"  # make sure this file is in same folder
IMAGES_DIR = "images"
MODEL_DIR = "saved_model"
RANDOM_SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 8

st.set_page_config(page_title="Rental Recommender (Image -> beds/baths/price)", layout="wide")

# ---------------------------
# HELPERS
# ---------------------------
def ensure_dirs():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

def load_table():
    if not os.path.exists(DATA_XLSX):
        st.error(f"File {DATA_XLSX} tidak ditemukan. Letakkan file Excel di folder yang sama dengan app.py")
        st.stop()
    df = pd.read_excel(DATA_XLSX)
    return df

def download_image(url, out_path):
    try:
        if os.path.exists(out_path):
            return True
        resp = requests.get(url, stream=True, timeout=10)
        if resp.status_code == 200:
            with open(out_path, 'wb') as f:
                resp.raw.decode_content = True
                shutil.copyfileobj(resp.raw, f)
            return True
    except Exception as e:
        return False
    return False

def prepare_images(df, nrows=None, force_download=False):
    """
    Download images and return dataframe with local image path and targets.
    """
    ensure_dirs()
    df2 = df.copy()
    if nrows:
        df2 = df2.head(nrows)
    df2 = df2.reset_index(drop=True)

    local_paths = []
    ok_idx = []
    for i, row in df2.iterrows():
        url = row.get('picture_url')
        if not isinstance(url, str) or url.strip()=="":
            local_paths.append(None)
            continue
        # build filename
        ext = url.split('?')[0].split('.')[-1]
        filename = f"{i}.{ext}"
        out_path = os.path.join(IMAGES_DIR, filename)
        if force_download or (not os.path.exists(out_path)):
            ok = download_image(url, out_path)
            if not ok:
                local_paths.append(None)
                continue
        local_paths.append(out_path)
        ok_idx.append(i)
    df2['image_path'] = local_paths
    df2 = df2[df2['image_path'].notnull()].reset_index(drop=True)
    return df2

def build_dataset(df, target_cols=['beds','baths','price_per_night'], img_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True):
    """
    Build tf.data dataset yielding (image_tensor, targets_vector)
    targets vector is [beds, baths, price_scaled]
    price scaling is applied (standard scaler).
    Returns dataset, scaler (for price)
    """
    # read numeric columns; fillna
    df = df.copy()
    for c in target_cols:
        if c not in df.columns:
            raise ValueError(f"Target column {c} not in dataframe")
    df = df[[ 'image_path'] + target_cols].dropna().reset_index(drop=True)
    # scale price_per_night
    price_col = 'price_per_night'
    scaler = StandardScaler()
    prices = df[[price_col]].astype(float)
    price_scaled = scaler.fit_transform(prices)
    df['price_scaled'] = price_scaled
    # prepare lists
    image_paths = df['image_path'].tolist()
    beds = df['beds'].astype(float).tolist()
    baths = df['baths'].astype(float).tolist()
    prices_scaled = df['price_scaled'].astype(float).tolist()

    def _load_image(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, img_size)
        img = img / 255.0
        return img

    def gen():
        for p,b,ba,pr in zip(image_paths, beds, baths, prices_scaled):
            try:
                img = _load_image(p).numpy()
                yield img.astype(np.float32), np.array([b, ba, pr], dtype=np.float32)
            except Exception as e:
                continue

    output_signature = (tf.TensorSpec(shape=(*img_size,3), dtype=tf.float32),
                        tf.TensorSpec(shape=(3,), dtype=tf.float32))
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    if shuffle:
        ds = ds.shuffle(buffer_size=64, seed=RANDOM_SEED)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, scaler

def build_model(img_size=IMG_SIZE, lr=1e-4, train_base=False):
    base = EfficientNetB0(include_top=False, input_shape=(*img_size,3), weights='imagenet')
    base.trainable = train_base
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.2)(x)
    # shared representation
    shared = layers.Dense(128, activation='relu')(x)

    # beds head (regression)
    beds_out = layers.Dense(32, activation='relu')(shared)
    beds_out = layers.Dense(1, name='beds')(beds_out)

    # baths head (regression)
    baths_out = layers.Dense(32, activation='relu')(shared)
    baths_out = layers.Dense(1, name='baths')(baths_out)

    # price head (regression)
    price_out = layers.Dense(64, activation='relu')(shared)
    price_out = layers.Dense(1, name='price')(price_out)

    model = models.Model(inputs=base.input, outputs=[beds_out, baths_out, price_out])
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss={'beds':'mse','baths':'mse','price':'mse'},
                  metrics={'beds':'mae','baths':'mae','price':'mae'})
    return model

def train_model(df, epochs=10, batch_size=BATCH_SIZE, lr=1e-4, train_base=False):
    ds, scaler = build_dataset(df, batch_size=batch_size)
    # split small manually: create train/val from ds by splitting list
    # For simplicity, convert dataset to lists (since dataset likely small)
    imgs = []
    targets = []
    for b in ds:
        imgs.extend(b[0].numpy())
        targets.extend(b[1].numpy())
    imgs = np.array(imgs)
    targets = np.array(targets)
    if len(imgs) < 8:
        st.warning("Jumlah gambar sangat sedikit (<8). Training mungkin tidak stabil.")
    X_train, X_val, y_train, y_val = train_test_split(imgs, targets, test_size=0.2, random_state=RANDOM_SEED)
    # build model
    model = build_model(lr=lr, train_base=train_base)
    # prepare y dict
    y_train_dict = {'beds': y_train[:,0], 'baths': y_train[:,1], 'price': y_train[:,2]}
    y_val_dict = {'beds': y_val[:,0], 'baths': y_val[:,1], 'price': y_val[:,2]}
    hist = model.fit(X_train, y_train_dict, validation_data=(X_val, y_val_dict),
                     epochs=epochs, batch_size=batch_size)
    return model, scaler, hist

def predict_image(model, scaler, img_pil):
    img = img_pil.convert('RGB').resize(IMG_SIZE)
    arr = np.array(img)/255.0
    arr = np.expand_dims(arr, 0).astype(np.float32)
    preds = model.predict(arr)
    # preds are lists [beds_pred, baths_pred, price_scaled_pred]
    beds_pred = float(preds[0].squeeze())
    baths_pred = float(preds[1].squeeze())
    price_scaled_pred = float(preds[2].squeeze())
    # inverse scale
    price = scaler.inverse_transform([[price_scaled_pred]])[0][0]
    # round sensible
    beds_r = max(0, round(beds_pred))
    baths_r = max(0, round(baths_pred))
    price_r = max(0.0, round(price, 2))
    return {'beds':beds_r, 'baths':baths_r, 'price':price_r}

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("🏡 Rental Recommender — Upload gambar → rekomendasi beds / baths / price")

st.markdown("""
Aplikasi ini:
- Membaca `bukitvista_analyst.xlsx` (kolom `picture_url`, `beds`, `baths`, `price_per_night`).
- Melatih model CNN (transfer learning EfficientNetB0) untuk memprediksi: jumlah kamar tidur (`beds`), jumlah kamar mandi (`baths`), dan `price_per_night`.
- Upload gambar untuk mendapatkan prediksi rekomendasi.
""")

ensure_dirs()

with st.expander("1) Tinjau data (sample)"):
    try:
        df = load_table()
        st.dataframe(df[['picture_url','beds','baths','price_per_night']].head(10))
        st.caption(f"Total baris di Excel: {len(df)}")
    except Exception as e:
        st.error("Gagal membaca file Excel: " + str(e))

st.markdown("---")
col1, col2 = st.columns([2,1])

with col1:
    st.header("A. Persiapan dan Download Gambar")
    n_download = st.number_input("Berapa baris yang akan dipakai (0 = semua)", min_value=0, value=0, step=1)
    force_dl = st.checkbox("Force download ulang gambar (overwrite)", value=False)
    if st.button("Mulai download gambar"):
        df = load_table()
        nrows = None if n_download==0 else n_download
        with st.spinner("Mengunduh gambar..."):
            df_images = prepare_images(df, nrows=nrows, force_download=force_dl)
        st.success(f"Terunduh dan tersedia {len(df_images)} gambar (folder: {IMAGES_DIR})")
        st.write(df_images[['image_path','beds','baths','price_per_night']].head(20))

with col2:
    st.header("B. Training model")
    st.write("Opsi training (jika model sudah dilatih, Anda bisa skip)")
    epochs = st.number_input("Epochs", min_value=1, max_value=200, value=8)
    batch_size = st.selectbox("Batch size", options=[4,8,16], index=1)
    lr = st.number_input("Learning rate", min_value=1e-6, max_value=1e-2, value=1e-4, format="%.6f")
    train_base = st.checkbox("Unfreeze base EfficientNet (fine-tune)", value=False)
    use_tuner = st.checkbox("Gunakan Keras Tuner (opsional) — requires keras-tuner paket", value=False)
    if st.button("Latih model sekarang"):
        df = load_table()
        df_images = prepare_images(df, force_download=False)
        if len(df_images) < 4:
            st.warning("Dataset terlalu kecil (kurang dari 4 gambar). Tambahkan lebih banyak gambar untuk pelatihan yang bermakna.")
        with st.spinner("Training model (ini bisa memakan waktu tergantung komputer)..."):
            try:
                model, scaler, hist = train_model(df_images, epochs=epochs, batch_size=batch_size, lr=lr, train_base=train_base)
                # save model and scaler
                model.save(os.path.join(MODEL_DIR, "rental_model"))
                joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
                st.success("Training selesai — model disimpan di folder 'saved_model/rental_model'")
                st.write("Training history keys:", list(hist.history.keys()))
                st.line_chart(pd.DataFrame(hist.history))
            except Exception as e:
                st.error("Training gagal: " + str(e))

st.markdown("---")
st.header("C. Upload gambar untuk prediksi rekomendasi")
model_loaded = None
scaler_loaded = None

# Try to load saved model if exists
if os.path.exists(os.path.join(MODEL_DIR, "rental_model")):
    try:
        model_loaded = tf.keras.models.load_model(os.path.join(MODEL_DIR, "rental_model"))
        scaler_loaded = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    except Exception as e:
        st.warning("Gagal memuat model tersimpan: " + str(e))

uploaded = st.file_uploader("Upload gambar properti (jpg/png)", type=['jpg','jpeg','png'])
if uploaded is not None:
    try:
        img = Image.open(uploaded)
        st.image(img, caption="Gambar yang diupload", use_column_width=True)
        if model_loaded is None:
            st.warning("Model belum tersedia. Silakan latih model terlebih dahulu pada bagian B, atau muat model tersimpan.")
        else:
            with st.spinner("Memprediksi..."):
                res = predict_image(model_loaded, scaler_loaded, img)
                st.subheader("Hasil prediksi rekomendasi")
                st.write(f"Rekomendasi jumlah kamar tidur (beds): **{res['beds']}**")
                st.write(f"Rekomendasi jumlah kamar mandi (baths): **{res['baths']}**")
                st.write(f"Perkiraan harga sewa per malam (price_per_night): **{res['price']}**")
                st.balloons()
    except Exception as e:
        st.error("Gagal memproses gambar: " + str(e))

st.markdown("---")
st.write("Catatan akhir:")
st.write("""
- Dataset kecil (45 baris) → hasil prediksi akan terbatas.  
- Untuk hasil lebih baik: tambahkan lebih banyak gambar, lakukan label cleaning, atau gunakan pendekatan multi-modal (gabungkan fitur teks/fitur numerik dari Excel).
- Jika ingin saya tambahkan fitur klasifikasi (mis. prediksi kategori kamar sebagai integer terbatas), atau tuning otomatis (KerasTuner), beri tahu — saya bisa modifikasi.
""")
