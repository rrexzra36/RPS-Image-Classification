import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations to avoid compatibility issues
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Show only errors (not warnings/info)

import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
from PIL import Image
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

TF_ENABLE_ONEDNN_OPTS=0

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Deteksi Batu Gunting Kertas",
    page_icon="✌️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- Memuat Model (dengan cache agar tidak loading berulang) ---
@st.cache_resource
def load_my_model():
    """Memuat model Keras yang telah disimpan."""
    try:
        model_path = "model/rps_model.keras"  
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        st.error(f"Pastikan file model '{model_path}' ada di direktori yang sama dengan app.py")
        st.stop()

model = load_my_model()

# Definisikan nama kelas sesuai urutan saat pelatihan
CLASS_NAMES = ['paper', 'rock', 'scissors']

# --- Fungsi untuk Prediksi (Disesuaikan dengan Kode Anda) ---
def predict(file_buffer):
    """
    Fungsi ini sekarang meniru alur kerja dari notebook Anda secara persis,
    TANPA menggunakan preprocess_input.
    """
    # 1. Muat gambar dari buffer file menggunakan Keras
    img = keras_image.load_img(file_buffer, target_size=(150, 150))

    # 2. Ubah gambar menjadi array
    x = keras_image.img_to_array(img)

    # 3. Tambahkan dimensi batch
    x = np.expand_dims(x, axis=0)
    
    # 4. Buat tumpukan gambar (sama seperti np.vstack di notebook Anda)
    images = np.vstack([x])

    # 5. Lakukan prediksi pada gambar mentah (tanpa normalisasi)
    predictions = model.predict(images)
    
    # Dapatkan hasil
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence = np.max(predictions[0]) * 100

    # Kembalikan juga gambar yang telah diolah untuk ditampilkan
    return predicted_class_name, confidence, predictions[0], images

# --- Kelas untuk Transformasi Video Real-time (disesuaikan juga) ---
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.model = load_my_model()
        self.class_names = CLASS_NAMES

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (150, 150))
        
        # Lakukan preprocessing yang sama (tanpa preprocess_input)
        x = np.expand_dims(img_resized, axis=0)
        images = np.vstack([x])

        # Prediksi
        prediction = self.model.predict(images)
        class_idx = np.argmax(prediction[0])
        class_name = self.class_names[class_idx]
        confidence = np.max(prediction[0]) * 100

        text = f"{class_name}: {confidence:.2f}%"
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return img

# --- Antarmuka Utama Streamlit ---
st.title("✌️ Deteksi Real-time Batu, Gunting, Kertas")

# Pilihan mode di sidebar
mode = st.sidebar.radio(
    "Pilih Mode Deteksi:",
    ("Unggah Gambar", "Kamera Langsung")
)
st.sidebar.markdown("---")
st.sidebar.info("Aplikasi ini menggunakan model yang telah Anda latih.")

if mode == "Unggah Gambar":
    st.header("Unggah Gambar Tangan Anda")
    uploaded_file = st.file_uploader(
        "Pilih file gambar...", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar Asli yang Diunggah", use_column_width=True)
        st.write("")

        if st.button("Lakukan Prediksi"):
            with st.spinner("Menganalisis gambar..."):
                predicted_class, confidence, all_probabilities, processed_images = predict(uploaded_file)
                
                # --- Tampilkan Gambar yang Telah Diolah ---
                st.subheader("Gambar Setelah Diolah (Input untuk Model)")
                # Konversi array ke tipe data yang bisa ditampilkan (uint8)
                displayable_img = processed_images[0].astype(np.uint8)
                st.image(displayable_img, caption="Gambar yang Telah Diolah", use_column_width=True)
                st.write("---")

                # --- Tampilkan Hasil Prediksi ---
                st.subheader("Hasil Prediksi")
                st.success(f"Prediksi Utama: **{predicted_class.capitalize()}**")
                st.info(f"Tingkat Keyakinan: **{confidence:.2f}%**")
                st.write("---")
                
                st.subheader("Rincian Probabilitas per Kelas:")
                for i, class_name in enumerate(CLASS_NAMES):
                    probability = all_probabilities[i] * 100
                    st.write(f"**{class_name.capitalize()}**: {probability:.4f}%") # Menambahkan presisi

elif mode == "Kamera Langsung":
    st.header("Deteksi Menggunakan Kamera")
    st.warning("Pastikan Anda memberikan izin akses kamera pada browser Anda.")
    webrtc_streamer(
        key="rock-paper-scissors-cam",
        video_processor_factory=VideoProcessor, # PERBAIKAN: Menggunakan argumen yang baru
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    st.info("Arahkan tangan Anda ke kamera untuk memulai deteksi.")
