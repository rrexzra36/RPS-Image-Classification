# Klasifikasi Gambar Batu, Gunting, Kertas

## ***Business Understanding***

Interaksi antara manusia dan komputer terus berkembang menuju antarmuka yang lebih natural dan intuitif. Salah satu area eksplorasi yang paling menjanjikan adalah penggunaan gestur tangan sebagai metode input, menggantikan atau melengkapi penggunaan mouse dan keyboard. Kemampuan sistem untuk mengenali gestur secara akurat dapat membuka berbagai kemungkinan aplikasi baru, mulai dari permainan interaktif, alat bantu bagi penyandang disabilitas, hingga sistem kontrol di lingkungan industri atau medis.

## **Permasalahan Bisnis**

Tantangan utamanya adalah membangun sebuah sistem yang dapat diandalkan untuk mengenali gestur tangan secara visual. Proyek ini berfokus pada pengembangan sebuah model *computer vision* sebagai fondasi untuk teknologi ini.

## **Cakupan Proyek**

Proyek ini mencakup keseluruhan alur kerja pengembangan model *deep learning*, mulai dari persiapan data, pembangunan arsitektur model, pelatihan, evaluasi, hingga implementasinya dalam sebuah aplikasi web interaktif.

### Pemahaman Data (*Data Understanding*)
Dataset yang digunakan adalah "Rock, Paper, Scissors" yang berisi 2.188 gambar gestur tangan. Dataset ini terbagi secara merata ke dalam tiga kelas:

- Kertas (*Paper*): 712 gambar
- Batu (*Rock*): 726 gambar
- Gunting (*Scissors*): 750 gambar

### Persiapan Data (*Data Preaparation*)
Tahap persiapan data dilakukan menggunakan `tf.keras.utils.image_dataset_from_directory` yang secara otomatis menangani pelabelan berdasarkan struktur direktori. Prosesnya meliputi:

- Dataset dibagi menjadi 80% data pelatihan (1751 gambar) dan 20% data validasi (437 gambar).
- Untuk mencegah overfitting dan meningkatkan kemampuan generalisasi model, data pelatihan diaugmentasi secara real-time dengan teknik seperti rotasi acak, zoom acak, pembalikan horizontal, dan penyesuaian kontras.
- Meskipun tidak menggunakan `preprocess_input` saat inferensi sesuai permintaan, model ini dilatih dengan lapisan `preprocess_input` dari `MobileNetV2` yang menormalkan nilai piksel gambar ke rentang [-1, 1].

### *Machine Learning Modeling*
Model ini dibangun menggunakan pendekatan **Transfer Learning** untuk memanfaatkan fitur-fitur yang telah dipelajari oleh model yang lebih besar.

- `MobileNetV2` yang telah dilatih pada dataset `ImageNet` digunakan sebagai model dasar. Lapisan klasifikasi teratasnya tidak digunakan (`include_top=False`).
- Bobot dari base_model dibekukan (*frozen*) agar tidak berubah selama tahap pelatihan awal. Di atas model dasar, ditambahkan beberapa *custom layer*:
  - `GlobalAveragePooling2D`: Untuk meratakan output fitur.
  - `Dropout`: Untuk mengurangi overfitting.
  - `Dense`: Lapisan terhubung penuh dengan 1024 neuron dan aktivasi `ReLU`.
- *Output layer* dengan 3 neuron (menyesuaikan jumlah kelas) dan aktivasi `softmax` untuk menghasilkan probabilitas.
- *Compiling model* menggunakan *optimizer* `Adam` dan fungsi *loss* `SparseCategoricalCrossentropy`.

### *Evaluation*
Performa model dievaluasi berdasarkan metrik akurasi pada set data validasi. Setelah 10 epoch pelatihan, model berhasil mencapai **akurasi validasi sekitar 97-98%**. Grafik akurasi dan loss menunjukkan bahwa model belajar dengan baik tanpa mengalami *overfitting* yang signifikan, di mana metrik pada data pelatihan dan validasi bergerak ke arah yang sama.

### *Deployment*
Model yang telah dilatih dan disimpan dalam format `.keras` (`rps_model.keras`) di-deploy sebagai aplikasi web menggunakan Streamlit. Aplikasi ini (`app.py`) menyediakan *user-interface* yang sederhana dengan dua mode operasi:

- Upload Gambar: Pengguna dapat mengunggah file gambar (JPG, JPEG, PNG) untuk diprediksi.
- Live Camera: Pengguna dapat mengaktifkan webcam untuk mendapatkan prediksi gestur tangan secara real-time.


## **Persiapan**

Sumber data pelatihan: 
Dicoding via Github
```
!wget --no-check-certificate \
    https://github.com/dicodingacademy/assets/releases/download/release/rockpaperscissors.zip \
    -O /tmp/rockpaperscissors.zip
```

*Setup environment*:
```
// virtual enviroment setup
python -m venv .env --> membuat virtual enviroment
.env\Scripts\activate --> mengaktifkan virtual enviroment
pip install -r requirements.txt --> instal requirements

// additional commad
pip list --> melihat library yang terinstal
deactivate --> mematikan virtual enviroment
Remove-Item -Recurse -Force .\.env --> menghapus virtual enviroment
```


## **Menjalankan Sistem *Machine Learning***

Aplikasi ini dibangun dengan Streamlit dan dapat dijalankan secara lokal. Cara menjalankan Aplikasinya adalah sebagai berikut:

1. Pastikan Anda telah menginstal semua pustaka yang diperlukan dari bagian Persiapan.
2. Letakkan file model yang telah dilatih (best_model.h5) di dalam direktori yang sama dengan file aplikasi (`app.py`).
3. Buka terminal atau command prompt, arahkan ke direktori proyek.
4. Jalankan perintah berikut:
    ```
    streamlit run app.py
    ```
5. Aplikasi akan otomatis terbuka di browser default Anda.
6. Gunakan sidebar di sebelah kiri untuk memilih antara mode "Unggah Gambar" atau "Kamera Langsung".

## ***Conclusion***

Proyek ini berhasil mengembangkan dan mendeploy sebuah model klasifikasi gambar yang akurat untuk gestur tangan batu, gunting, dan kertas. Dengan menggunakan *transfer learning*, model mencapai akurasi validasi yang tinggi (~98%) dengan waktu pelatihan yang efisien. Aplikasi Streamlit yang dihasilkan menyediakan platform yang interaktif dan mudah digunakan untuk demonstrasi kemampuan model, baik melalui prediksi statis dari gambar maupun deteksi dinamis dari kamera.