# 🖼️ AI vs Real Image Detection  
**Sistem Deteksi Gambar AI dan Gambar Nyata Berbasis Machine Learning (SVM)**

Aplikasi web berbasis **Streamlit** untuk mendeteksi apakah sebuah gambar merupakan **hasil kecerdasan buatan (AI)** atau **gambar nyata**, menggunakan pendekatan **Machine Learning** dengan algoritma **Support Vector Machine (SVM)**.

---

## 🧠 Metode yang Digunakan

### 🔹 Algoritma
- **Support Vector Machine (SVM)**  

### 🔹 Ekstraksi Fitur
- **RGB Histogram**  
- **Histogram of Oriented Gradients (HOG)**  
- **Local Binary Pattern (LBP)** 

### 🔹 Augmentasi Data
- Original image
- Flip horizontal
- Rotasi 10°
- Zoom (85%)

---

## 🔄 Alur Sistem

1. Dataset gambar AI dan gambar nyata
2. Augmentasi data citra
3. Pra-pemrosesan (resize, grayscale, normalisasi)
4. Ekstraksi fitur (RGB Histogram, HOG, LBP)
5. Pembagian data latih dan data uji (80:20)
6. Pelatihan model SVM
7. Evaluasi performa model

---

## 🖥️ Fitur Aplikasi

### 🏠 Beranda
- Deskripsi aplikasi
- Alur Machine Learning
- Informasi dataset, algoritma, dan fitur

### 🔍 Prediksi Gambar
- Upload gambar (JPG, JPEG, PNG)
- Prediksi label:
  - **Gambar Hasil AI**
  - **Gambar Nyata**
- Menampilkan tingkat kepercayaan (confidence)

### 📊 Analisis Model
- Statistik dataset sebelum & sesudah augmentasi
- Visualisasi augmentasi gambar
- Tahapan pra-pemrosesan citra
- Visualisasi:
  - RGB Histogram
  - HOG
  - LBP
- Contoh vektor fitur
- Evaluasi performa model SVM

---
## Cara Menjalankan Aplikasi

1. Clone repository:
   ```bash
   git clone https://github.com/LouisJonathan88/ai-vs-real-image-detector.git

   cd ai-vs-real-image-detector
    ```
2. Install dependencies:
   ```bash
    pip install -r requirements.txt
    ```
3. Jalankan aplikasi:
    ```bash
     streamlit run app.py
     `

