# 👁️ Perbandingan ResNet50 dan EfficientNetB0 dengan Explainable AI Grad-CAM++ untuk Klasifikasi Penyakit Retina pada Citra OCT

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

Aplikasi web interaktif untuk membandingkan performa arsitektur **ResNet50** dan **EfficientNetB0** dalam mengklasifikasikan penyakit retina pada citra Optical Coherence Tomography (OCT), dilengkapi dengan teknik Explainable AI **Grad-CAM++** untuk visualisasi area perhatian model.

---

## 📋 Daftar Isi

- [Demo](#-demo)
- [Fitur](#-fitur)
- [Dataset](#-dataset)
- [Arsitektur Model](#-arsitektur-model)
- [Grad-CAM++](#-grad-cam)
- [Instalasi](#-instalasi)
- [Cara Penggunaan](#-cara-penggunaan)
- [Struktur Repositori](#-struktur-repositori)
- [Hasil Evaluasi](#-hasil-evaluasi)
- [Referensi](#-referensi)

---

## 🚀 Demo

Jalankan aplikasi secara lokal:

```bash
streamlit run app.py
```

---

## ✨ Fitur

- **🔬 Prediksi Real-time** — Upload citra OCT, kedua model (ResNet50 & EfficientNetB0) langsung memprediksi kelas penyakit beserta confidence score secara berdampingan
- **🗺️ Grad-CAM++ Visualization** — Heatmap dan overlay visual yang menunjukkan area fokus model saat membuat keputusan
- **📊 Analisis 5 Benar + 5 Salah** — Kumpulkan contoh prediksi benar dan salah, generate analisis Grad-CAM++ lengkap dengan teks analisis visual otomatis
- **📈 Perbandingan Metrik** — Input hasil evaluasi (Accuracy, Precision, Recall, F1, AUC-ROC) dan tampilkan grafik perbandingan kedua model
- **🏷️ 8 Kelas Penyakit Retina** — CNV, DME, DRUSEN, NORMAL, MH, DR, SCR, AMD

---

## 🗂️ Dataset

Dataset yang digunakan adalah citra **Optical Coherence Tomography (OCT)** retina dengan **8 kelas** penyakit:

| Kelas | Nama Lengkap | Deskripsi |
|-------|-------------|-----------|
| **CNV** | Choroidal Neovascularization | Pertumbuhan pembuluh darah abnormal di bawah retina |
| **DME** | Diabetic Macular Edema | Pembengkakan makula akibat komplikasi diabetes |
| **DRUSEN** | Drusen | Endapan lipid/protein di bawah epitel pigmen retina |
| **NORMAL** | Normal | Retina sehat tanpa tanda-tanda patologi |
| **MH** | Macular Hole | Lubang kecil di pusat makula (fovea) |
| **DR** | Diabetic Retinopathy | Kerusakan pembuluh darah retina akibat diabetes |
| **SCR** | Serous Chorioretinopathy | Akumulasi cairan serosa di bawah retina |
| **AMD** | Age-related Macular Degeneration | Degenerasi makula terkait usia |

---

## 🧠 Arsitektur Model

### ResNet50
- **Tahun:** 2015 (He et al.)
- **Parameter:** ~25.6 juta
- **Inovasi:** Residual connections (skip connections) untuk mengatasi vanishing gradient
- **Blok Utama:** Bottleneck Block (1×1 → 3×3 → 1×1 Conv)

### EfficientNetB0
- **Tahun:** 2019 (Tan & Le)
- **Parameter:** ~5.3 juta
- **Inovasi:** Compound Scaling — menyeimbangkan depth, width, dan resolution secara bersamaan
- **Blok Utama:** MBConv (Mobile Inverted Bottleneck + Squeeze-and-Excitation)

Kedua model menggunakan **transfer learning** dari bobot ImageNet dengan fine-tuning pada dataset OCT retina.

---

## 🔍 Grad-CAM++

**Grad-CAM++** (Gradient-weighted Class Activation Mapping++) adalah peningkatan dari Grad-CAM yang menggunakan kombinasi bobot alpha berdasarkan turunan orde-dua:

```
α_k^c = (∂²y^c / ∂A_ij^k)² / (2·(∂²y^c / ∂A_ij^k)² + Σ A_ab^k·(∂³y^c / ∂A_ij^k)³)
```

Grad-CAM++ menghasilkan lokalisasi yang lebih presisi dibandingkan Grad-CAM, terutama untuk objek yang muncul beberapa kali dalam satu citra atau objek berukuran kecil — kondisi yang relevan pada deteksi lesi retina.

---

## ⚙️ Instalasi

### 1. Clone repositori

```bash
git clone https://github.com/USERNAME/NAMA-REPO.git
cd NAMA-REPO
```

### 2. Buat virtual environment (opsional tapi disarankan)

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Jalankan aplikasi

```bash
streamlit run app.py
```

> **Catatan:** Model akan otomatis diunduh dari Google Drive saat pertama kali dijalankan. Pastikan koneksi internet tersedia.

---

## 📖 Cara Penggunaan

### Tab 1 — Prediksi Gambar
1. Upload 1 gambar OCT dari data test kamu (JPG/PNG)
2. Sistem otomatis menjalankan prediksi pada **kedua model** secara bersamaan
3. Lihat hasil: kelas prediksi, confidence score, distribusi probabilitas top-5
4. Klik **"Generate Grad-CAM++ Kedua Model"** untuk melihat visualisasi heatmap berdampingan

### Tab 2 — Grad-CAM++ Analysis
1. Pilih model (ResNet50 atau EfficientNetB0) untuk analisis
2. Upload gambar satu per satu, pilih **True Label** via dropdown
3. Klik **"Simpan"** — sistem otomatis mengkategorikan sebagai prediksi benar/salah
4. Ulangi hingga terkumpul **5 benar + 5 salah**
5. Klik **"Generate Analisis Grad-CAM++"** untuk menampilkan figure lengkap + analisis teks

### Tab 3 — Perbandingan Model
1. Masukkan metrik evaluasi hasil training (Accuracy, Precision, Recall, F1, AUC-ROC)
2. Klik **"Tampilkan Grafik Perbandingan"** untuk visualisasi bar chart
3. Baca kesimpulan otomatis yang dihasilkan sistem

---

## 📁 Struktur Repositori

```
├── app.py                  # Aplikasi Streamlit utama
├── requirements.txt        # Dependencies Python
├── README.md               # Dokumentasi ini
├── models/                 # Folder model (auto-download dari Google Drive)
│   ├── resnet50_retina.h5
│   └── efficientnetb0_retina.h5
└── notebooks/              # (opsional) Notebook training & evaluasi
    ├── training_resnet50.ipynb
    └── training_efficientnetb0.ipynb
```

---

## 📊 Hasil Evaluasi

> Isi tabel ini dengan hasil aktual dari notebook training kamu.

| Metrik | ResNet50 | EfficientNetB0 |
|--------|----------|----------------|
| **Accuracy** | - % | - % |
| **Precision** | - % | - % |
| **Recall** | - % | - % |
| **F1-Score** | - % | - % |
| **AUC-ROC** | - % | - % |

---

## 🛠️ Tech Stack

- **Framework ML:** TensorFlow / Keras
- **Web App:** Streamlit
- **Explainability:** Grad-CAM++ (implementasi custom dengan TensorFlow GradientTape)
- **Visualisasi:** Matplotlib, OpenCV
- **Model Hosting:** Google Drive + gdown

---

## 📚 Referensi

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR.
2. Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML.
3. Chattopadhay, A., et al. (2018). *Grad-CAM++: Generalized Gradient-Based Visual Explanations for Deep Convolutional Networks.* WACV.
4. Kermany, D. S., et al. (2018). *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning.* Cell.

---

## 👤 Author

**[Nama Kamu]**  
[Universitas / Institusi]  
[Email] · [LinkedIn] · [GitHub]

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
