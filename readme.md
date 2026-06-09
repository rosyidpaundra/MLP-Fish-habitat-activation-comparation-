Berikut adalah isi lengkap untuk file `README.md` yang super lengkap, profesional, dan siap Anda *copy-paste* langsung ke repositori GitHub Anda.

Semua teks, diagram kontes, struktur repositori, hingga petunjuk teknis untuk mahasiswa sudah disesuaikan dengan skema penelitian yang ada pada foto dan kode awal Anda.

---

### 📄 Salin Teks di Bawah Ini ke File `README.md` Anda:

```markdown
# Sensitivity Analysis of MLP Activation Functions for Pelagic Habitat Modeling in FMA 573

[![QGIS](https://img.shields.io/badge/QGIS-3.x-green.svg)](https://qgis.org/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Latest-orange.svg)](https://scikit-learn.org/)

Repositori ini berisi pustaka kode Python untuk **QGIS Processing Toolbox** yang dirancang untuk melakukan **Species Distribution Modeling (SDM)** pelagis di Wilayah Pengelolaan Perikanan Negara Republik Indonesia (**WPP NRI 573 / FMA 573**). 

Riset ini berfokus pada **Analisis Sensitivitas Fungsi Aktivasi (Tanh, ReLU, dan Sigmoid)** pada algoritma *Multi-Layer Perceptron (MLP)* untuk memetakan Zona Potensial Penangkapan Ikan (ZPPI) berbasis data spasial multi-sensor.

---

## 📌 1. Skema & Kerangka Penelitian

Sebagian besar literatur global menggunakan fungsi aktivasi bawaan (*default*) pada jaringan saraf tiruan tanpa adanya justifikasi empiris berbasis bukti lokal. Repositori ini memecah batasan tersebut (*research gap*) dengan menyediakan skema pengujian komparatif tiga fungsi aktivasi kustom pada zona dinamika *upwelling* selatan Jawa.


```

```
   [ DATA INPUT ]                  [ PRE-PROCESSING ]                  [ MODEL MLP ]                   [ EVALUASI & HASIL ]

```

┌─────────────────────────┐     ┌────────────────────────────┐     ┌────────────────────────┐     ┌───────────────────────────┐
│ • Variabel X: GEE       │     │ • Pembersihan Outlier IQR  │     │ • Model A: Tanh        │     │ • Metrik: Akurasi, F1,    │
│   (SST, CHL-a, SSH, Sal,│────>│ • Data Augmentation (0.12) │────>│ • Model B: ReLU        │────>│   Precision, Recall       │
│    Arus, Bathy, Karang) │     │ • Pseudo-Absence Generation│     │ • Model C: Sigmoid     │     │ • Kurva ROC-AUC (.png)    │
│ • Variabel Y: Logbook   │     │ • Robust Scaling           │     │ • Hyperparameter Tetap │     │ • Peta Raster Probabilitas│
│   DKP Cilacap           │     └────────────────────────────┘     └────────────────────────┘     └───────────────────────────┘
└─────────────────────────┘

```

### Karakteristik Fungsi Aktivasi yang Diuji:
1. **Tanh (Hyperbolic Tangent):** Memiliki sifat *zero-centered* (output antara -1 hingga +1), mempermudah konvergensi model pada data oseanografi terstandardisasi.
2. **ReLU (Rectified Linear Unit):** Komputasi sangat cepat (output $a \ge 0$), namun memiliki risiko *dying ReLU* jika variasi noise spasial terlalu ekstrem.
3. **Sigmoid:** Secara alami memetakan nilai ke dalam skala probabilitas habitat ($0$ hingga $1$), namun rentan terhadap risiko *vanishing gradient* pada arsitektur tertentu.

---

## 📁 2. Struktur Repositori

Pustaka ini dibagi menjadi 3 file instalan kustom agar dapat dimasukkan ke dalam QGIS secara terpisah:

```text
├─── scripts/                  # Script Utama untuk QGIS Processing Toolbox
│    ├── mlp_target_tanh.py    # Algoritma MLP dengan aktivasi Tanh
│    ├── mlp_target_relu.py    # Algoritma MLP dengan aktivasi ReLU
│    └── mlp_target_sigmoid.py # Algoritma MLP dengan aktivasi Sigmoid
├─── docs/                     # Dokumentasi tambahan & Gambar Kurva ROC
└─── README.md                 # Dokumentasi utama proyek

```

---

## 🛠️ 3. Prasyarat Sistem & Instalasi (Panduan Mahasiswa)

Sebelum menjalankan *toolbox* ini di dalam QGIS, Anda wajib menginstal pustaka analisis data Python pada sistem QGIS Anda.

### Langkah 1: Instalasi Pustaka Python via OSGeo4W Shell

Jika Anda menggunakan Windows, buka **OSGeo4W Shell** (cari di menu Start dengan hak akses *Administrator*), lalu jalankan perintah berikut:

```bash
pip install numpy pandas scikit-learn matplotlib

```

*Bagi pengguna Mac/Linux, jalankan perintah `pip install` tersebut pada terminal yang terintegrasi dengan lingkungan Python milik QGIS Anda.*

### Langkah 2: Memasukkan Script ke QGIS Processing Toolbox

1. Jalankan aplikasi **QGIS** (Disarankan versi 3.22 LTR atau yang lebih baru).
2. Buka panel **Processing Toolbox** (Jika belum muncul, tekan `Ctrl + Alt + T` atau klik menu *Processing* -> *Toolbox*).
3. Di bagian atas panel Toolbox, klik ikon **Python** (ular hijau) dan pilih **"Add Script from File..."**.
4. Pilih salah satu atau ketiga file `.py` yang ada di dalam folder `scripts/` dari repositori ini.
5. Algoritma akan otomatis muncul di dalam rumpun menu **Perikanan** -> **mlp_anti_overfitting** pada panel toolbox Anda.

---

## 🚀 4. Panduan Pengoperasian (Step-by-Step)

Ketiga script membutuhkan parameter input yang **identik**, sehingga Anda dapat menjalankannya secara bergantian dengan data yang sama untuk mendapatkan perbandingan metrik yang adil.

### Parameter Input yang Dibutuhkan:

* **Titik Tangkapan (Layer Vektor):** Data titik koordinat (Point) lokasi penangkapan ikan pelagis asli (bernilai `1` atau berbasis data kehadiran logbook DKP Cilacap).
* **7 Layer Raster Lingkungan:**
1. *Raster SST* (Sea Surface Temperature / Suhu Permukaan Laut)
2. *Raster Chlorophyll-a* (Konsentrasi Klorofil-a)
3. *Raster Batimetri* (Kedalaman Laut)
4. *Raster Coral Reef* (Jarak/Peta Terumbu Karang)
5. *Raster Salinitas* (Kadar Garam Air Laut)
6. *Raster Sea Surface Elevation / Height* (Tinggi Muka Air Laut)
7. *Raster Sea Current Velocity* (Kecepatan Arus Laut)


* **Hasil Prediksi MLP (Raster Destination):** Tentukan lokasi penyimpanan dan nama file output `.tif` Anda.

### Cara Menjalankan:

1. Klik dua kali pada nama alat (misal: `MLP Fishing Ground - Tanh Activation`).
2. Masukkan semua *layer* sesuai dengan kolom parameternya masing-masing.
3. Klik **Run**.

---

## 📊 5. Hasil Evaluasi & Output Penelitian

Setiap kali proses selesai dijalankan, algoritma secara otomatis akan menghasilkan 3 luaran ilmiah utama:

1. **Peta Probabilitas Habitat Spasial:** Sebuah layer raster baru berbentuk format GeoTIFF (`.tif`) dengan nilai piksel berskala kontinu antara `0.0` (tidak sesuai) hingga `1.0` (sangat berpotensi/sesuai).
2. **Grafik Kurva ROC (`.png`):** File gambar grafik performa model yang otomatis tersimpan di folder yang sama dengan file raster output Anda. Nama file akan otomatis menyesuaikan fungsi aktivasi, contoh: `roc_curve_tanh.png`.
3. **Analisis Nilai Penting Variabel (Permutation Importance):** Log teks yang tercetak pada panel QGIS Log Messages yang mengurutkan parameter oseanografi mana yang paling berpengaruh terhadap keberadaan ikan pelagis.

---

## 🎓 6. Kutipan & Sitasi (Untuk Keperluan Akademik)

Bagi mahasiswa, dosen, atau peneliti yang menggunakan, memodifikasi, atau mereplikasi skema dari kode di repositori ini untuk keperluan skripsi, tesis, atau publikasi jurnal ilmiah, harap menyertakan sitasi berikut:

```text
Paundra, R. (2026). Sensitivity Analysis of MLP Activation Functions for Pelagic Habitat Modeling in the Upwelling Zone of FMA 573. GitHub Repository: [Masukkan Link Repo Anda Disini]

```

---

*Dikembangkan dengan 💙 untuk mendukung Data Science Kelautan, Spatial Computing, dan Keberlanjutan Perikanan Tangkap Indonesia.*

```

```
