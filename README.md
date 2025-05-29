# Laporan Proyek Machine Learning - Dava Ikhsan R

## Domain Proyek
Investasi di pasar modal, khususnya saham, membutuhkan pertimbangan matang yang berbasis data. Dalam konteks ini, proyek ini bertujuan untuk membangun model prediktif terhadap harga penutupan saham PT Telkom Indonesia. Dengan memanfaatkan data historis harga saham dan teknik machine learning, investor diharapkan dapat memperoleh wawasan untuk pengambilan keputusan jual atau beli.

## Business Understanding

### Problem Statements
- Bagaimana memprediksi harga penutupan saham TLKM berdasarkan data historis?
- Model machine learning mana yang paling akurat dalam melakukan prediksi harga saham?
- Bagaimana hasil prediksi tersebut dapat digunakan dalam konteks pengambilan keputusan investasi?

### Goals
- Membangun model machine learning untuk memprediksi harga penutupan (Close) saham TLKM.
- Membandingkan performa tiga model: Linear Regression, Random Forest, dan XGBoost.
- Memberikan insight bisnis dari hasil model sebagai dasar pertimbangan investasi.

### Solution Statements
- Menggunakan tiga algoritma regresi: Linear Regression, Random Forest Regressor, dan XGBoost Regressor.
- Menggunakan PCA (Principal Component Analysis) untuk reduksi dimensi dan peningkatan efisiensi modeling.
- Evaluasi model dilakukan menggunakan metrik: RMSE, MAE, dan R².

## Data Understanding

Dataset yang digunakan adalah data historis saham PT Telkom Indonesia (TLKM) dari Kaggle:

📊 **Sumber dataset**: [Telecommunications Company Stock Price – Kaggle](https://www.kaggle.com/datasets/brmil07/telecommunications-company-stock-price)

Dataset berisi:

- `Date`: Tanggal transaksi
- `Open`: Harga pembukaan
- `High`: Harga tertinggi
- `Low`: Harga terendah
- `Close`: Harga penutupan
- `Adj Close`: Harga penutupan yang disesuaikan
- `Volume`: Jumlah saham yang diperdagangkan

EDA (Exploratory Data Analysis) menunjukkan:
- Harga saham menunjukkan distribusi multimodal, mencerminkan beberapa fase tren harga.
  ![image](https://github.com/user-attachments/assets/3f366175-8c40-4ab2-8eb7-979c33bef60d)

1.  **Distribusi harga (`Open`, `High`, `Low`, `Close`) menunjukkan pola multimodal**, menandakan adanya beberapa fase harga dominan sepanjang waktu. Ini bisa mencerminkan perubahan tren atau siklus pasar selama periode data.
2. Harga-harga cenderung terkonsentrasi pada dua rentang besar: sekitar **1500–2000** dan **3500–4000**, mengindikasikan dua fase utama harga saham TLKM.
3. **Distribusi `Volume` sangat skewed ke kanan** (right-skewed), menunjukkan sebagian besar volume perdagangan berada di angka rendah hingga menengah, sementara terdapat beberapa hari dengan volume ekstrem tinggi.


- Korelasi tinggi antar fitur harga (`Open`, `High`, `Low`, `Close`, `Adj Close`).
![image](https://github.com/user-attachments/assets/eef32346-ff3c-406e-a65a-c5663102601e)

1. Terlihat hubungan **linear sangat kuat** antara fitur-fitur harga saham (`Open`, `High`, `Low`, `Close`) — konsisten dengan hasil korelasi sebelumnya.
2. Penyebaran data pada scatter plot membentuk garis diagonal rapat, menandakan bahwa nilai-nilai harga bergerak sangat beriringan antar fitur.
3. Distribusi pada histogram diagonal (untuk masing-masing fitur) kembali menunjukkan **pola multimodal**, menguatkan asumsi bahwa terdapat beberapa periode atau fase harga yang berbeda dalam data historis TLKM.
4. **Implikasi untuk pemodelan:** karena fitur saling berkorelasi tinggi, **menggunakan salah satu dari fitur tersebut, seperti `Close`, sudah cukup merepresentasikan dinamika harga untuk prediksi ke depan**.



- Volume perdagangan tidak berkorelasi kuat dengan harga, namun mengandung outlier yang mencerminkan aktivitas tidak biasa.
![image](https://github.com/user-attachments/assets/282055b3-7ab2-49c2-8c50-e9a8224d3d36)

## Data Preparation

- Mengubah tipe data `Date` menjadi datetime.
- Menghapus baris dengan nilai kosong (missing values).
- Menyortir data berdasarkan tanggal.
- Menggunakan PCA untuk mereduksi dimensi data agar lebih efisien dalam pemodelan.
- Split data menjadi training (80%) dan testing (20%).

## Modeling

Tiga algoritma machine learning digunakan:

1. **Linear Regression**: model sederhana berbasis hubungan linier antar variabel.
2. **Random Forest Regressor**: model ensemble berbasis decision trees.
3. **XGBoost Regressor**: model boosting yang sangat populer untuk prediksi tabular.

### Hasil Evaluasi:

| Model              | RMSE     | MAE      | R²     |
|--------------------|----------|----------|--------|
| Linear Regression  | 54.8690  | 42.3002  | 0.9976 |
| Random Forest      | 55.2107  | 38.2549  | 0.9976 |
| XGBoost            | 56.6222  | 39.6326  | 0.9974 |

Model terbaik berdasarkan RMSE adalah **Linear Regression**.

## Evaluation

![image](https://github.com/user-attachments/assets/75f3f6a2-29f6-445e-996b-8a346d6427f4)

## 📈 Perbandingan Performa Model

Setelah semua model dilatih dan dievaluasi, terdapat **tabel perbandingan metrik performa** untuk melihat model mana yang paling optimal.

Tiga metrik utama yang dibandingkan:
- **RMSE (Root Mean Squared Error)** – Semakin kecil, semakin baik.
- **MAE (Mean Absolute Error)** – Selisih absolut rata-rata, semakin kecil lebih akurat.
- **R² Score (Koefisien Determinasi)** – Semakin mendekati 1, semakin baik.


### Visualisasi Performa Model

Diaggram batang di atas menampilkan perbandingan performa dari ketiga model:

- **RMSE:** Mengukur rata-rata error dalam satuan harga saham.
- **MAE:** Menunjukkan rata-rata deviasi absolut prediksi.
- **R²:** Mengindikasikan seberapa baik model menjelaskan variasi data target.

> Dari visualisasi ini terlihat bahwa **ketiga model memiliki performa yang sangat mirip**, dengan Random Forest sedikit unggul di MAE (kesalahan absolut terkecil).

---


## Prediksi vs aktual menunjukkan sebaran titik yang rapat terhadap garis ideal pada Linear Regression.
![image](https://github.com/user-attachments/assets/b020f890-9d39-4c64-ae06-d2d2ed41aa01)

## 🔍 Visualisasi Prediksi vs Aktual

Untuk mengevaluasi kualitas prediksi tiap model, disini menampilkan grafik sebaran antara **nilai aktual** dan **nilai prediksi**. Garis merah putus-putus merepresentasikan garis ideal (prediksi = aktual).

Jika titik-titik prediksi mendekati garis ini, maka model berhasil melakukan prediksi yang baik.

> Hasil Linear Regression menunjukkan sebaran titik yang sangat rapat dengan garis ideal, menandakan akurasi tinggi dalam memodelkan hubungan antara fitur-fitur dan harga saham.

---

## 🧮 Distribusi Error (Residual)

Distribusi residual menggambarkan selisih antara nilai aktual dan prediksi. Distribusi yang **berbentuk simetris dan mendekati nol** menandakan bahwa model tidak bias dan performanya konsisten.

Semua model menunjukkan distribusi residual yang cukup simetris, namun **Linear Regression dan Random Forest** cenderung memiliki distribusi error yang lebih sempit.

---

## Insight Bisnis

- Model Linear Regression memberikan performa terbaik dalam hal RMSE dan kestabilan.
- Prediksi dari model ini dapat digunakan sebagai dasar untuk memutuskan waktu beli atau jual saham TLKM.
- Namun, investor tetap perlu memperhatikan faktor eksternal (politik, ekonomi, dll) yang tidak tercakup oleh model.
- Disarankan untuk memperbarui model secara berkala agar tetap relevan dengan kondisi pasar terkini.
