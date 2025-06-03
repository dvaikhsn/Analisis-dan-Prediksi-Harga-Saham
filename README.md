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

### 🗂️ Struktur Dataset

- **Jumlah baris:** 4705  
- **Jumlah kolom:** 7  
- Dataset ini terdiri dari data historis harga saham harian TLKM, dengan fitur-fitur sebagai berikut:

| Fitur        | Deskripsi |
|--------------|-----------|
| `Date`       | Tanggal perdagangan (masih bertipe *object*, perlu dikonversi ke *datetime*) |
| `Open`       | Harga pembukaan saham pada hari tersebut |
| `High`       | Harga tertinggi saham pada hari tersebut |
| `Low`        | Harga terendah saham pada hari tersebut |
| `Close`      | Harga penutupan saham pada hari tersebut |
| `Adj Close`  | Harga penutupan yang telah disesuaikan untuk dividen dan *stock split* |
| `Volume`     | Jumlah saham yang diperdagangkan pada hari tersebut |

---

### 🔎 Kondisi Data

**1. Missing Values:**
```
# Cek missing value
print("\nJumlah missing value per kolom:")
print(df.isnull().sum())
```

- Terdapat **1 baris** dengan nilai kosong (*missing*) pada **semua kolom numerik** (`Open`, `High`, `Low`, `Close`, `Adj Close`, dan `Volume`).
- Kolom `Date` **tidak memiliki nilai kosong**.
- Baris dengan nilai kosong ini akan ditangani pada tahap *data preprocessing*.

**2. Data Duplikat:**
```
#Cek data duplikat
print(df.duplicated().sum())
```

- Tidak ditemukan baris duplikat dalam dataset.

**3. Outlier:**

- Dari hasil eksplorasi awal, ditemukan beberapa outlier pada kolom `Volume` (terlihat sebagai lonjakan yang tidak lazim dalam volume perdagangan).
- Outlier akan dianalisis lebih lanjut pada tahap eksplorasi data (EDA) dan *preprocessing* untuk menentukan apakah perlu ditangani atau tidak.



## 📊 Exploratory Data Analysis (EDA)

### 1. Distribusi Harga Saham (`Open`, `High`, `Low`, `Close`)
```
# Plot distribusi semua fitur numerik
features = ['Open', 'High', 'Low', 'Close', 'Volume']

plt.figure(figsize=(15, 10))
for i, col in enumerate(features):
    plt.subplot(2, 3, i + 1)
    sns.histplot(df[col], kde=True, bins=30, color='skyblue')
    plt.title(f'Distribusi {col}')
plt.tight_layout()
plt.show()

```
![Distribusi Harga](https://github.com/user-attachments/assets/3f366175-8c40-4ab2-8eb7-979c33bef60d)

- Distribusi dari keempat harga saham (`Open`, `High`, `Low`, `Close`) menunjukkan pola **multimodal**. Hal ini mencerminkan bahwa selama periode pengamatan, harga saham TLKM mengalami beberapa **fase tren berbeda**, kemungkinan akibat faktor ekonomi makro, perubahan regulasi, atau performa perusahaan.
- Teridentifikasi dua rentang harga yang paling sering muncul:
  - Sekitar **Rp1.500–2.000**
  - Sekitar **Rp3.500–4.000**
- Kedua rentang ini menunjukkan dua fase utama pergerakan harga TLKM dalam jangka panjang: fase harga rendah dan fase harga tinggi.

---

### 2. Korelasi Antar Fitur Harga
```
# Salin dataframe tanpa kolom Date
df_numerik = df.select_dtypes(include='number')

# Visualisasi korelasi
plt.figure(figsize=(8, 6))
sns.heatmap(df_numerik.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap Korelasi Fitur Numerik')
plt.show()
```
![image](https://github.com/user-attachments/assets/3fcb8979-f044-43d7-8b2d-96703d8aa388)

- Terdapat **korelasi yang sangat kuat (mendekati 1.00)** antar harga saham (`Open`, `High`, `Low`, `Close`, `Adj Close`), menandakan bahwa pergerakan harga harian saling mengikuti secara konsisten.
- Korelasi `Adj Close` terhadap `Close` sebesar **0.98**, cukup tinggi meski sedikit lebih rendah — ini bisa terjadi karena penyesuaian tertentu pada data historis.
- **Volume tidak berkorelasi signifikan** dengan fitur harga (korelasi mendekati 0). Artinya, **volume perdagangan harian tidak banyak mempengaruhi harga harian secara langsung**, atau ada faktor lain yang lebih dominan.
- Kesimpulan: karena fitur harga sangat berkorelasi satu sama lain, **pemilihan satu fitur harga (misalnya `Close`) sudah cukup mewakili pergerakan harga untuk modeling**.


---

### 3. Pairplot Fitur Harga Saham
```
# Pairplot fitur harga
sns.pairplot(df[['Open', 'High', 'Low', 'Close']])
plt.suptitle('Pairplot Fitur Harga Saham', y=1.02)
plt.show()
```
![image](https://github.com/user-attachments/assets/febbadeb-780a-42d6-ad08-a5eedc8b6298)

- Terlihat hubungan **linear sangat kuat** antara fitur-fitur harga saham (`Open`, `High`, `Low`, `Close`) — konsisten dengan hasil korelasi sebelumnya.
- Penyebaran data pada scatter plot membentuk garis diagonal rapat, menandakan bahwa nilai-nilai harga bergerak sangat beriringan antar fitur.
- Distribusi pada histogram diagonal (untuk masing-masing fitur) kembali menunjukkan **pola multimodal**, menguatkan asumsi bahwa terdapat beberapa periode atau fase harga yang berbeda dalam data historis TLKM.
- **Implikasi untuk pemodelan:** karena fitur saling berkorelasi tinggi, **menggunakan salah satu dari fitur tersebut, seperti `Close`, sudah cukup merepresentasikan dinamika harga untuk prediksi ke depan**.


### 4. Scatterplot volume vs close
```
# Scatterplot volume vs close
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Volume', y='Close', data=df, alpha=0.5)
plt.title('Volume vs Harga Close')
plt.xlabel('Volume')
plt.ylabel('Harga Close')
plt.show()
```
![image](https://github.com/user-attachments/assets/5fa14ecb-d6b4-4fc0-9ba5-619c070861fe)

- Scatter plot menunjukkan tidak adanya pola yang jelas antara volume dan harga close.

- Banyak titik terakumulasi di volume rendah, dengan sebaran harga yang cukup merata, dan beberapa outlier volume tinggi tidak menunjukkan hubungan signifikan.


### 5. Deteksi outlier
```
plt.figure(figsize=(15, 8))
for i, col in enumerate(features):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(x=df[col], color='salmon')
    plt.title(f'Boxplot {col}')
plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/55b91ed4-5171-4306-a49e-eea39fc10583)

- Harga (Open, High, Low, Close) memiliki distribusi yang cukup simetris dan tidak menunjukkan banyak outlier.

- Volume menampilkan banyak outlier (titik-titik jauh dari box utama), mengindikasikan bahwa terdapat beberapa hari dengan volume perdagangan yang tinggi dibanding hari-hari lainnya.


---

### 🧠 Ringkasan EDA

- Harga saham TLKM menunjukkan pola **multifase** dengan distribusi multimodal.
- Terdapat **korelasi sangat tinggi antar fitur harga**, memungkinkan penggunaan satu fitur utama untuk pemodelan.
- `Volume` mengandung **outlier signifikan** dan distribusinya sangat skewed, tetapi tidak berkorelasi kuat dengan harga.



## Data Preparation

Langkah-langkah berikut dilakukan untuk menyiapkan data sebelum proses pemodelan. Semua teknik disusun sesuai urutan eksekusi di notebook:

---

### 1. Menghapus Kolom Non-Numerik (`Date`)
- Kolom `Date` dihapus dari dataset karena bersifat non-numerik dan tidak digunakan sebagai fitur dalam model prediksi.
```python
df_clean = df.drop(columns=['Date'])
```

### 2. Menghapus Baris dengan Target (Close) yang Kosong
- Baris yang memiliki nilai kosong pada kolom Close dihapus, karena target yang hilang tidak bisa digunakan dalam supervised learning.
```python
df_clean = df_clean.dropna(subset=['Close'])
```

### 3. Memisahkan Fitur dan Target
- Dataset dipisahkan menjadi:
- X: fitur input (tanpa kolom Close)
- y: target variabel (Close)
```python
X = df_clean.drop(columns=['Close'])
y = df_clean['Close']
```

### 4. Imputasi Nilai Kosong
- Menggunakan SimpleImputer dengan strategi mean untuk mengisi nilai kosong pada fitur input X.
```python
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
```
### 5. Standardisasi Fitur
- Menggunakan StandardScaler untuk menstandardisasi fitur agar memiliki mean = 0 dan standar deviasi = 1. Ini penting untuk PCA dan banyak algoritma machine learning.
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
```

### 6. Reduksi Dimensi dengan PCA
- Menggunakan Principal Component Analysis (PCA) untuk menurunkan dimensi data sambil mempertahankan 90% variansi total.
- Tujuannya adalah menyederhanakan data, mengurangi noise, dan mempercepat pelatihan model.
```python
pca = PCA(n_components=0.9, random_state=42)
X_pca = pca.fit_transform(X_scaled)
```
- Jumlah komponen utama yang dihasilkan:
```python
print(f"Jumlah komponen PCA yang dipilih: {pca.n_components_}")
Jumlah komponen PCA yang dipilih: 2
```

### 7. Pembagian Data
- **Langkah**: Membagi data hasil PCA ke dalam data latih dan data uji dengan rasio 80:20.
- **Kode**:
  ```python
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
  ```
- **Alasan**: Pembagian ini dilakukan agar model dapat dilatih pada sebagian data dan diuji pada data yang belum pernah dilihat, guna mengevaluasi performa secara objektif.

### 8. Hasil PCA
- **Output**: Jumlah komponen utama yang dipilih: **`{pca.n_components_}`**
  `Jumlah komponen PCA yang dipilih: 2`
- PCA telah mengidentifikasi sejumlah fitur baru (lebih sedikit dari jumlah awal) yang dapat menjelaskan 90% variansi dalam data awal yang telah ditransformasi.

## Modeling
Dalam tahap ini, tiga algoritma machine learning digunakan untuk membangun model prediktif terhadap harga penutupan saham (`Close`). Setiap algoritma memiliki kelebihan dan mekanisme kerja yang berbeda. Berikut penjelasannya:

### 1. Linear Regression

Linear Regression adalah algoritma statistik yang digunakan untuk memodelkan hubungan linier antara satu atau lebih fitur input (`X`) dengan target output (`y`). Model ini berusaha mencari garis lurus terbaik yang meminimalkan selisih kuadrat antara prediksi dan nilai aktual (Least Squares Error).

- Parameter yang digunakan: default dari `LinearRegression()`

```python
lr_model = LinearRegression()
```

### 2. Random Forest Regressor
Random Forest adalah algoritma ensemble yang membangun banyak pohon keputusan (decision tree) secara acak, lalu menggabungkan hasil prediksi dari seluruh pohon untuk menghasilkan prediksi akhir (biasanya dengan rata-rata dalam regresi). Ini membuat model lebih stabil dan mengurangi overfitting.

Parameter yang digunakan:
- n_estimators=100: membuatn 100 pohon keputusan.
- random_state=42: memastikan hasil replikasi yang konsisten.
```python
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
```

### 3. XGBoost Regressor
XGBoost (Extreme Gradient Boosting) adalah algoritma boosting berbasis gradient descent. Model dibangun secara bertahap dengan setiap pohon baru berfokus untuk memperbaiki kesalahan prediksi dari pohon sebelumnya. XGBoost dikenal karena performa tinggi dan efisiensinya dalam menangani data tabular.

Parameter yang digunakan:
- objective='reg:squarederror': fungsi loss regresi berbasis error kuadrat.
- random_state=42: memastikan hasil yang konsisten.
```python
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
```


## Evaluasi Model
Setelah pelatihan, performa masing-masing model dievaluasi menggunakan tiga metrik:
- RMSE (Root Mean Squared Error): akar dari rata-rata kuadrat kesalahan.
- MAE (Mean Absolute Error): rata-rata nilai absolut dari kesalahan prediksi.
- R² Score: seberapa baik model menjelaskan variasi data.
```python
def evaluate(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} Performance:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}\n")
    return rmse, mae, r2
```
### Hasil Evaluasi Model

| Model              | RMSE     | MAE      | R²     |
|--------------------|----------|----------|--------|
| Linear Regression  | 54.8690  | 42.3002  | 0.9976 |
| Random Forest      | 55.2107  | 38.2549  | 0.9976 |
| XGBoost            | 56.6222  | 39.6326  | 0.9974 |

📌 Model terbaik berdasarkan RMSE: **Linear Regression.**

📌 Model dengan MAE terkecil: **Random Forest.**


## 📈 Perbandingan Performa Model

Tiga metrik utama yang dibandingkan antar model:

- **RMSE:** Mengukur rata-rata error dalam satuan harga saham.
- **MAE:** Mengukur rata-rata deviasi absolut dari prediksi.
- **R² Score:** Mengindikasikan seberapa baik model menjelaskan variasi pada target.

### 🖼️ Visualisasi Perbandingan Metrik

> Diagram batang berikut menunjukkan perbandingan RMSE, MAE, dan R² antar model:

![Perbandingan Metrik Model](https://github.com/user-attachments/assets/75f3f6a2-29f6-445e-996b-8a346d6427f4)

- Ketiga model memiliki performa yang sangat mirip.
- **Random Forest unggul sedikit pada MAE.**

---

## 🔍 Visualisasi Prediksi vs Aktual

Grafik scatter di bawah memperlihatkan seberapa dekat prediksi model terhadap nilai aktual. Garis merah putus-putus menandai prediksi sempurna (`prediksi = aktual`).

### 📊 Linear Regression
![Linear Regression](https://github.com/user-attachments/assets/903bfad2-225f-4e24-8323-f69a30936626)

### 🌲 Random Forest
![Random Forest](https://github.com/user-attachments/assets/e2c8c8bf-40fb-4d1d-bcc5-1ae90c2cae33)

### ⚡ XGBoost
![XGBoost](https://github.com/user-attachments/assets/6d657ebc-8f15-4887-a42e-3b09cc5c1052)

> Sebaran titik yang rapat terhadap garis menunjukkan prediksi yang akurat.  
> **Linear Regression** menunjukkan distribusi paling rapat.

---

## 📉 Distribusi Error (Residual)

Visualisasi residual memperlihatkan selisih antara nilai prediksi dan aktual. Distribusi yang simetris dan terpusat di nol menandakan prediksi tidak bias.

### 📊 Linear Regression
![Residual LR](https://github.com/user-attachments/assets/8ee40ffd-2730-42db-af50-624f3c848c25)

### 🌲 Random Forest
![Residual RF](https://github.com/user-attachments/assets/37a61055-cafa-4e9a-bd6c-1ab73f57ba3e)

### ⚡ XGBoost
![Residual XGB](https://github.com/user-attachments/assets/3312d501-7995-4100-95c5-eacb877bbdb2)

> **Linear Regression** dan **Random Forest** menunjukkan distribusi error yang lebih simetris dan sempit dibanding XGBoost.

---


# **Evaluasi Terhadap Business Understanding**

---

## ✅ Problem Statements

1. **Bagaimana memprediksi harga penutupan saham TLKM berdasarkan data historis?**

   Model berhasil dibangun menggunakan data historis harga saham TLKM. Dengan pendekatan supervised learning berbasis regresi, model mampu memprediksi harga penutupan (`Close`) menggunakan fitur-fitur numerik (`Open`, `High`, `Low`, `Adj Close`, `Volume`) yang telah diproses dan direduksi menggunakan PCA.

2. **Model machine learning mana yang paling akurat dalam melakukan prediksi harga saham?**

   Tiga model dievaluasi: **Linear Regression**, **Random Forest**, dan **XGBoost**. Hasil evaluasi menunjukkan bahwa:

   - Linear Regression memiliki **RMSE** terkecil (54.8690).
   - Random Forest memiliki **MAE** terkecil (38.2549).
   - Ketiga model memiliki nilai **R² > 0.997**, yang menandakan akurasi sangat tinggi.

   Ini memberi gambaran model mana yang paling layak digunakan sesuai kebutuhan (minim error absolut atau minim error kuadrat).

3. **Bagaimana hasil prediksi tersebut dapat digunakan dalam konteks pengambilan keputusan investasi?**

   Model prediksi dapat digunakan untuk memproyeksikan harga penutupan harian ke depan, yang menjadi acuan investor untuk melakukan aksi jual atau beli. Prediksi yang stabil dan akurat membantu meminimalkan risiko keputusan yang salah.

---

## ✅ Goals

1. **Membangun model machine learning untuk memprediksi harga penutupan (Close) saham TLKM.**

   Seluruh pipeline machine learning telah dijalankan dengan baik, dari preprocessing, PCA, hingga pelatihan dan evaluasi tiga model regresi.

2. **Membandingkan performa tiga model: Linear Regression, Random Forest, dan XGBoost.**

   Ketiga model telah dilatih dan diuji. Performa masing-masing model disajikan dalam bentuk tabel metrik (RMSE, MAE, R²) serta visualisasi prediksi dan residual.

3. **Memberikan insight bisnis dari hasil model sebagai dasar pertimbangan investasi.**

   Insight bisnis disusun berdasarkan performa model, distribusi error, serta interpretasi visualisasi prediksi vs aktual. Hasilnya digunakan untuk menyarankan model terbaik dan mempertimbangkan penggunaannya dalam strategi investasi.

---

## ✅ Solution Statements

1. **Menggunakan tiga algoritma regresi: Linear Regression, Random Forest Regressor, dan XGBoost Regressor.**

   Semua algoritma dijalankan dengan parameter yang sesuai dan digunakan dalam evaluasi performa.

2. **Menggunakan PCA (Principal Component Analysis) untuk reduksi dimensi.**

   PCA digunakan untuk menyederhanakan fitur input menjadi 2 komponen utama yang menjelaskan 90% variansi, yang terbukti cukup kuat untuk prediksi.

3. **Evaluasi model dilakukan menggunakan metrik: RMSE, MAE, dan R².**

   Ketiga metrik ini digunakan dalam evaluasi untuk membandingkan performa prediktif model, dan hasilnya dilaporkan secara menyeluruh.

---

## 💡 Insight Bisnis

- **Linear Regression** memberikan performa terbaik dalam hal RMSE dan distribusi residual yang rapat, menandakan stabilitas tinggi.
- Model ini dapat digunakan sebagai dasar pengambilan keputusan terkait **waktu beli/jual saham TLKM**.
- **Random Forest** juga layak dipertimbangkan karena memiliki kesalahan absolut (MAE) paling rendah.
- Namun, **model hanya memanfaatkan data historis**. Faktor eksternal seperti **kondisi ekonomi, politik, dan regulasi** tidak tercakup dalam model.
- **Rekomendasi:** Lakukan pembaruan dan retraining model secara berkala untuk menjaga relevansi dengan kondisi pasar terkini.

