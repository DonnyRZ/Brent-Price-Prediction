# Panduan Proses Proyek (Tujuan → Feature Engineering)

Dokumen ini adalah **panduan menyeluruh** yang menjelaskan proses proyek dari **Tujuan** hingga **Feature Engineering**. Setiap keputusan disertai **alasan (why)** agar alur berpikirnya jelas dan konsisten.

---

## 1. Tujuan Proyek (Kenapa Proyek Ini Ada)

**Tujuan utama:** memprediksi **harga penutupan (close) Brent Crude Oil harian**.

**Kenapa fokus ke Brent?**
- Brent adalah **benchmark utama pasar minyak global** dan sering dipakai sebagai acuan internasional.
- Nama dan deskripsi proyek di `README.md` menegaskan fokus pada Brent.
- Menjaga **konsistensi target** agar perbandingan antar model adil.

**Implikasi:**
- Semua pipeline berikutnya harus menjaga target konsisten pada `close_x` (Brent).

---

## 2. Sumber Data (Kenapa Ambil Data Ini)

**Sumber:** Kaggle – Historical Crude Oil Futures Prices (WTI & Brent).

**Kenapa pakai data WTI juga?**
- Brent dan WTI **sangat berkorelasi** (terlihat di heatmap korelasi EDA).
- WTI dapat menjadi **sinyal pendukung** untuk memprediksi Brent.
- Menambah konteks pasar global tanpa mengubah target utama.

**Keputusan:** WTI dipakai sebagai **fitur**, bukan target.

---

## 3. Struktur Data (Kenapa Dipisah Raw vs Processed)

**Aturan:**
- `data/raw/` = data mentah (immutable, tidak boleh diubah).
- `data/processed/` = data hasil preprocessing (boleh di-generate ulang).

**Kenapa dipisah?**
- Menjaga **integritas data** dan reproducibility.
- Memastikan pipeline bisa diulang kapan saja tanpa merusak data asli.

**Implikasi bisnis:** jika data rusak, cukup ulang preprocessing, tidak perlu unduh ulang dari sumber.

---

## 4. Preprocessing & Alignment (Kenapa Merge Dulu)

**Langkah:** menggunakan `src/data_preprocessing.py` untuk:
- membaca `brent_prices.csv` dan `wti_prices.csv`,
- menggabungkan berdasarkan `date` (inner join),
- mengurutkan tanggal.

**Kenapa merge dulu sebelum feature engineering?**
- Brent dan WTI punya **hari libur bursa yang berbeda**.
- Jika membuat lag/MA sebelum align, bisa terjadi **mismatch tanggal**.
- Inner join memastikan **setiap baris mewakili hari yang sama** untuk Brent dan WTI.

**Keputusan:** alignment adalah **prasyarat** sebelum feature engineering.

---

## 5. EDA (Kenapa Dilakukan & Kenapa Hanya Deskriptif)
**EDA yang dilakukan (sesuai `notebooks/EDA.ipynb`):**
1. **Load data processed** dari `data/processed/merged_oil_prices.csv`.
2. **Konversi kolom `date`** ke tipe datetime.
3. **Ringkasan awal data**:
   - `head()` untuk melihat 5 baris pertama.
   - `info()` untuk tipe data & missing value.
   - `describe()` untuk statistik dasar.
4. **Time series plot** perbandingan `close_x` (Brent) vs `close_y` (WTI) sepanjang waktu.
5. **Correlation heatmap** antar fitur harga (open/high/low/close) Brent & WTI.
6. **Distribution plot** (histogram + KDE) untuk membandingkan sebaran harga Brent dan WTI.


**Tujuan EDA di proyek ini:**
1. Memahami pola dasar harga Brent dan WTI.
2. Mengonfirmasi korelasi antar fitur.
3. Melihat distribusi dan range harga.

**Kenapa heatmap korelasi tidak diikuti aksi teknis?**
- EDA dipakai sebagai **visual insight**, bukan tahap optimasi.
- Fokus proyek ini adalah **membandingkan model**, bukan melakukan feature selection mendalam.
- Model baseline dijalankan “apa adanya” untuk melihat performa awal.

**Implikasi:**
- Multikolinearitas di MLR bisa membuat koefisien tidak stabil.
- Namun performa prediksi tetap baik karena sinyal temporal kuat (pola nilai masa lalu yang konsisten mempengaruhi nilai hari berikutnya, misalnya lag dan MA yang masih relevan).

---

## 6. Feature Engineering (Kenapa Fitur Ini Dibuat)

### 6.1 Kenapa Pakai Lag Features?
- Harga harian minyak memiliki **autokorelasi** tinggi.
- Lag 1, 3, 5, 7 menangkap pola **jangka pendek dan mingguan**.
- Membantu model menebak harga esok dari histori terdekat.

### 6.2 Kenapa Pakai Moving Average (MA)?
- MA menghaluskan noise harian dan menonjolkan tren.
- MA 5 dan 10 hari adalah jendela pendek yang umum untuk time series harian.

### 6.3 Kenapa Ada Feature Tambahan di Random Forest?
- Random Forest lebih toleran terhadap fitur redundant.
- Bisa menangkap **pola non-linear** dan interaksi antar fitur.
- Fitur tambahan seperti spread, high-low range, dan volume lag menambah informasi tanpa merusak performa RF.

### 6.4 Kenapa `dropna()` Setelah Membuat Lag/MA?
- Lag/MA menghasilkan NaN di baris awal/akhir.
- Model tidak dapat dilatih dengan NaN.
- `dropna()` adalah solusi **paling aman dan konsisten** untuk data time series.

---

### 6.5 Detail Feature Engineering per Model (Apa yang Dibuat)

**Ringkasan istilah:**
- **X** = fitur/input yang dipakai model untuk belajar.
- **y** = target/output yang ingin diprediksi (di proyek ini: harga penutupan Brent hari berikutnya / `close_x` t+1).


#### 6.5.1 Multiple Linear Regression (`notebooks/Multiple_Linear_Regression.ipynb`)

**Sumber data:** `data/processed/merged_oil_prices.csv`

**Langkah detail:**
1. **Copy data:** `data = df.copy()`
2. **Buat lag Brent (close_x):**
   - `brent_lag_1`, `brent_lag_3`, `brent_lag_5`, `brent_lag_7`
3. **Buat lag WTI (close_y):**
   - `wti_lag_1`, `wti_lag_3`, `wti_lag_5`, `wti_lag_7`
4. **Buat Moving Average Brent:**
   - `brent_ma_5`, `brent_ma_10` (rolling mean)
5. **Buat Moving Average WTI:**
   - `wti_ma_5`, `wti_ma_10` (rolling mean)
6. **Buat target (label):**
   - `target = close_x.shift(-1)` (harga penutupan Brent hari berikutnya)
7. **Hapus NaN:** `data_clean = data.dropna()`

**Fitur akhir yang dipakai (X):**
- Fitur dasar Brent: `close_x, open_x, high_x, low_x, average_x`
- Lag Brent: `brent_lag_1, brent_lag_3, brent_lag_5, brent_lag_7`
- MA Brent: `brent_ma_5, brent_ma_10`
- Fitur dasar WTI: `close_y, open_y, high_y, low_y, average_y`
- Lag WTI: `wti_lag_1, wti_lag_3, wti_lag_5, wti_lag_7`
- MA WTI: `wti_ma_5, wti_ma_10`

**Target (y):** `target`

---

#### 6.5.2 Random Forest (`notebooks/Random_Forest.ipynb`)

**Sumber data:** `data/processed/merged_oil_prices.csv`

**Langkah detail:**
1. **Copy data:** `df = df.copy()`
2. **Buat lag close Brent:**
   - `brent_close_lag_1, brent_close_lag_3, brent_close_lag_5, brent_close_lag_7`
3. **Buat lag close WTI:**
   - `wti_close_lag_1, wti_close_lag_3, wti_close_lag_5, wti_close_lag_7`
4. **Buat lag volume Brent:**
   - `brent_volume_lag_1, brent_volume_lag_3, brent_volume_lag_5, brent_volume_lag_7`
5. **Buat Moving Average close Brent:**
   - `brent_close_ma_5, brent_close_ma_10`
6. **Buat Moving Average close WTI:**
   - `wti_close_ma_5, wti_close_ma_10`
7. **Buat fitur tambahan (spread & range):**
   - `brent_high_low_diff = high_x - low_x`
   - `wti_high_low_diff = high_y - low_y`
   - `brent_open_close_diff = close_x - open_x`
   - `brent_wti_spread = close_x - close_y`
8. **Buat target:**
   - `target_brent_next_day = close_x.shift(-1)`
9. **Hapus NaN:** `df_clean = df.dropna()`

**Fitur akhir yang dipakai (X):**
- Lag Brent close: `brent_close_lag_1, brent_close_lag_3, brent_close_lag_5, brent_close_lag_7`
- Lag WTI close: `wti_close_lag_1, wti_close_lag_3, wti_close_lag_5, wti_close_lag_7`
- Lag Brent volume: `brent_volume_lag_1, brent_volume_lag_3, brent_volume_lag_5, brent_volume_lag_7`
- MA Brent close: `brent_close_ma_5, brent_close_ma_10`
- MA WTI close: `wti_close_ma_5, wti_close_ma_10`
- Fitur tambahan: `brent_high_low_diff, wti_high_low_diff, brent_open_close_diff, brent_wti_spread`
- Fitur dasar Brent: `open_x, high_x, low_x, close_x, volume_x, average_x`
- Fitur dasar WTI: `open_y, high_y, low_y, close_y, volume_y, average_y`

**Target (y):** `target_brent_next_day`

---

#### 6.5.3 Neural Network (`notebooks/Neural_Network.ipynb`)

**Sumber data:** `data/processed/merged_oil_prices.csv`

**Langkah detail:**
1. **Copy data:** `data = df.copy()`
2. **Buat lag Brent (close_x):**
   - `brent_lag_1, brent_lag_3, brent_lag_5, brent_lag_7`
3. **Buat lag WTI (close_y):**
   - `wti_lag_1, wti_lag_3, wti_lag_5, wti_lag_7`
4. **Buat Moving Average Brent:**
   - `brent_ma_5, brent_ma_10`
5. **Buat Moving Average WTI:**
   - `wti_ma_5, wti_ma_10`
6. **Buat target:**
   - `target = close_x.shift(-1)`
7. **Hapus NaN:** `data_clean = data.dropna()`

**Fitur akhir yang dipakai (X):**
- Sama persis dengan MLR (fitur dasar Brent/WTI + lag + MA)

**Target (y):** `target`

## 7. Ringkasan Decision Map (Tujuan → Feature Engineering)

1. **Target = `close_x` (Brent)** karena proyek fokus Brent.
2. **WTI digunakan** sebagai fitur karena korelasi tinggi dengan Brent.
3. **Raw data tidak diubah** untuk menjaga integritas.
4. **Merge & alignment dulu** agar timeline Brent-WTI sinkron.
5. **EDA dilakukan** untuk insight, bukan optimasi pipeline.
6. **Lag + MA** dipakai untuk menangkap pola waktu.
7. **RF punya fitur lebih kaya** karena lebih fleksibel terhadap redundansi.
8. **dropna()** wajib setelah lag/MA untuk menjaga data bersih.

---

## 8. Catatan Pengembangan (Opsional)

Jika scope diperluas, keputusan berikut bisa ditambahkan:
- Feature selection berbasis korelasi/VIF.
- Regularisasi (Ridge/Lasso) untuk MLR.
- Eksperimen mengurangi fitur redundant dan membandingkan metrik.

---

## 9. File Rujukan Utama

- `README.md` (tujuan proyek)
- `data/README.md` (aturan raw vs processed)
- `src/data_preprocessing.py` (preprocessing & alignment)
- `notebooks/EDA.ipynb` (EDA)
- `notebooks/Multiple_Linear_Regression.ipynb` (feature engineering MLR)
- `notebooks/Random_Forest.ipynb` (feature engineering RF)
- `notebooks/Neural_Network.ipynb` (feature engineering NN)