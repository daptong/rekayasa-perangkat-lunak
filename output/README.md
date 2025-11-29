# UML Diagram Assessment - Iteration Analysis Report
 
> **Total Iterations:** 5  
> **Diagrams per Iteration:** 50  
> **Total Assessments:** 250

---

## Ringkasan

Laporan ini merangkum hasil evaluasi dari **5 iterasi independen** sistem penilaian diagram UML, yang memanfaatkan **mBERT embeddings** and **Graph Edit Distance (GED)** untuk menilai diagram UML buatan mahasiswa (berbahasa Indonesia) terhadap diagram acuan (berbahasa Inggris).

### Temuan Utama

| Metrik | Target | Capaian (Agregat) | Status |
|--------|--------|---------------------|--------|
| Korelasi Pearson | ≥ 0.76 | **0.8942** | Lulus |
| Signifikansi Statistik | p < 0.05 | p ≈ 1.55×10⁻⁸⁸ | Sangat signifikan |
| Kesesuaian ±10 | > 90% | **96.0%** | Sangat baik |

**Kesimpulan:** Sistem mampu mempertahankan korelasi tinggi dengan penilaian manusia pada setiap iterasi, jauh di atas target minimal 0.76

---

## Hasil Tiap Iterasi

### Ringkasan (Iterasi 01-05)

| Iterasi | Pearson (r) | p-value | MAE | RMSE | Akurasi ±5 | Akurasi ±10 | Target |
|-----------|-----------|---------|-----|------|--------------|---------------|------------|
| Iter 01 | 0.9095 | 6.31×10⁻²⁰ | 3.11 | 4.13 | 82.0% | 94.0% | ✓ |
| Iter 02 | 0.9083 | 8.60×10⁻²⁰ | 2.50 | 3.35 | 88.0% | 98.0% |  ✓ |
| Iter 03 | 0.8729 | 1.43×10⁻¹⁶ | 3.20 | 4.07 | 88.0% | 96.0% | ✓ |
| Iter 04 | 0.9037 | 2.65×10⁻¹⁹ | 2.81 | 3.48 | 90.0% | 98.0% | ✓ |
| Iter 05 | 0.8946 | 2.08×10⁻¹⁸ | 3.31 | 4.16 | 90.0% | 94.0% | ✓ |
| **Agregat** | **0.8942** | **1.55×10⁻⁸⁸** | **2.99** | **3.85** | **87.6%** | **96.0%** | **✓** |

### Temuan

1. **Konsistensi:** Nilai korelasi Pearson berada di rentang 0.8729 hingga 0.9095 pada setiap iterasi — semuanya melampaui target yang ditetapkan
2. **Iterasi Terbasik:** Iterasi 01 mencapai korelasi tertinggi (0.9095)
3. **Iterasi Terburuk:** Iterasi 03 memiliki korelasi terendah (0.8729), namun tetap jauh di atas ambang minimum 0.76
4. **Metrik Error:** MAE secara konsisten berada di bawah 3.5, dan RMSE berada di bahwa 4.2
5. **Kesesuaian Nilai:** Sebesar 94-98% skor sistem berada dalam selsiih ±10 poin dari skor penilai manusia

---

## Analisis Iterasi Secara Detail

### Iterasi 1

| Metrik | Nilai |
|--------|-------|
| Pearson (r) | 0.9095 |
| Spearman (r) | 0.8918 |
| MAE | 3.11 |
| RMSE | 4.13 |
| Mean Bias | +1.80 |
| Akurasi ±5 | 82.0% |
| Akurasi ±10 | 94.0% |

**Catatan:** Korelasi paling kuat. Sistem cenderung memberi nilai lebih tinggi (~1.8 poin)

---

### Iterasi 2

| Metrik | Nilai |
|--------|-------|
| Pearson (r) | 0.9083 |
| Spearman (r) | 0.8781 |
| MAE | 2.50 |
| RMSE | 3.35 |
| Mean Bias | +1.19 |
| Akurasi ±5 | 88.0% |
| Akurasi ±10 | 98.0% |

**Catatan:** Memiliki MAE terendah, juga memberikan kesesuaian ±10 poin tertinggi (98%)

---

### Iterasi 3

| Metrik | Nilai |
|--------|-------|
| Pearson (r) | 0.8729 |
| Spearman (r) | 0.8676 |
| MAE | 3.20 |
| RMSE | 4.07 |
| Mean Bias | +1.14 |
| Akurasi ±5 | 88.0% |
| Akurasi ±10 | 96.0% |

**Catatan:** Korelasi terendah di antara iterasi, tetapi tetap kuat. Bias tetap positif

---

### Iterasi 4

| Metrik | Nilai |
|--------|-------|
| Pearson (r) | 0.9037 |
| Spearman (r) | 0.8767 |
| MAE | 2.81 |
| RMSE | 3.48 |
| Mean Bias | +0.91 |
| Akurasi ±5 | 90.0% |
| Akurasi ±10 | 98.0% |

**Catatan:** Memiliki bias terkecil (+0.91), menyamai akurasi tertinggi dalam selisih ±10 poin

---

### Iterasi 5

| Metrik | Nilai |
|--------|-------|
| Pearson (r) | 0.8946 |
| Spearman (r) | 0.8648 |
| MAE | 3.31 |
| RMSE | 4.16 |
| Mean Bias | +2.32 |
| Akurasi ±5 | 90.0% |
| Akurasi ±10 | 94.0% |

**Catatan:** Bias paling besar (+2.323), yang menunjukkan kecenderungan sistem memberikan skor lebih tinggi dibandingkan penilai manusia

---

## Analisis Agregat (250 Penilaian)

### Metrik Keseluruhan

| Metrik | Nilai |
|--------|-------|
| **Pearson (r)** | **0.8942** |
| **p-value** | 1.55×10⁻⁸⁸ |
| **Spearman (r)** | 0.8816 |
| **MAE** | 2.99 |
| **RMSE** | 3.85 |
| **Mean Bias** | +1.47 |
| **Akurasi ±5** | 87.6% |
| **Akurasi ±10** | 96.0% |
| **Sampel** | 250 |

### Kinerja Berdasarkan Jenis Variasi Diagram

| Jenis Variasi | n | Pearson (r) | MAE | Mean Bias | Skor Sistem | Skor Manusia |
|----------------|---|-----------|-----|-----------|--------------|-------------|
| Perfect Translation | 25 | N/A* | 2.30 | +1.36 | 89.86 | 88.49 |
| Minor Semantic | 50 | 0.792 | 2.76 | +0.20 | 88.63 | 88.43 |
| Spelling Errors | 25 | 0.423 | 2.19 | — | — | — |
| Missing Classes | 25 | 0.927 | 2.03 | +0.30 | 80.18 | 79.88 |
| Extra Classes | 25 | 0.429 | 2.14 | +1.26 | 84.51 | 83.24 |
| Missing Attributes | 25 | 0.948 | 2.18 | +0.04 | 78.41 | 78.37 |
| Missing Methods | 25 | 0.916 | 2.68 | +0.13 | 76.92 | 76.79 |
| Wrong Relationships | 25 | — | — | — | — | — |
| Combination Errors | 25 | 0.412 | 6.74 | +6.74 | 76.51 | 69.77 |

*\* Jenis Perfect translation tidak terdapat variasi pada skor sistem (diagramnya identik), sehingga nilai Pearson (r) tidak dapat dihitung*

### Temuan Berdasarkan Jenis Variasi

1. **Performa Terbaik:**
   - **Missing Attributes** (r=0.948): Sistem sangat unggul dalam mendeteksi perbedaan atribut
   - **Missing Classes** (r=0.927): Deteksi elemen struktur (kelas) yang hilang sangat kuat
   - **Missing Methods** (r=0.916): Penilaian pada level method konsisten dan dapat diandalkan

2. **Performa Menengah:**
   - **Minor Semantic** (r=0.792): Deteksi kesamaan semantik berjalan baik karena mBERT
   - **Extra Classes** (r=0.429): Sistem kesulitan memberikan penalti yang tepat untuk kelas tambahan
   - **Spelling Errors** (r=0.423): mBERT membantu namun korelasinya tetap lebih rendah

3. **Kasus yang Menantang:**
   - **Combination Errors** (r=0.412, MAE=6.74): Berbagai jenis kesalahan yang muncul secara bersamaan, memperburuk kompleksitas penilaian
   - Sistem cenderung **memberikan skor terlalu tinggi** pada kasus ini, rata-rata sekitar +6.7 poin dari skor sebenarnya

---

## Analisis Statistik

### Stabilitas Korelasi

| Statistik | Pearson (r) (5 iterasi) |
|-----------|--------------------------------|
| Mean | 0.8978 |
| Std Dev | 0.0138 |
| Min | 0.8729 |
| Max | 0.9095 |
| Range | 0.0366 |

**Interpretasi:** Korelasi sangat stabil dengan standar deviasi rendah (1.4%). Ini menunjukkan kinerja sistem konsistem pada dataset yang berbeda.

### Pengujian Hipotesis

- **Hipotesis Nol (H₀):** Tidak ada korelasi antara skor sistem dan skor penilai manusia
- **Hipotesis Alternatif (H₁):** Ada korelasi positif yang signifikan
- **Hasil:** Nilai p-values < 0.001 (agregat p ≈ 10⁻⁸⁸).
- **Kesimpulan:** **H₀ ditolak** Artinya korelasi sangat signifikan pada level kepercayaan apapun (α = 0.05, 0.01, maupun 0.001)

### Effect Size (Ukuran Efek)

- Pearson r = 0.894 termasuk kategori **very strong**
- Berdasarkan panduan Cohen, r > 0.5 sudah dianggap efek besar — hasil sistem berada jauh di atas batas tersebut

---

## Ringkasan Visualisasi

Visualisasi berikut tersedia pada folder `output/iter_XX/visualizations/` dan folder `output/aggregate/visualizations/`:

1. **scatter_plot.png** — Plot hubungan skor sistem vs skor manusia dengan garis regresi
2. **scores_by_type.png** — Bosplot distribusi skor berdasarkan jenis variasi diagram
3. **error_distribution.png** — Histogram distribusi error penilaian
4. **confusion_matrix.png** — Confusion matrix dengan kategori (Poor/Fair/Good/Excellent)
5. **component_scores.png** — Analisis komponen penilaian (semantik, struktur, dan relasi)
6. **correlation_by_type.png** — Grafik korelasi berdasarkan tipe kesalahan diagram

---

## Kesimpulan

### Strengths

1. **Korelasi Tinggi:** Sistem mencapai korelasi sekitar 0.89-0.91 secara konsisten, jauh di atas target minimum 0.76
2. **Valid Secara Statistik:** Nilai p sangat kecil membuktikan bahwa korelasi bukan hasil kebetulab
3. **Error Rendah:** MAE selalu berada di bawah 3.5
4. **Deteksi Struktur:** Sistem sangatn baik dalam mengenali kelas, atribut, dan method yang hilang
5. **Kemampuan Multibahasa:** mBERT efektif menjembatani perbedaan Indonesia ↔ Inggris

### Keterbatasan

1. **Kombinasi Kesalahan:** Sistem cenderung memberikan skor terlalu tinggi ketika diagram memiliki banyak jenis kesalahan
2. **Variasi Nol:** Pada diagram yang identik (Perfect Translation), korelasi tidak dapat dihitung karena tidak ada variasi skor
3. **Bias Positif:** Sistem rata-rata memberi nilai lebih tinggi daripada manusia (±1.5 poin)

### Rekomendasi

1. **Penyesuaian Penalti:** Tingkatkan penalti untuk diagram dengan kombinasi kesalahan
2. **Kalibrasi Skor:** Terapkan korelasi bias sekitar -1.5 poin agar lebih selaras dengan skor penilai manusia

---
