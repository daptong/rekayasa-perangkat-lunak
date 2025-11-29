# UML Diagram Assessment System
## Multilingual Assessment using mBERT and Graph Edit Distance

Sistem penilaian otomatis untuk diagram kelas UML multibahasa (Indonesia/Inggris) menggunakan multilingual BERT (mBERT) untuk kesamaan semantik dan Graph Edit Distance (GED) untuk perbandingan struktural

## Getting Started

```bash
# Check current Python
python --version

# If you have multiple Python versions, explicitly use 3.10
py -3.10 --version

# Create and activate a 3.10 virtual environment
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1

# Upgrade pip and install requirements
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Latar Belakang Metodologi

Sistem ini mengimplementasikan metodologi yang dijelaskan dalam penelitian tentang penilaian otomatis diagram UML:

1. **Dukungan Multibahasa**: mBERT memungkinkan perbandingan semantik antara bahasa Indonesia dan Inggris
2. **Analisis Struktural**: Graph Edit Distance menangkap perbedaan semantik dan struktural
3. **Validasi**: Korelasi Pearson dengan skor manusia memvaldasi penilaian otomatis
4. **Target Kinerja**: Mencapai r ≥ 0.76 dengan penilai ahli

---

## Inovasi Utama

- **Semantic-Aware GED**: Mengintegrasikan embedding mBERT ke dalam fungsi cost GED tradisional
- **Penilaian Multibahasa**: Pemahaman multibahasa tanpa penerjemahan langsung
- **Penilaian Komponen**: Skor terpisah untuk semantik, struktur, dan hubungan
- **Validasi Komprehensi**: Bermacam metrik dan visualisasi

## Scoring

### Rentang Skor

- **90-100**: Sangat baik - Hanya perbedaan terjemahan minor
- **70-89**: Baik - Beberapa elemen hilang atau variasi semantik
- **50-69**: Cukup - Banyak kesalahan atau hubungan yang salah
- **0-49**: Buruk - Perbedaan struktural yang signifikan

### Korelasi Pearson

- **r > 0.80**: Kesepakatan kuat dengan penilai manusia
- **r = 0.70-0.80**: Kesepakatan baik (rentang target)
- **r = 0.50-0.70**: Kesepakatan sedang
- **r < 0.50**: Kesepakatan lemah (perlu perbaikan)

### Statistik

- **p < 0.05**: Hasil signifikan secara statistik
- **p ≥ 0.05**: Hasil mungkin terjadi karena kebetulan

---

## Konfigurasi

Ubah `config.yaml` untuk melakukan kustomisasi:

```yaml
assessment:
  weights:
    class_name: 0.5      # Bobot untuk kesamaan nama kelas
    attributes: 0.25     # Bobot untuk kesamaan atribut
    methods: 0.25        # Bobot untuk kesamaan metode
  
  penalties:
    node_deletion: 1.0
    node_insertion: 1.0
    edge_deletion: 0.5
    wrong_relationship: 0.8

mbert:
  model_name: 'bert-base-multilingual-cased'
  max_length: 512

evaluation:
  correlation_threshold: 0.76  # Target dari penelitian
```

---

### Menggunakan Modul Individu

```python
from modules.mbert_processor import MBERTProcessor
from modules.ged_calculator import GEDCalculator

# Process dengan mBERT
mbert = MBERTProcessor()
embedding = mbert.generate_embedding("Pelanggan")
similarity = mbert.calculate_semantic_similarity("Customer", "Pelanggan")

# Hitung GED
ged_calc = GEDCalculator(mbert, config)
ged_value, breakdown = ged_calc.calculate_ged(G_key, G_student)
```

---

### 1. **Pemrosesan Semantik mBERT**

- Model: `bert-base-multilingual-cased`
- Menghasilkan 768-dimensional embedding
- Menghitung cost semantik: `Cost = 1 - CosineSimilarity`
- Mendukung perbandingan multibahasa Indonesia ↔ Inggris

### 2. **Graph Edit Distance**

Menghitung cost minimum untuk mengubah graf siswa menjadi graf kunci:

**Node Costs (Simpul):**
```
C_node = w₁·C_name + w₂·C_attr + w₃·C_method
```

**Edge Costs: (Sisi)**
- Tipe hubungan yang benar: 0.0
- Tipe hubungan yang salah: 0.8
- Edge hilang/lebih: 0.5

**Total GED:**
```
GED = Σ (node_operations + edge_operations)
```

### 3. **Penilaian**

```
Score = (1 / (1 + GED)) × 100
```

Menghasilkan:
- Skor kesamaan keseluruhan (0-100)
- Skor semantik (kesamaan nama)
- Skor struktural (jumlah node/edge)
- Skor hubungan (kebenaran edge)

### 4. **Evaluasi**

- **Korelasi Pearson**: Skor sistem vs manusia
- **Target**: r ≥ 0.76 (dari penelitian)
- **Validasi**: p-value < 0.05 untuk signifikansi
- **Metrik Kesalahan**: MAE, RMSE, tingkat kesepakatan (agreement rates)

---





