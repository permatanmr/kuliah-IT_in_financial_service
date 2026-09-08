"""
Membangun notebook praktikum (.ipynb) secara terprogram dengan nbformat,
supaya markdown, kode, dan urutan sel konsisten dan mudah diperbarui.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

# ============================================================
# 0. Sampul
# ============================================================
md("""# Praktikum: Arsitektur Data dan Analitik Jasa Keuangan
### Pertemuan 5 — Mata Kuliah IT in Financial Service
**Program Studi Digital Business Technology, School of STEM, Universitas Prasetiya Mulya**

---

**Tujuan Praktikum (Sub-CPMK-5):**
Mahasiswa mampu merancang model data dan membangun pipeline analitik untuk kasus jasa keuangan.

**Yang akan dikerjakan pada notebook ini:**
1. Membersihkan dataset transaksi sintetis (1 juta baris)
2. Membangun agregasi kohort nasabah dan metrik retensi
3. Menghitung metrik kunci jasa keuangan (CASA ratio, NPL/TWP90, CAC, LTV, cost-to-income ratio)
4. Menjalankan kueri analitik dengan **DuckDB** dan membandingkannya dengan **pandas**
5. Visualisasi dashboard sederhana dengan **Plotly**

**Asesmen:** Notebook analitik + interpretasi bisnis (bobot 5% dari nilai akhir mata kuliah).

> Sebelum menjalankan notebook ini, jalankan `python 01_generate_dataset.py` satu kali untuk
> membangkitkan dataset di folder `data/`.
""")

code("""import pandas as pd
import numpy as np
import duckdb
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import time

pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
DATA_DIR = Path("data")
print("Library siap. DuckDB versi:", duckdb.__version__)
""")

# ============================================================
# BAGIAN 1 — MEMUAT & MEMERIKSA DATA
# ============================================================
md("""## Bagian 1 — Memuat dan Memeriksa Dataset

Dataset `fact_transaksi.csv` berisi sekitar 1 juta baris transaksi, dan **sengaja dibuat kotor**
untuk mensimulasikan kondisi data mentah di institusi keuangan sungguhan (data dari banyak sistem
sumber, migrasi sistem lama, input manual, dsb). `dim_nasabah.csv` berisi dimensi nasabah dengan
riwayat **SCD Type-2** sederhana (kolom `valid_from`, `valid_to`, `is_current`).

Langkah pertama dalam pipeline analitik apa pun: **selalu periksa data sebelum dipercaya.**
""")

code("""t0 = time.time()
fact_raw = pd.read_csv(DATA_DIR / "fact_transaksi.csv")
dim_nasabah = pd.read_csv(DATA_DIR / "dim_nasabah.csv", parse_dates=["valid_from", "valid_to"])
print(f"Waktu muat data: {time.time() - t0:.2f} detik")
print(f"fact_transaksi : {len(fact_raw):,} baris, {fact_raw.shape[1]} kolom")
print(f"dim_nasabah    : {len(dim_nasabah):,} baris, {dim_nasabah.shape[1]} kolom")
fact_raw.head()
""")

code("""# Profil kualitas data awal — jangan langsung analitik sebelum tahu kondisi datanya
print("=== Tipe data ===")
print(fact_raw.dtypes)
print()
print("=== Jumlah nilai kosong per kolom ===")
print(fact_raw.isna().sum())
print()
print("=== Jumlah baris duplikat (transaction_id) ===")
print(fact_raw["transaction_id"].duplicated().sum())
""")

md("""**Pertanyaan Refleksi 1.1:** Dari hasil profil di atas, kolom mana yang paling bermasalah dan mengapa
hal itu penting untuk institusi keuangan yang datanya akan digunakan pada laporan regulasi?

*(Tulis jawaban Anda pada sel markdown di bawah ini.)*
""")

md("_Jawaban:_ ")

# ============================================================
# BAGIAN 2 — PEMBERSIHAN DATA
# ============================================================
md("""## Bagian 2 — Membersihkan Dataset Transaksi

Kita akan menangani tujuh masalah kualitas data secara berurutan. Setiap langkah mencatat
berapa baris yang terdampak, sehingga proses pembersihan **terlacak (traceable)** —
prinsip yang sama dengan *data lineage* yang dibahas pada sesi teori.
""")

code("""fact = fact_raw.copy()
log_pembersihan = []

def catat(langkah, sebelum, sesudah):
    log_pembersihan.append({
        "langkah": langkah,
        "baris_sebelum": sebelum,
        "baris_sesudah": sesudah,
        "baris_terdampak": sebelum - sesudah,
    })

n0 = len(fact)
""")

code("""# 2.1 — Hapus baris duplikat murni (transaction_id sama)
n_sebelum = len(fact)
fact = fact.drop_duplicates(subset="transaction_id", keep="first")
catat("Hapus duplikat transaction_id", n_sebelum, len(fact))
print(f"Baris duplikat dihapus: {n_sebelum - len(fact):,}")
""")

code("""# 2.2 — Standardisasi format tanggal (campuran ISO dan dd/mm/yyyy)
def parse_tanggal_campuran(kolom):
    # Coba format default dulu, baris yang gagal (NaT) dicoba lagi dengan format dd/mm/yyyy
    hasil = pd.to_datetime(kolom, errors="coerce", format="mixed")
    return hasil

fact["tanggal"] = parse_tanggal_campuran(fact["tanggal"])
n_gagal_parse = fact["tanggal"].isna().sum()
print(f"Baris yang gagal diparse menjadi tanggal valid: {n_gagal_parse:,}")
""")

code("""# 2.3 — Standardisasi kapitalisasi kanal (\"ATM\" vs \"atm\" vs \"Atm\")
print("Nilai unik kolom kanal sebelum standardisasi:")
print(fact["kanal"].unique())

fact["kanal"] = fact["kanal"].str.strip().str.title()
fact["kanal"] = fact["kanal"].replace({"Qris": "QRIS"})

print("\\nNilai unik kolom kanal setelah standardisasi:")
print(fact["kanal"].unique())
""")

code("""# 2.4 — Tangani customer_id yang tidak valid (kosong, None, atau string 'UNKNOWN')
n_sebelum = len(fact)
mask_invalid_cust = fact["customer_id"].isna() | fact["customer_id"].isin(["", "UNKNOWN"])
print(f"Transaksi dengan customer_id tidak valid: {mask_invalid_cust.sum():,} ({mask_invalid_cust.mean():.2%})")

# Keputusan bisnis: transaksi tanpa identitas nasabah yang valid TIDAK BISA dipakai
# untuk analitik berbasis nasabah (kohort, CASA, dsb), sehingga dikeluarkan dari fact
# — namun tetap dicatat jumlahnya karena ini indikator kualitas data yang harus dilaporkan.
fact_tanpa_identitas = fact[mask_invalid_cust].copy()
fact = fact[~mask_invalid_cust].copy()
catat("Keluarkan transaksi tanpa customer_id valid", n_sebelum, len(fact))
""")

code("""# 2.5 — Tangani nilai nominal yang hilang, negatif, dan outlier ekstrem
print("Statistik nominal sebelum dibersihkan:")
print(fact["nominal"].describe())

n_sebelum = len(fact)

# Nominal hilang: pada institusi keuangan, transaksi tanpa nominal tidak bisa
# diasumsikan nol — baris ini dikeluarkan karena tidak bisa dipertanggungjawabkan.
fact = fact[fact["nominal"].notna()]

# Nominal negatif: kemungkinan kesalahan input tanda, kita ambil nilai absolutnya
n_negatif = (fact["nominal"] < 0).sum()
fact["nominal"] = fact["nominal"].abs()
print(f"\\nNominal negatif yang dikoreksi (diabsolutkan): {n_negatif:,}")

# Outlier ekstrem: gunakan batas IQR yang diperlebar (metode umum untuk data finansial
# yang secara alami skewed) alih-alih menghapus, kita batasi (cap) pada persentil 99.5
batas_atas = fact["nominal"].quantile(0.995)
n_outlier = (fact["nominal"] > batas_atas).sum()
fact["nominal"] = fact["nominal"].clip(upper=batas_atas)
print(f"Outlier di atas persentil 99.5 (Rp {batas_atas:,.0f}) di-cap: {n_outlier:,} baris")

catat("Bersihkan nominal (hilang/negatif/outlier)", n_sebelum, len(fact))
""")

code("""# 2.6 — Buang baris yang gagal diparse tanggalnya (setelah semua langkah lain)
n_sebelum = len(fact)
fact = fact[fact["tanggal"].notna()]
catat("Hapus baris tanggal tidak valid", n_sebelum, len(fact))

print("\\n=== Ringkasan Log Pembersihan ===")
log_df = pd.DataFrame(log_pembersihan)
log_df["persen_terdampak"] = (log_df["baris_terdampak"] / n0 * 100).round(2)
log_df
""")

code("""print(f"Baris awal (termasuk duplikat)  : {n0:,}")
print(f"Baris setelah pembersihan        : {len(fact):,}")
print(f"Total baris dibuang               : {n0 - len(fact):,} ({(n0-len(fact))/n0:.2%})")
print()
print("Cek akhir — tidak ada lagi nilai kosong pada kolom kunci:")
print(fact[["transaction_id", "customer_id", "tanggal", "nominal", "kanal"]].isna().sum())
""")

md("""**Pertanyaan Refleksi 2.1:** Kita memilih men-*cap* outlier nominal pada persentil 99.5 alih-alih
menghapusnya. Dalam konteks deteksi fraud, mengapa keputusan ini bisa jadi kurang tepat?
Apa alternatif yang lebih aman?

*(Tulis jawaban Anda di sini.)*
""")

md("_Jawaban:_ ")

code("""# Simpan dataset bersih agar bisa dipakai ulang tanpa mengulang seluruh pembersihan
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
fact.to_parquet(OUTPUT_DIR / "fact_transaksi_bersih.parquet", index=False)
print(f"Dataset bersih disimpan: {len(fact):,} baris -> output/fact_transaksi_bersih.parquet")
""")

# ============================================================
# BAGIAN 3 — DUCKDB VS PANDAS
# ============================================================
md("""## Bagian 3 — DuckDB vs pandas: Kueri Analitik pada Skala Besar

DuckDB adalah *in-process* analytical database yang menjalankan SQL langsung di atas
DataFrame pandas atau file Parquet/CSV, tanpa perlu server terpisah — cocok untuk analitik
lokal berskala jutaan baris. Kita bandingkan waktu eksekusi kueri agregasi yang sama
pada pandas murni dan pada DuckDB.

> **Catatan penting:** Pada kueri sederhana dengan data yang sudah ada di memori (seperti
> `groupby` satu kolom di bawah), pandas sering kali *sama cepat atau lebih cepat* daripada
> DuckDB, karena ada overhead saat DuckDB memindai ulang DataFrame pandas setiap kali
> dipanggil. Keunggulan DuckDB baru terasa jelas pada: (1) kueri dengan **banyak join dan
> agregasi kompleks sekaligus**, (2) data yang **tidak muat di memori** dan dibaca langsung
> dari file Parquet/CSV di disk, dan (3) ketika Anda ingin menulis logikanya dalam SQL
> daripada rangkaian operasi pandas. Percobaan di bawah ini dirancang untuk memperlihatkan
> kedua sisinya secara jujur — jangan berasumsi satu alat selalu menang.
""")

code("""con = duckdb.connect()

# Daftarkan DataFrame pandas sebagai \"tabel virtual\" yang bisa langsung diquery dengan SQL
con.register("fact_transaksi", fact)
con.register("dim_nasabah", dim_nasabah)

print("Tabel yang terdaftar di DuckDB:")
print(con.sql("SHOW TABLES").df())
""")

code("""# Kueri sederhana: total nominal transaksi per kanal — satu groupby, satu kolom
t0 = time.time()
hasil_pandas = fact.groupby("kanal")["nominal"].sum().sort_values(ascending=False)
waktu_pandas = time.time() - t0

t0 = time.time()
hasil_duckdb = con.sql('''
    SELECT kanal, SUM(nominal) AS total_nominal
    FROM fact_transaksi
    GROUP BY kanal
    ORDER BY total_nominal DESC
''').df()
waktu_duckdb = time.time() - t0

print(f"Waktu pandas  groupby : {waktu_pandas*1000:.1f} ms")
print(f"Waktu DuckDB  SQL     : {waktu_duckdb*1000:.1f} ms")
print("-> Untuk kueri satu-kolom sesederhana ini, jangan kaget jika pandas menang.")
print()
print(hasil_duckdb)
""")

md("""**Pertanyaan Refleksi 3.1:** Kueri di atas hanya melakukan satu `groupby` sederhana.
Sekarang kita coba kueri yang jauh lebih berat: **join** tiga arah antara transaksi, dimensi
nasabah, DAN sebuah agregasi kohort bulanan sekaligus per segmen — mendekati kompleksitas kueri
laporan regulasi sungguhan. Perhatikan bagaimana selisih waktunya berubah dibanding kueri sederhana
di atas, lalu jelaskan mengapa hal itu terjadi.
""")

code("""# Kueri berat: total nominal per (segmen nasabah x bulan transaksi x kanal),
# hanya untuk nasabah dengan status is_current = True — mensimulasikan laporan
# ringkasan bulanan per segmen yang biasa diminta manajemen risiko.
t0 = time.time()
fact_gabung = fact.merge(
    dim_nasabah[dim_nasabah["is_current"]][["customer_id", "segmen", "cabang"]],
    on="customer_id", how="inner",
)
fact_gabung["bulan"] = fact_gabung["tanggal"].dt.to_period("M").astype(str)
hasil_join_pandas = (
    fact_gabung.groupby(["segmen", "bulan", "kanal"])["nominal"]
    .agg(["sum", "count", "mean"])
    .reset_index()
    .sort_values("sum", ascending=False)
)
waktu_join_pandas = time.time() - t0

t0 = time.time()
hasil_join_duckdb = con.sql('''
    SELECT
        d.segmen,
        strftime(f.tanggal, '%Y-%m') AS bulan,
        f.kanal,
        SUM(f.nominal) AS sum,
        COUNT(*) AS count,
        AVG(f.nominal) AS mean
    FROM fact_transaksi f
    JOIN dim_nasabah d
      ON f.customer_id = d.customer_id
     AND d.is_current = true
    GROUP BY d.segmen, bulan, f.kanal
    ORDER BY sum DESC
''').df()
waktu_join_duckdb = time.time() - t0

print(f"Waktu pandas  (merge + groupby 3 dimensi) : {waktu_join_pandas*1000:.1f} ms")
print(f"Waktu DuckDB  (SQL join 3 dimensi)         : {waktu_join_duckdb*1000:.1f} ms")
print(f"Jumlah baris hasil                          : {len(hasil_join_duckdb):,}")
print()
print(hasil_join_duckdb.head())
""")

md("_Jawaban Refleksi 3.1:_ ")

# ============================================================
# BAGIAN 4 — AGREGASI KOHORT & RETENSI
# ============================================================
md("""## Bagian 4 — Agregasi Kohort Nasabah dan Metrik Retensi

**Analisis kohort** mengelompokkan nasabah berdasarkan periode mereka pertama kali terdaftar
(kohort akuisisi), lalu melacak seberapa banyak dari kelompok tersebut yang masih aktif
bertransaksi pada bulan-bulan berikutnya. Ini adalah teknik analitik standar untuk mengukur
retensi nasabah pada bisnis jasa keuangan digital.
""")

code("""# Tentukan bulan kohort setiap nasabah = bulan transaksi pertama yang tercatat pada data bersih
transaksi_per_nasabah = fact.groupby("customer_id")["tanggal"].min().rename("tanggal_transaksi_pertama")
kohort_nasabah = transaksi_per_nasabah.dt.to_period("M").rename("bulan_kohort")

fact_kohort = fact.merge(kohort_nasabah, left_on="customer_id", right_index=True)
fact_kohort["bulan_transaksi"] = fact_kohort["tanggal"].dt.to_period("M")

# Jarak dalam bulan antara transaksi dan bulan kohort nasabah tersebut
fact_kohort["indeks_bulan"] = (
    (fact_kohort["bulan_transaksi"] - fact_kohort["bulan_kohort"]).apply(lambda x: x.n)
)

fact_kohort[["customer_id", "bulan_kohort", "bulan_transaksi", "indeks_bulan"]].head()
""")

code("""# Tabel kohort: jumlah nasabah unik yang aktif pada setiap indeks bulan, per kohort akuisisi
tabel_kohort = (
    fact_kohort.groupby(["bulan_kohort", "indeks_bulan"])["customer_id"]
    .nunique()
    .reset_index()
    .pivot(index="bulan_kohort", columns="indeks_bulan", values="customer_id")
)

ukuran_kohort = tabel_kohort[0]
tabel_retensi = tabel_kohort.divide(ukuran_kohort, axis=0)

print("Ukuran setiap kohort (jumlah nasabah baru per bulan):")
print(ukuran_kohort)
""")

md("""Perhatikan bahwa kohort di bulan-bulan awal (2023) jauh lebih besar daripada kohort
bulan-bulan terakhir. Ini **bukan bug** — ini pola umum \"birthday problem\": karena dataset
mencakup +/- 3,5 tahun transaksi, sebagian besar nasabah yang pernah bertransaksi sudah
melakukan transaksi pertamanya di awal periode data. Kohort-kohort besar dan matang inilah
yang paling berguna untuk melihat pola retensi jangka panjang, sehingga kita fokus ke kohort
tersebut pada heatmap berikut.
""")

code("""# Fokus ke kohort-kohort yang cukup besar (>= 50 nasabah baru) agar persentase retensi
# tidak dipengaruhi angka kecil, dan tampilkan retensi bulan ke-0 s.d. bulan ke-6
kohort_signifikan = ukuran_kohort[ukuran_kohort >= 50].index
kolom_tersedia = [c for c in range(0, 7) if c in tabel_retensi.columns]
retensi_plot = (tabel_retensi.loc[kohort_signifikan, kolom_tersedia] * 100).round(1)
retensi_plot.index = retensi_plot.index.astype(str)

fig_retensi = px.imshow(
    retensi_plot,
    labels=dict(x="Bulan ke-N sejak kohort", y="Kohort akuisisi", color="Retensi (%)"),
    color_continuous_scale="Teal",
    text_auto=True,
    aspect="auto",
    title="Heatmap Retensi Nasabah per Kohort Akuisisi Bulanan (%) — Kohort ≥ 50 Nasabah",
)
fig_retensi.update_layout(height=500)
fig_retensi.show()
""")

md("""**Pertanyaan Refleksi 4.1:** Perhatikan heatmap retensi di atas. Apakah ada kohort tertentu yang
retensinya jauh lebih rendah dari kohort lain pada indeks bulan yang sama? Sebutkan kemungkinan
penyebab bisnisnya (bukan penyebab teknis pada data).

*(Tulis jawaban Anda di sini.)*
""")

md("_Jawaban:_ ")

# ============================================================
# BAGIAN 5 — METRIK KUNCI JASA KEUANGAN
# ============================================================
md("""## Bagian 5 — Menghitung Metrik Kunci Jasa Keuangan

Kita akan menghitung lima metrik yang dibahas pada sesi teori: **CASA ratio**, **NPL/TWP90**,
**CAC**, **LTV**, dan **cost-to-income ratio**. Karena dataset ini berisi data transaksi
(bukan data pinjaman atau biaya operasional penuh), sebagian metrik dihitung dengan **proksi**
yang wajar dan asumsinya dijelaskan secara eksplisit — praktik yang umum ketika data lengkap
belum tersedia, asalkan asumsi didokumentasikan dengan transparan.
""")

code("""# 5.1 — CASA ratio (proksi)
# Definisi asli: (Giro + Tabungan) / Total Dana Pihak Ketiga x 100%
# Proksi pada dataset ini: transaksi "Setor Tunai" dan "Top Up e-Wallet" dianggap mewakili
# perilaku dana murah (giro/tabungan), dibandingkan seluruh nominal transaksi sebagai basis.
dana_murah = fact[fact["jenis_transaksi"].isin(["Setor Tunai", "Top Up e-Wallet"])]["nominal"].sum()
total_dana = fact["nominal"].sum()
casa_ratio_proksi = dana_murah / total_dana * 100

print(f"CASA ratio (proksi dari data transaksi)  : {casa_ratio_proksi:.2f}%")
print("Catatan: metrik CASA yang sesungguhnya dihitung dari saldo dana pihak ketiga,")
print("bukan dari volume transaksi — di sini hanya digunakan sebagai ilustrasi proksi.")
""")

code("""# 5.2 — NPL / TWP90 (simulasi, karena dataset ini tidak memiliki data pinjaman)
# Kita simulasikan status kredit sederhana per nasabah untuk keperluan latihan menghitung metrik ini.
rng = np.random.default_rng(7)
nasabah_unik = fact["customer_id"].unique()
status_kredit = pd.DataFrame({
    "customer_id": nasabah_unik,
    "punya_pinjaman": rng.random(len(nasabah_unik)) < 0.35,
})
status_kredit["wanprestasi_90hari"] = np.where(
    status_kredit["punya_pinjaman"],
    rng.random(len(nasabah_unik)) < 0.04,   # ~4% dari yang punya pinjaman mengalami TWP90
    False,
)

n_pinjaman = status_kredit["punya_pinjaman"].sum()
n_wanprestasi = status_kredit["wanprestasi_90hari"].sum()
twp90 = n_wanprestasi / n_pinjaman * 100

print(f"Jumlah nasabah dengan pinjaman aktif (simulasi) : {n_pinjaman:,}")
print(f"Jumlah wanprestasi > 90 hari (simulasi)         : {n_wanprestasi:,}")
print(f"NPL / TWP90 (simulasi)                          : {twp90:.2f}%")
""")

code("""# 5.3 — CAC dan LTV (proksi)
# CAC = total biaya akuisisi (diasumsikan) / jumlah nasabah baru pada seluruh periode data
# LTV = rata-rata margin per nasabah per bulan x estimasi masa hidup nasabah (bulan)
#
# Asumsi di bawah ini dipilih agar berada pada kisaran yang wajar untuk bank digital/fintech
# skala menengah di Indonesia — SELALU nyatakan asumsi seperti ini secara eksplisit pada
# laporan bisnis nyata, karena hasil akhirnya sangat bergantung padanya.
TOTAL_BIAYA_AKUISISI_ASUMSI = 3_000_000_000   # asumsi total biaya marketing sepanjang periode data (Rp)
MARGIN_ASUMSI = 0.03                           # asumsi margin bank atas volume transaksi, 3%
MASA_HIDUP_BULAN_ASUMSI = 24                   # estimasi rata-rata nasabah aktif bertransaksi 24 bulan

total_nasabah_terakuisisi = fact["customer_id"].nunique()
cac_rata_rata = TOTAL_BIAYA_AKUISISI_ASUMSI / total_nasabah_terakuisisi

# Margin bulanan per nasabah = rata-rata nominal transaksi per bulan aktif x margin asumsi
nominal_per_nasabah_per_bulan = (
    fact_kohort.groupby("customer_id")["nominal"].sum()
    / fact_kohort.groupby("customer_id")["bulan_transaksi"].nunique()
)
margin_bulanan_per_nasabah = nominal_per_nasabah_per_bulan * MARGIN_ASUMSI
ltv_per_nasabah = margin_bulanan_per_nasabah * MASA_HIDUP_BULAN_ASUMSI

rasio_ltv_cac = ltv_per_nasabah.mean() / cac_rata_rata

print(f"Total nasabah terakuisisi (basis CAC)      : {total_nasabah_terakuisisi:,}")
print(f"CAC rata-rata (asumsi)                     : Rp {cac_rata_rata:,.0f}")
print(f"LTV rata-rata per nasabah (asumsi)          : Rp {ltv_per_nasabah.mean():,.0f}")
print(f"Rasio LTV : CAC                             : {rasio_ltv_cac:.2f}")
print()
print("Aturan umum industri: rasio LTV:CAC yang sehat umumnya berada di atas 3:1.")
""")

code("""# 5.4 — Cost-to-income ratio (proksi sederhana)
# Formula asli: Beban Operasional / Pendapatan Operasional
# Proksi: pendapatan operasional = margin asumsi x total nominal transaksi bersih
#         beban operasional      = biaya akuisisi (5.3) + asumsi biaya operasional tetap per bulan
BIAYA_OPERASIONAL_TETAP_PER_BULAN = 150_000_000  # asumsi biaya SDM, sistem, sewa, dsb (Rp)

jumlah_bulan_data = fact["tanggal"].dt.to_period("M").nunique()
pendapatan_operasional = total_dana * MARGIN_ASUMSI
beban_operasional_asumsi = (
    TOTAL_BIAYA_AKUISISI_ASUMSI + BIAYA_OPERASIONAL_TETAP_PER_BULAN * jumlah_bulan_data
)
cost_to_income = beban_operasional_asumsi / pendapatan_operasional * 100

print(f"Estimasi pendapatan operasional (proksi) : Rp {pendapatan_operasional:,.0f}")
print(f"Estimasi beban operasional (proksi)      : Rp {beban_operasional_asumsi:,.0f}")
print(f"Cost-to-income ratio (proksi)            : {cost_to_income:.2f}%")
print()
print("Acuan umum industri perbankan: cost-to-income ratio yang efisien biasanya di bawah 50-60%.")
""")

code("""# Ringkasan seluruh metrik dalam satu tabel
ringkasan_metrik = pd.DataFrame([
    {"Metrik": "CASA ratio (proksi)", "Nilai": f"{casa_ratio_proksi:.2f}%"},
    {"Metrik": "NPL / TWP90 (simulasi)", "Nilai": f"{twp90:.2f}%"},
    {"Metrik": "CAC rata-rata (asumsi)", "Nilai": f"Rp {cac_rata_rata:,.0f}"},
    {"Metrik": "LTV rata-rata (asumsi)", "Nilai": f"Rp {ltv_per_nasabah.mean():,.0f}"},
    {"Metrik": "Rasio LTV : CAC", "Nilai": f"{rasio_ltv_cac:.2f}"},
    {"Metrik": "Cost-to-income ratio (proksi)", "Nilai": f"{cost_to_income:.2f}%"},
])
ringkasan_metrik
""")

md("""**Pertanyaan Refleksi 5.1:** Semua metrik pada Bagian 5 di atas menggunakan proksi atau simulasi
karena dataset transaksi ini tidak memiliki data saldo, pinjaman, atau biaya operasional yang lengkap.
Jika Anda bekerja di institusi keuangan sungguhan, tabel data (fact/dimension) apa saja yang perlu
ditambahkan ke arsitektur data agar metrik-metrik ini bisa dihitung tanpa proksi?

*(Tulis jawaban Anda di sini — kaitkan dengan star schema yang dibahas pada sesi teori.)*
""")

md("_Jawaban:_ ")

# ============================================================
# BAGIAN 6 — DASHBOARD SEDERHANA
# ============================================================
md("""## Bagian 6 — Dashboard Analitik Sederhana

Bagian ini menyusun beberapa visualisasi kunci menjadi satu tampilan ringkas — inilah yang akan
menjadi dasar dashboard Streamlit pada file `dashboard_streamlit.py` yang menyertai notebook ini.
""")

code("""# Grafik 1 — Tren volume transaksi bulanan
tren_bulanan = fact.groupby(fact["tanggal"].dt.to_period("M"))["nominal"].agg(["sum", "count"]).reset_index()
tren_bulanan["tanggal"] = tren_bulanan["tanggal"].astype(str)

fig_tren = go.Figure()
fig_tren.add_trace(go.Scatter(
    x=tren_bulanan["tanggal"], y=tren_bulanan["sum"],
    mode="lines", name="Total Nominal", line=dict(color="#00376A", width=2.5),
))
fig_tren.update_layout(
    title="Tren Total Nominal Transaksi per Bulan",
    xaxis_title="Bulan", yaxis_title="Total Nominal (Rp)",
    height=420, template="plotly_white",
)
fig_tren.show()
""")

code("""# Grafik 2 — Komposisi transaksi per kanal
komposisi_kanal = fact.groupby("kanal")["nominal"].sum().sort_values(ascending=True).reset_index()

fig_kanal = px.bar(
    komposisi_kanal, x="nominal", y="kanal", orientation="h",
    title="Total Nominal Transaksi per Kanal",
    labels={"nominal": "Total Nominal (Rp)", "kanal": ""},
    color_discrete_sequence=["#E30513"],
)
fig_kanal.update_layout(height=380, template="plotly_white")
fig_kanal.show()
""")

code("""# Grafik 3 — Distribusi segmen nasabah aktif
segmen_aktif = dim_nasabah[dim_nasabah["is_current"]]["segmen"].value_counts().reset_index()
segmen_aktif.columns = ["segmen", "jumlah"]

fig_segmen = px.bar(
    segmen_aktif.sort_values("jumlah"), x="jumlah", y="segmen", orientation="h",
    title="Jumlah Nasabah Aktif per Segmen",
    labels={"jumlah": "Jumlah Nasabah", "segmen": ""},
    color_discrete_sequence=["#164E91"],
)
fig_segmen.update_layout(height=320, template="plotly_white")
fig_segmen.show()
""")

md("""## Penutup dan Langkah Selanjutnya

Anda telah menjalankan pipeline analitik dari data mentah hingga dashboard: membersihkan data,
membandingkan performa pandas dan DuckDB, membangun analisis kohort, menghitung metrik kunci
jasa keuangan, dan memvisualisasikannya.

**Untuk laporan akhir (asesmen notebook 5%):**
1. Lengkapi seluruh sel jawaban refleksi (1.1, 2.1, 3.1, 4.1, 5.1) di atas.
2. Tambahkan interpretasi bisnis 1 halaman terpisah (lihat dokumen panduan praktikum) yang
   menjelaskan temuan utama dan rekomendasi untuk institusi keuangan fiktif pada dataset ini.
3. Jalankan `streamlit run dashboard_streamlit.py` untuk melihat versi dashboard interaktifnya,
   lalu sertakan tangkapan layarnya pada laporan.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

with open("praktikum_analitik_jasa_keuangan.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook dibuat: praktikum_analitik_jasa_keuangan.ipynb")
print(f"Total sel: {len(cells)}")
