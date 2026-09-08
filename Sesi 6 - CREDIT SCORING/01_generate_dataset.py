"""
Membangkitkan dataset sintetis untuk praktikum Credit Scoring, Alternative Data,
dan Machine Learning (Pertemuan 6 — IT in Financial Service, DBT-3204).

Dataset ini MURNI SINTETIS (tidak berasal dari data nasabah sungguhan) dan dirancang
untuk memiliki:
- Fitur data tradisional (mirip SLIK OJK): riwayat pinjaman, rasio utang, lama riwayat kredit
- Fitur data alternatif: pola pulsa/telko, perilaku e-commerce, cashflow UMKM
- Target biner `default_flag` (1 = gagal bayar dalam 90 hari / setara NPL)
- Atribut demografis (wilayah, kelompok usia, jenis kelamin) — DIGUNAKAN HANYA untuk
  latihan uji bias/fairness pada Bagian 3, bukan sebagai fitur model prediksi.
- Bias yang DISENGAJA ditanamkan pada satu wilayah (Luar Jawa) agar mahasiswa memiliki
  sesuatu yang nyata untuk ditemukan saat uji bias — ini adalah desain pedagogis.

Jalankan sekali sebelum membuka notebook:
    python 01_generate_dataset.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
N = 12000
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

WILAYAH = ["Jabodetabek", "Jawa Non-Jabodetabek", "Sumatra", "Kalimantan", "Sulawesi", "Luar Jawa Lainnya"]
WILAYAH_P = [0.28, 0.22, 0.18, 0.12, 0.10, 0.10]
KELOMPOK_USIA = ["18-25", "26-35", "36-45", "46-55", "56+"]
USIA_P = [0.18, 0.32, 0.27, 0.15, 0.08]
SEGMEN = ["Karyawan Formal", "UMKM/Wirausaha", "Pekerja Informal/Gig"]
SEGMEN_P = [0.35, 0.35, 0.30]

def generate():
    df = pd.DataFrame({
        "customer_id": [f"CUST{100000+i}" for i in range(N)],
        "wilayah": RNG.choice(WILAYAH, size=N, p=WILAYAH_P),
        "kelompok_usia": RNG.choice(KELOMPOK_USIA, size=N, p=USIA_P),
        "jenis_kelamin": RNG.choice(["L", "P"], size=N, p=[0.52, 0.48]),
        "segmen_pekerjaan": RNG.choice(SEGMEN, size=N, p=SEGMEN_P),
    })

    # ---------- Fitur tradisional (mirip SLIK OJK) ----------
    # thin-file: proporsi nasabah tanpa riwayat kredit formal, lebih tinggi di segmen informal/UMKM
    thin_file_prob = np.select(
        [df["segmen_pekerjaan"] == "Pekerja Informal/Gig", df["segmen_pekerjaan"] == "UMKM/Wirausaha"],
        [0.75, 0.55], default=0.15,
    )
    df["thin_file"] = RNG.binomial(1, thin_file_prob)

    df["lama_riwayat_kredit_bulan"] = np.where(
        df["thin_file"] == 1, 0, RNG.gamma(shape=3.0, scale=14, size=N).round().clip(1, 240)
    )
    df["jumlah_fasilitas_kredit_aktif"] = np.where(
        df["thin_file"] == 1, 0, RNG.poisson(1.6, size=N)
    )
    df["rasio_utang_pendapatan"] = np.clip(RNG.normal(0.35, 0.18, size=N), 0.02, 1.4).round(3)
    df["riwayat_keterlambatan_90hari"] = np.where(
        df["thin_file"] == 1, np.nan, RNG.poisson(0.35, size=N)
    )

    # ---------- Fitur alternatif: data telko ----------
    # infrastruktur telko cenderung sedikit kurang stabil di luar Jawa — korelasi geografis
    # yang SAH secara statistik, namun berpotensi menjadi proxy discrimination bila model
    # tidak diperiksa, karena berkorelasi dengan wilayah tanpa mencerminkan risiko individu.
    telko_penalti = np.where(df["wilayah"] == "Luar Jawa Lainnya", -0.12, 0.0)
    df["konsistensi_isi_ulang_pulsa"] = np.clip(
        RNG.normal(0.7 + telko_penalti, 0.2, size=N), 0, 1
    ).round(3)
    df["lama_kepemilikan_nomor_bulan"] = RNG.gamma(shape=4, scale=18, size=N).round().clip(1, 200)
    df["rata_rata_pulsa_bulanan_rp"] = (RNG.lognormal(mean=10.5, sigma=0.55, size=N)).round(-3).clip(10000, 500000)

    # ---------- Fitur alternatif: perilaku e-commerce ----------
    df["frekuensi_transaksi_ecommerce_bulanan"] = RNG.poisson(4.5, size=N)
    df["rating_rata_penjual_pembeli"] = np.clip(RNG.normal(4.4, 0.5, size=N), 1, 5).round(2)
    df["rasio_retur_pembelian"] = np.clip(RNG.beta(1.2, 12, size=N), 0, 0.6).round(3)

    # ---------- Fitur alternatif: cashflow UMKM ----------
    is_umkm = (df["segmen_pekerjaan"] == "UMKM/Wirausaha").astype(int)
    base_cashflow = RNG.lognormal(mean=15.2, sigma=0.6, size=N)
    df["rata_rata_kas_masuk_bulanan_rp"] = np.where(
        is_umkm == 1, base_cashflow.round(-4), (base_cashflow * 0.55).round(-4)
    ).clip(500000, 200000000)
    df["volatilitas_kas_cv"] = np.clip(
        np.where(is_umkm == 1, RNG.normal(0.42, 0.15, size=N), RNG.normal(0.22, 0.1, size=N)), 0.02, 1.2
    ).round(3)
    df["jumlah_hari_kas_negatif_per_bulan"] = np.where(
        is_umkm == 1, RNG.poisson(3.2, size=N), RNG.poisson(1.0, size=N)
    ).clip(0, 30)

    # ---------- Target: probabilitas gagal bayar (default_flag) ----------
    # Sinyal risiko "sah" — berlaku sama untuk semua wilayah/kelompok
    logit = (
        -3.15
        + 3.4 * df["rasio_utang_pendapatan"]
        + 0.85 * df["riwayat_keterlambatan_90hari"].fillna(0.9)   # thin-file diberi nilai netral, bukan 0
        - 1.9 * df["konsistensi_isi_ulang_pulsa"]
        + 2.6 * df["volatilitas_kas_cv"]
        + 0.14 * df["jumlah_hari_kas_negatif_per_bulan"]
        - 0.85 * df["rating_rata_penjual_pembeli"].sub(4.4)
        + 1.8 * df["rasio_retur_pembelian"]
        - 0.006 * (df["lama_riwayat_kredit_bulan"].fillna(0))
        + 2.2 * df["volatilitas_kas_cv"] * df["rasio_utang_pendapatan"]  # interaksi non-linear (menguntungkan XGBoost)
        + RNG.normal(0, 0.35, size=N)  # noise idiosinkratik
    )

    # BIAS YANG DISENGAJA (untuk latihan Bagian 3 — uji bias):
    # proxy discrimination via wilayah "Luar Jawa Lainnya" — menaikkan skor risiko
    # secara tidak adil meskipun profil keuangan sebanding, mensimulasikan proxy
    # bias yang bisa lolos dari model bila fitur wilayah/geografis tidak diperiksa.
    proxy_bias = np.where(df["wilayah"] == "Luar Jawa Lainnya", 1.5, 0.0)
    logit = logit + proxy_bias

    prob_default = 1 / (1 + np.exp(-logit))
    df["default_flag"] = RNG.binomial(1, prob_default)

    # bersihkan tipe data agar rapi untuk pandas/DuckDB
    df["riwayat_keterlambatan_90hari"] = df["riwayat_keterlambatan_90hari"].astype("Int64")

    return df


if __name__ == "__main__":
    df = generate()
    out_path = DATA_DIR / "kredit_alternatif.csv"
    df.to_csv(out_path, index=False)
    print(f"Dataset dibuat: {out_path} ({len(df):,} baris, {df.shape[1]} kolom)")
    print(f"Tingkat default keseluruhan: {df['default_flag'].mean():.2%}")
    print(df.groupby("wilayah")["default_flag"].mean().sort_values(ascending=False).round(3))
