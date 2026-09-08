"""
01_generate_dataset.py
========================
Membangkitkan dataset transaksi sintetis untuk praktikum
"Arsitektur Data dan Analitik Jasa Keuangan" (Pertemuan 5, IT in Financial Service).

Dataset ini SENGAJA dibuat kotor (missing values, duplikat, outlier, format
tanggal tidak konsisten) agar mahasiswa berlatih membersihkan data sebelum
melakukan analitik — sesuai instruksi praktikum di RPS.

Output:
- data/dim_nasabah.csv      (~20.000 nasabah, dengan riwayat SCD Type-2 sederhana)
- data/fact_transaksi.csv   (1.000.000 baris transaksi, KOTOR)

Jalankan:
    python 01_generate_dataset.py
"""

import numpy as np
import pandas as pd
from faker import Faker
from pathlib import Path

RNG_SEED = 42
N_NASABAH = 20_000
N_TRANSAKSI = 1_000_000
OUT_DIR = Path(__file__).parent / "data"
OUT_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(RNG_SEED)
fake = Faker("id_ID")
Faker.seed(RNG_SEED)

SEGMEN = ["Reguler", "Prioritas", "Wealth", "Mahasiswa"]
CABANG = [
    "Jakarta Pusat", "Jakarta Selatan", "Bandung", "Surabaya", "Medan",
    "Semarang", "Makassar", "Denpasar", "Yogyakarta", "Palembang",
]
STATUS_KYC = ["Terverifikasi", "Menunggu Verifikasi", "Perlu Update"]
KANAL = ["Mobile Banking", "ATM", "QRIS", "Internet Banking", "Teller Cabang"]
JENIS_TRANSAKSI = ["Transfer", "Pembayaran", "Tarik Tunai", "Setor Tunai", "Top Up e-Wallet"]

print("1) Membangkitkan dimensi nasabah (dim_nasabah) dengan riwayat SCD Type-2 sederhana...")

tanggal_mulai = pd.Timestamp("2023-01-01")
tanggal_akhir = pd.Timestamp("2026-06-30")

baris_nasabah = []
for i in range(1, N_NASABAH + 1):
    customer_id = f"CUST{i:06d}"
    tgl_daftar = fake.date_between(start_date=tanggal_mulai, end_date=tanggal_akhir - pd.Timedelta(days=30))
    segmen_awal = rng.choice(SEGMEN, p=[0.55, 0.25, 0.08, 0.12])
    cabang = rng.choice(CABANG)

    # Baris versi awal (valid_from = tanggal daftar)
    baris_nasabah.append({
        "customer_id": customer_id,
        "nama": fake.name(),
        "segmen": segmen_awal,
        "cabang": cabang,
        "status_kyc": rng.choice(STATUS_KYC, p=[0.8, 0.12, 0.08]),
        "valid_from": tgl_daftar,
        "valid_to": None,
        "is_current": True,
    })

    # ~15% nasabah mengalami perubahan segmen (upgrade/downgrade) → baris SCD Type-2 baru
    if rng.random() < 0.15:
        tgl_ubah = fake.date_between(start_date=tgl_daftar, end_date=tanggal_akhir)
        segmen_baru = rng.choice([s for s in SEGMEN if s != segmen_awal])
        # tutup baris lama
        baris_nasabah[-1]["valid_to"] = tgl_ubah
        baris_nasabah[-1]["is_current"] = False
        # tambahkan baris baru yang aktif
        baris_nasabah.append({
            "customer_id": customer_id,
            "nama": baris_nasabah[-1]["nama"],
            "segmen": segmen_baru,
            "cabang": cabang,
            "status_kyc": "Terverifikasi",
            "valid_from": tgl_ubah,
            "valid_to": None,
            "is_current": True,
        })

dim_nasabah = pd.DataFrame(baris_nasabah)
dim_nasabah.to_csv(OUT_DIR / "dim_nasabah.csv", index=False)
print(f"   -> {len(dim_nasabah):,} baris dimensi nasabah disimpan ke data/dim_nasabah.csv")

print("2) Membangkitkan fact_transaksi (1.000.000 baris, dibuat KOTOR secara sengaja)...")

customer_ids = dim_nasabah["customer_id"].unique()
# Bobot: sebagian kecil nasabah jauh lebih aktif (mendekati pola dunia nyata / Pareto)
bobot_aktivitas = rng.pareto(a=1.6, size=len(customer_ids)) + 0.1
bobot_aktivitas = bobot_aktivitas / bobot_aktivitas.sum()

pilihan_nasabah = rng.choice(customer_ids, size=N_TRANSAKSI, p=bobot_aktivitas)
tanggal_transaksi = pd.to_datetime(
    rng.integers(tanggal_mulai.value // 10**9, tanggal_akhir.value // 10**9, size=N_TRANSAKSI),
    unit="s",
)
jenis = rng.choice(JENIS_TRANSAKSI, size=N_TRANSAKSI, p=[0.35, 0.25, 0.15, 0.1, 0.15])
kanal = rng.choice(KANAL, size=N_TRANSAKSI, p=[0.4, 0.15, 0.25, 0.1, 0.1])

# Nominal transaksi: lognormal agar realistis (banyak transaksi kecil, sedikit yang besar)
nominal = rng.lognormal(mean=12.5, sigma=1.3, size=N_TRANSAKSI).round(0)

fact = pd.DataFrame({
    "transaction_id": [f"TRX{i:08d}" for i in range(1, N_TRANSAKSI + 1)],
    "customer_id": pilihan_nasabah,
    "tanggal": tanggal_transaksi,
    "jenis_transaksi": jenis,
    "kanal": kanal,
    "nominal": nominal,
})

# --- Sengaja mengotori data (skenario realistis untuk dibersihkan mahasiswa) ---

# a) ~2% nominal hilang (NaN)
idx_missing_nominal = rng.choice(fact.index, size=int(0.02 * N_TRANSAKSI), replace=False)
fact.loc[idx_missing_nominal, "nominal"] = np.nan

# b) ~0.5% nominal negatif (kesalahan input/kode kanal)
idx_negatif = rng.choice(fact.index, size=int(0.005 * N_TRANSAKSI), replace=False)
fact.loc[idx_negatif, "nominal"] = -fact.loc[idx_negatif, "nominal"]

# c) ~0.3% nominal ekstrem outlier (kesalahan sistem, contoh salah input digit)
idx_outlier = rng.choice(fact.index, size=int(0.003 * N_TRANSAKSI), replace=False)
fact.loc[idx_outlier, "nominal"] = fact.loc[idx_outlier, "nominal"] * rng.integers(500, 2000, size=len(idx_outlier))

# d) ~1% baris duplikat murni (transaction_id sama, tercatat dobel oleh sistem sumber)
duplikat = fact.sample(n=int(0.01 * N_TRANSAKSI), random_state=RNG_SEED).copy()
fact = pd.concat([fact, duplikat], ignore_index=True)

# e) ~1.5% customer_id kosong/tidak valid (gagal match saat integrasi sistem)
idx_cust_invalid = rng.choice(fact.index, size=int(0.015 * len(fact)), replace=False)
fact.loc[idx_cust_invalid, "customer_id"] = rng.choice(["", "UNKNOWN", None], size=len(idx_cust_invalid))

# f) Format tanggal dibuat tidak konsisten pada ~5% baris (string, bukan datetime)
fact["tanggal"] = fact["tanggal"].astype(object)
idx_format_beda = rng.choice(fact.index, size=int(0.05 * len(fact)), replace=False)
fact.loc[idx_format_beda, "tanggal"] = pd.to_datetime(
    fact.loc[idx_format_beda, "tanggal"]
).dt.strftime("%d/%m/%Y")

# g) Kapitalisasi kanal tidak konsisten pada sebagian baris
idx_kanal_kotor = rng.choice(fact.index, size=int(0.04 * len(fact)), replace=False)
fact.loc[idx_kanal_kotor, "kanal"] = fact.loc[idx_kanal_kotor, "kanal"].str.upper()

# Acak ulang urutan baris agar tidak mengelompok
fact = fact.sample(frac=1.0, random_state=RNG_SEED).reset_index(drop=True)

fact.to_csv(OUT_DIR / "fact_transaksi.csv", index=False)
print(f"   -> {len(fact):,} baris fact transaksi (termasuk duplikat) disimpan ke data/fact_transaksi.csv")
print("\nSelesai. Dataset siap digunakan pada notebook praktikum.")
print("Ringkasan masalah data yang disengaja:")
print("  - Nominal hilang (NaN)      : ~2.0%")
print("  - Nominal negatif           : ~0.5%")
print("  - Nominal outlier ekstrem   : ~0.3%")
print("  - Baris duplikat            : ~1.0%")
print("  - customer_id tidak valid   : ~1.5%")
print("  - Format tanggal tercampur  : ~5.0%")
print("  - Kapitalisasi kanal campur : ~4.0%")
