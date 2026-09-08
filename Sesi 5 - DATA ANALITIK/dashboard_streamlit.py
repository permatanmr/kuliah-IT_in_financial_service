"""
Dashboard Streamlit — Analitik Data Jasa Keuangan
Praktikum Pertemuan 5: IT in Financial Service
Program Studi Digital Business Technology, School of STEM, Universitas Prasetiya Mulya

Cara menjalankan:
    streamlit run dashboard_streamlit.py

Prasyarat:
    1. Jalankan `python 01_generate_dataset.py` untuk membuat data mentah.
    2. Jalankan seluruh sel di `praktikum_analitik_jasa_keuangan.ipynb` sampai Bagian 2
       selesai, sehingga file `output/fact_transaksi_bersih.parquet` tersedia.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------------
# Konfigurasi halaman
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Analitik Jasa Keuangan",
    layout="wide",
)

WARNA_NAVY = "#00376A"
WARNA_MERAH = "#E30513"
WARNA_BIRU = "#164E91"


@st.cache_data
def muat_data():
    fact = pd.read_parquet("output/fact_transaksi_bersih.parquet")
    fact["tanggal"] = pd.to_datetime(fact["tanggal"])
    dim_nasabah = pd.read_csv("data/dim_nasabah.csv")
    return fact, dim_nasabah


st.title("Dashboard Analitik Data Jasa Keuangan")
st.caption(
    "Praktikum Pertemuan 5 — IT in Financial Service · "
    "Program Studi Digital Business Technology, School of STEM, Universitas Prasetiya Mulya"
)

try:
    fact, dim_nasabah = muat_data()
except FileNotFoundError:
    st.error(
        "Data belum ditemukan. Jalankan `01_generate_dataset.py` lalu eksekusi Bagian 1-2 "
        "pada notebook `praktikum_analitik_jasa_keuangan.ipynb` sebelum membuka dashboard ini."
    )
    st.stop()

nasabah_terkini = dim_nasabah[dim_nasabah["is_current"]]

# ------------------------------------------------------------------
# Filter di sidebar
# ------------------------------------------------------------------
st.sidebar.header("Filter")
segmen_terpilih = st.sidebar.multiselect(
    "Segmen nasabah",
    options=sorted(nasabah_terkini["segmen"].unique()),
    default=sorted(nasabah_terkini["segmen"].unique()),
)
kanal_terpilih = st.sidebar.multiselect(
    "Kanal transaksi",
    options=sorted(fact["kanal"].unique()),
    default=sorted(fact["kanal"].unique()),
)

nasabah_filter = nasabah_terkini[nasabah_terkini["segmen"].isin(segmen_terpilih)]
fact_filter = fact[
    fact["kanal"].isin(kanal_terpilih)
    & fact["customer_id"].isin(nasabah_filter["customer_id"])
]

# ------------------------------------------------------------------
# KPI ringkas
# ------------------------------------------------------------------
kolom1, kolom2, kolom3, kolom4 = st.columns(4)
kolom1.metric("Total Transaksi", f"{len(fact_filter):,}")
kolom2.metric("Total Nominal", f"Rp {fact_filter['nominal'].sum():,.0f}")
kolom3.metric("Nasabah Aktif", f"{fact_filter['customer_id'].nunique():,}")
kolom4.metric(
    "Rata-rata Nominal/Transaksi",
    f"Rp {fact_filter['nominal'].mean():,.0f}" if len(fact_filter) else "-",
)

st.divider()

# ------------------------------------------------------------------
# Grafik 1: Tren nominal transaksi bulanan
# ------------------------------------------------------------------
tren_bulanan = (
    fact_filter.assign(bulan=fact_filter["tanggal"].dt.to_period("M").astype(str))
    .groupby("bulan")["nominal"]
    .sum()
    .reset_index()
)
fig_tren = px.line(
    tren_bulanan, x="bulan", y="nominal",
    title="Tren Total Nominal Transaksi per Bulan",
    labels={"bulan": "Bulan", "nominal": "Total Nominal (Rp)"},
    markers=True,
)
fig_tren.update_traces(line_color=WARNA_NAVY)
st.plotly_chart(fig_tren, use_container_width=True)

kolom_kiri, kolom_kanan = st.columns(2)

# ------------------------------------------------------------------
# Grafik 2: Total nominal per kanal
# ------------------------------------------------------------------
with kolom_kiri:
    per_kanal = (
        fact_filter.groupby("kanal")["nominal"].sum().sort_values().reset_index()
    )
    fig_kanal = px.bar(
        per_kanal, x="nominal", y="kanal", orientation="h",
        title="Total Nominal Transaksi per Kanal",
        labels={"nominal": "Total Nominal (Rp)", "kanal": "Kanal"},
    )
    fig_kanal.update_traces(marker_color=WARNA_MERAH)
    st.plotly_chart(fig_kanal, use_container_width=True)

# ------------------------------------------------------------------
# Grafik 3: Jumlah nasabah aktif per segmen
# ------------------------------------------------------------------
with kolom_kanan:
    aktif_per_segmen = (
        fact_filter.merge(
            nasabah_filter[["customer_id", "segmen"]], on="customer_id", how="left"
        )
        .groupby("segmen")["customer_id"]
        .nunique()
        .sort_values()
        .reset_index(name="jumlah_nasabah")
    )
    fig_segmen = px.bar(
        aktif_per_segmen, x="jumlah_nasabah", y="segmen", orientation="h",
        title="Jumlah Nasabah Aktif per Segmen",
        labels={"jumlah_nasabah": "Jumlah Nasabah", "segmen": "Segmen"},
    )
    fig_segmen.update_traces(marker_color=WARNA_BIRU)
    st.plotly_chart(fig_segmen, use_container_width=True)

st.divider()
st.caption(
    "Sumber data: dataset transaksi sintetis yang dibuat oleh `01_generate_dataset.py` "
    "untuk keperluan praktikum. Bukan data nasabah sungguhan."
)
