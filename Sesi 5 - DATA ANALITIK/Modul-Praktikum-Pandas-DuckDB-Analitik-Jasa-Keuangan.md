# Modul Praktikum: Analitik Data Jasa Keuangan dengan pandas dan DuckDB

**IT in Financial Service — Sesi 5: Arsitektur Data dan Analitik Jasa Keuangan**
Program Studi Digital Business Technology, School of STEM, Universitas Prasetiya Mulya

---

## Tujuan

Setelah menyelesaikan praktikum ini, Anda mampu:

1. Membersihkan dataset transaksi sintetis berskala besar (1 juta baris) yang mengandung duplikasi, nilai kosong, format tanggal tidak konsisten, dan outlier.
2. Membangun **agregasi kohort nasabah** dan menghitung metrik retensi bulanan.
3. Menjalankan kueri analitik dengan **pandas** dan **DuckDB**, lalu membandingkan performanya secara jujur pada kueri sederhana maupun kueri gabungan (join) yang berat.
4. Menghitung proksi metrik bisnis jasa keuangan — **CASA ratio, NPL/TWP90, CAC, LTV, cost-to-income ratio** — sekaligus memahami keterbatasan proksi tersebut dibanding star schema yang lengkap.
5. Membangun **dashboard sederhana** dengan Plotly (di dalam notebook) dan Streamlit (aplikasi mandiri).

Praktikum ini mengikuti materi Pertemuan 5 pada RPS mata kuliah IT in Financial Service dan menjadi dasar penilaian **notebook analitik + interpretasi bisnis (bobot 5% dari nilai akhir)**.

> Catatan: seluruh data pada praktikum ini **sintetis (dibuat secara acak)** — bukan data nasabah sungguhan. Metrik bisnis seperti CASA ratio dan CAC/LTV dihitung sebagai **proksi ilustratif** karena dataset transaksi saja tidak memuat seluruh informasi yang dibutuhkan (misalnya saldo dana pihak ketiga, biaya marketing riil per nasabah). Bagian 5 modul ini menjelaskan proksi apa yang dipakai dan tabel apa yang dibutuhkan pada star schema nyata agar metrik tersebut bisa dihitung tepat.

---

## Daftar Isi

1. Prasyarat
2. Struktur Berkas Modul
3. Menyiapkan Lingkungan Python
4. Menghasilkan Dataset Sintetis
5. Menjalankan Notebook Analitik
6. Ringkasan Isi Notebook per Bagian
7. Menjalankan Dashboard Streamlit
8. Materi Pendukung: Data Warehouse, Star Schema, dan Regulasi
9. Struktur Laporan Praktikum dan Rubrik Penilaian
10. Referensi

---

## 1. Prasyarat

| Kebutuhan | Versi minimum | Cek dengan |
|---|---|---|
| Python | 3.10 atau lebih baru | `python --version` |
| pip | bawaan Python | `pip --version` |
| Jupyter Notebook / JupyterLab | terbaru | `jupyter --version` |
| Ruang disk kosong | ± 200 MB | untuk dataset + notebook + parquet |
| Editor kode | VS Code / JupyterLab disarankan | — |

Pengetahuan dasar Python (fungsi, list, dictionary) dan konsep dasar SQL (`SELECT`, `GROUP BY`, `JOIN`) diasumsikan sudah dikuasai dari sesi-sesi sebelumnya.

---

## 2. Struktur Berkas Modul

```
praktikum-data-analitik/
├── 01_generate_dataset.py                       # skrip pembuat dataset sintetis
├── build_notebook.py                            # skrip pembangun notebook (untuk dosen/pengembang)
├── praktikum_analitik_jasa_keuangan.ipynb        # notebook praktikum utama — mahasiswa mengisi ini
├── dashboard_streamlit.py                        # aplikasi dashboard mandiri (Streamlit)
├── Modul-Praktikum-Pandas-DuckDB-Analitik-Jasa-Keuangan.md   # dokumen panduan ini
├── data/
│   ├── dim_nasabah.csv                           # dimensi nasabah (dengan riwayat SCD Type-2)
│   └── fact_transaksi.csv                        # tabel fakta transaksi (1 juta+ baris, sengaja "kotor")
└── output/
    └── fact_transaksi_bersih.parquet             # dihasilkan otomatis oleh notebook (Bagian 2)
```

Mahasiswa hanya perlu berinteraksi dengan `01_generate_dataset.py`, `praktikum_analitik_jasa_keuangan.ipynb`, dan `dashboard_streamlit.py`. Berkas `build_notebook.py` disediakan untuk dosen apabila ingin mengubah struktur notebook untuk sesi berikutnya.

---

## 3. Menyiapkan Lingkungan Python

Buat virtual environment (disarankan, opsional) dan instal seluruh dependensi:

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install pandas duckdb jupyter nbformat plotly streamlit faker pyarrow matplotlib openpyxl
```

Jalankan Jupyter untuk memastikan instalasi berhasil:

```bash
jupyter notebook
```

Browser akan terbuka menampilkan file explorer Jupyter — pastikan tidak ada galat pada terminal.

---

## 4. Menghasilkan Dataset Sintetis

Dataset **tidak disertakan langsung** dalam bentuk jadi (ukurannya besar) — jalankan skrip generator terlebih dahulu:

```bash
cd praktikum-data-analitik
python 01_generate_dataset.py
```

Skrip ini akan membuat dua berkas di folder `data/`:

| Berkas | Baris | Isi |
|---|---|---|
| `dim_nasabah.csv` | ± 23.000 | Dimensi nasabah: `customer_id`, `nama`, `segmen`, `cabang`, `status_kyc`, `valid_from`, `valid_to`, `is_current` — sekitar 15% nasabah memiliki dua baris riwayat (contoh sederhana **SCD Type-2**, yaitu teknik menyimpan riwayat perubahan atribut dimensi dari waktu ke waktu) |
| `fact_transaksi.csv` | 1.010.000 | Tabel fakta transaksi: `transaction_id`, `customer_id`, `tanggal`, `jenis_transaksi`, `kanal`, `nominal` |

**Dataset transaksi dibuat sengaja "kotor"** agar praktikum pembersihan data terasa realistis:

- ± 1% baris duplikat (`transaction_id` sama)
- ± 2% nilai `nominal` kosong
- ± 0,5% nilai `nominal` negatif (harus dijadikan absolut, bukan dihapus)
- ± 0,3% outlier ekstrem pada `nominal`
- ± 1,5% `customer_id` tidak valid/kosong
- ± 5% format tanggal tidak konsisten (campuran ISO `2024-03-15` dan `15/03/2024`)
- ± 4% variasi kapitalisasi pada kolom `kanal` (contoh: `Mobile Banking` vs `MOBILE BANKING`)

Proses ini memakan waktu sekitar 10-20 detik. Setelah selesai, Anda akan melihat ringkasan jumlah baris pada terminal.

---

## 5. Menjalankan Notebook Analitik

Buka `praktikum_analitik_jasa_keuangan.ipynb` di Jupyter/JupyterLab, lalu jalankan sel **secara berurutan dari atas ke bawah** (jangan melompat) — banyak sel bergantung pada variabel dari sel sebelumnya.

Notebook terbagi menjadi 6 bagian utama plus sampul dan penutup. Setiap bagian diakhiri **pertanyaan refleksi** yang wajib diisi mahasiswa dengan interpretasi bisnis, bukan hanya jawaban teknis.

---

## 6. Ringkasan Isi Notebook per Bagian

### Bagian 1 — Memuat dan Memeriksa Data Mentah

Memuat `fact_transaksi.csv` dan `dim_nasabah.csv`, memeriksa tipe data, jumlah nilai kosong per kolom, dan jumlah baris duplikat. Tujuannya membangun kebiasaan **selalu profiling data sebelum membersihkannya** — jangan langsung mengasumsikan data sudah rapi.

### Bagian 2 — Pembersihan Data

Langkah pembersihan dilakukan bertahap dengan log setiap tahap:

1. Menghapus baris duplikat berdasarkan `transaction_id`.
2. Mem-parsing kolom tanggal yang formatnya campuran (ISO dan `dd/mm/yyyy`) menjadi satu tipe `datetime` konsisten.
3. Menstandardisasi kapitalisasi kolom `kanal` (misalnya `MOBILE BANKING` → `Mobile Banking`).
4. Membuang transaksi dengan `customer_id` tidak valid/kosong — beserta penjelasan mengapa baris ini **dibuang, bukan diisi** (tidak bisa dikaitkan ke nasabah mana pun, sehingga tidak berguna untuk agregasi kohort per nasabah).
5. Membersihkan kolom `nominal`: baris dengan nilai kosong dibuang, nilai negatif dijadikan absolut (dianggap kesalahan input, bukan retur), dan outlier ekstrem di-*cap* pada persentil ke-99,5 (bukan dibuang, agar tidak menghilangkan transaksi besar yang sah).
6. Menyimpan hasil akhir sebagai `output/fact_transaksi_bersih.parquet` — format Parquet dipilih karena lebih ringkas dan lebih cepat dibaca ulang dibanding CSV, terutama untuk beban kerja analitik kolom demi kolom.

**Pertanyaan Refleksi 2.1** meminta mahasiswa menjelaskan trade-off dari keputusan *cap* vs *drop* pada outlier — jawaban yang baik akan menyinggung dampaknya pada perhitungan agregat (rata-rata, total) versus risiko menghilangkan sinyal transaksi mencurigakan yang justru penting untuk deteksi fraud pada sesi berikutnya.

### Bagian 3 — DuckDB vs pandas: Kueri Analitik pada Skala Besar

Bagian ini secara sengaja dirancang untuk **tidak selalu memenangkan DuckDB** — ini poin pedagogis penting. Pada kueri `GROUP BY` satu kolom yang sederhana dan datanya sudah di memori, pandas sering **sama cepat atau lebih cepat**, karena DuckDB memiliki overhead saat memindai ulang DataFrame pandas setiap kali dipanggil. Keunggulan DuckDB terlihat jelas pada kueri yang lebih berat: join tiga arah antara tabel fakta dan dimensi ditambah agregasi berlapis (segmen × bulan × kanal), yang mendekati kompleksitas kueri laporan regulasi sungguhan.

**Pertanyaan Refleksi 3.1** meminta mahasiswa membandingkan selisih waktu pada kueri sederhana vs kueri berat, dan menjelaskan implikasinya: kapan sebaiknya tim data memilih pandas (eksplorasi cepat, dataset kecil-menengah, tim yang lebih nyaman dengan Python) versus DuckDB/SQL (kueri kompleks berulang, dataset besar, atau integrasi dengan data warehouse yang sudah berbasis SQL).

### Bagian 4 — Agregasi Kohort Nasabah dan Metrik Retensi

Setiap nasabah dikelompokkan ke dalam **kohort akuisisi** berdasarkan bulan transaksi pertamanya yang tercatat pada data bersih, lalu dilacak berapa persen dari kohort tersebut masih aktif bertransaksi pada bulan ke-1, ke-2, dst. Hasilnya divisualisasikan sebagai **heatmap retensi** menggunakan Plotly.

Karena dataset mencakup ± 3,5 tahun transaksi, kohort di bulan-bulan awal jauh lebih besar daripada kohort belakangan (pola umum yang disebut *birthday problem* dalam analisis kohort) — notebook menjelaskan mengapa ini bukan kesalahan, dan memfokuskan heatmap pada kohort yang cukup besar (≥ 50 nasabah baru) agar persentase retensi tidak menyesatkan akibat pembagi yang terlalu kecil.

**Pertanyaan Refleksi 4.1** meminta interpretasi bisnis dari pola retensi yang terlihat — misalnya apakah ada penurunan tajam pada bulan tertentu yang mengindikasikan masalah onboarding atau produk.

### Bagian 5 — Metrik Bisnis Jasa Keuangan (Proksi)

Bagian ini menghitung lima metrik kunci yang disebutkan pada materi RPS, dengan **asumsi yang dinyatakan secara eksplisit** pada setiap sel kode:

| Metrik | Formula Asli | Proksi yang Dipakai di Notebook |
|---|---|---|
| CASA ratio | (Giro + Tabungan) / Total Dana Pihak Ketiga | Rasio nominal transaksi bertipe "setoran/tabungan" terhadap total volume transaksi (bukan saldo riil) |
| NPL / TWP90 | Kredit macet > 90 hari / Total kredit disalurkan | Disimulasikan secara acak per nasabah karena dataset tidak memuat data pinjaman |
| CAC (Customer Acquisition Cost) | Total Biaya Akuisisi / Jumlah Nasabah Baru | Asumsi total biaya marketing dibagi jumlah total nasabah unik pada data |
| LTV (Customer Lifetime Value) | Margin per periode × Estimasi masa hidup nasabah | Margin bulanan (asumsi 3% dari volume transaksi) × asumsi masa aktif 24 bulan |
| Cost-to-income ratio | Beban Operasional / Pendapatan Operasional | Beban = biaya akuisisi + biaya operasional tetap asumsi; Pendapatan = margin asumsi × total volume transaksi |

Setelah dihitung, rasio LTV:CAC dan cost-to-income ratio dibandingkan terhadap acuan umum industri (LTV:CAC sehat di atas 3:1; cost-to-income efisien umumnya di bawah 50-60%) sehingga mahasiswa bisa langsung menilai apakah hasil proksi berada pada kisaran yang wajar.

**Pertanyaan Refleksi 5.1** — pertanyaan paling penting pada modul ini — meminta mahasiswa menyebutkan **tabel dan kolom tambahan** apa yang dibutuhkan pada star schema nyata agar kelima metrik ini bisa dihitung tanpa proksi (misalnya tabel saldo harian per nasabah untuk CASA ratio, tabel fasilitas kredit dan status tunggakan untuk NPL/TWP90, tabel biaya marketing per kampanye/kanal akuisisi untuk CAC). Jawaban ini menghubungkan praktikum kembali ke materi pemodelan data (star schema, SCD Type-2) pada RPS.

### Bagian 6 — Dashboard Sederhana (Plotly, di dalam notebook)

Tiga visualisasi ringkas ditampilkan langsung di notebook menggunakan Plotly:

1. Tren total nominal transaksi per bulan (line chart)
2. Total nominal transaksi per kanal (bar chart horizontal)
3. Jumlah nasabah aktif per segmen (bar chart horizontal)

Visualisasi ini adalah versi ringkas dari dashboard mandiri pada Bagian 7 di bawah.

### Penutup

Instruksi deliverable akhir: melengkapi kelima sel refleksi, menambahkan interpretasi bisnis 1 halaman, serta menjalankan dan menyertakan tangkapan layar dashboard Streamlit.

---

## 7. Menjalankan Dashboard Streamlit

Setelah notebook dijalankan minimal sampai Bagian 2 selesai (agar `output/fact_transaksi_bersih.parquet` tersedia), jalankan:

```bash
streamlit run dashboard_streamlit.py
```

Browser akan terbuka menampilkan dashboard interaktif dengan:

- **Filter** segmen nasabah dan kanal transaksi di sidebar
- **KPI ringkas**: total transaksi, total nominal, jumlah nasabah aktif, rata-rata nominal per transaksi
- **Tiga grafik** yang sama dengan Bagian 6 notebook, namun interaktif dan dapat difilter secara langsung

Ambil tangkapan layar dashboard (dengan minimal satu filter diterapkan) untuk dilampirkan pada laporan praktikum.

---

## 8. Materi Pendukung: Data Warehouse, Star Schema, dan Regulasi

Praktikum ini berkaitan langsung dengan materi konseptual Pertemuan 5 pada RPS:

- **Data warehouse vs data lake vs lakehouse** — data warehouse menyimpan data terstruktur yang sudah dimodelkan untuk kueri analitik (seperti `fact_transaksi.csv` dan `dim_nasabah.csv` pada praktikum ini); data lake menyimpan data mentah dalam berbagai format; data lakehouse menggabungkan fleksibilitas data lake dengan kemampuan manajemen data warehouse ([AWS](https://aws.amazon.com/what-is/data-lakehouse/), [IBM](https://www.ibm.com/think/topics/data-warehouse-vs-data-lake-vs-data-lakehouse)).
- **Star schema dan SCD Type-2** — `fact_transaksi.csv` berperan sebagai tabel fakta dan `dim_nasabah.csv` sebagai tabel dimensi, mengikuti pola star schema standar pada data warehouse jasa keuangan. Kolom `valid_from`, `valid_to`, dan `is_current` pada `dim_nasabah.csv` adalah implementasi sederhana **SCD Type-2** — teknik menyimpan riwayat perubahan atribut nasabah (misalnya perubahan segmen) tanpa menimpa data lama.
- **Data governance** — konsep *data lineage* (jejak asal-usul data), *data quality* (yang dipraktikkan langsung pada Bagian 2), *golden record* nasabah, dan *single customer view* menjadi relevan ketika satu nasabah memiliki data yang tersebar di banyak sistem sumber.
- **Regulatory reporting** — pelaporan penyelenggara Inovasi Teknologi Sektor Keuangan (ITSK) di Indonesia diatur melalui **SEOJK 4/SEOJK.07/2025** ([OJK](https://ojk.go.id/id/regulasi/Pages/SEOJK-4SEOJK072025-Pelaporan-Penyelenggara-Inovasi-Teknologi-Sektor-Keuangan-Yang-Memiliki-Izin-Usaha-di-OJK.aspx), [SSEK Law Firm](https://ssek.com/blog/reporting-timelines-for-indonesias-fintech-innovators-new-deadlines-under-ojk-circular-letter-4-2025/?lang=id)) — pipeline analitik yang rapi seperti pada praktikum ini adalah prasyarat teknis untuk memenuhi kewajiban pelaporan semacam ini secara otomatis dan tepat waktu.
- **Metrik kunci jasa keuangan** — definisi CASA ratio dikonfirmasi dari [Wikipedia](https://en.wikipedia.org/wiki/CASA_ratio): CASA Ratio = (Giro + Tabungan) / Total Dana Pihak Ketiga × 100. Definisi TWP90 (tingkat wanprestasi 90 hari pada pinjaman fintech P2P) mengacu pada [siaran pers OJK](https://ojk.go.id/id/berita-dan-kegiatan/siaran-pers/Pages/Pembiayaan-UMKM-Lewat-Pinjaman-Online-terus-Berkembang,-Pinjaman-Masyarakat-masih-Terkendali.aspx).

---

## 9. Struktur Laporan Praktikum dan Rubrik Penilaian

Sesuai RPS, asesmen Pertemuan 5 berupa **notebook analitik + interpretasi bisnis (bobot 5% dari nilai akhir)**. Laporan yang dikumpulkan minimal memuat:

1. Notebook `.ipynb` yang telah dijalankan penuh (semua sel bebas galat) dengan **kelima pertanyaan refleksi terisi**.
2. Interpretasi bisnis 1 halaman yang menjawab: metrik mana yang paling mengkhawatirkan dari hasil analisis, dan rekomendasi tindak lanjut apa yang akan disampaikan ke manajemen.
3. Tangkapan layar dashboard Streamlit dengan minimal satu filter diterapkan.

| Komponen | Bobot dalam Asesmen |
|---|---|
| Kebenaran teknis pembersihan data (Bagian 2) | 25% |
| Ketepatan agregasi kohort dan interpretasi retensi (Bagian 4) | 25% |
| Kedalaman analisis perbandingan pandas vs DuckDB (Bagian 3) | 15% |
| Pemahaman keterbatasan proksi metrik bisnis (Bagian 5, termasuk Refleksi 5.1) | 20% |
| Kualitas dashboard dan interpretasi bisnis 1 halaman | 15% |

---

## 10. Referensi

- Amazon Web Services. "What is a Data Lakehouse?" [aws.amazon.com/what-is/data-lakehouse](https://aws.amazon.com/what-is/data-lakehouse/)
- IBM. "Data Warehouse vs Data Lake vs Data Lakehouse." [ibm.com/think/topics/data-warehouse-vs-data-lake-vs-data-lakehouse](https://www.ibm.com/think/topics/data-warehouse-vs-data-lake-vs-data-lakehouse)
- Otoritas Jasa Keuangan. "SEOJK 4/SEOJK.07/2025 — Pelaporan Penyelenggara Inovasi Teknologi Sektor Keuangan yang Memiliki Izin Usaha di OJK." [ojk.go.id](https://ojk.go.id/id/regulasi/Pages/SEOJK-4SEOJK072025-Pelaporan-Penyelenggara-Inovasi-Teknologi-Sektor-Keuangan-Yang-Memiliki-Izin-Usaha-di-OJK.aspx)
- SSEK Law Firm. "Reporting Timelines for Indonesia's Fintech Innovators: New Deadlines Under OJK Circular Letter 4/2025." [ssek.com](https://ssek.com/blog/reporting-timelines-for-indonesias-fintech-innovators-new-deadlines-under-ojk-circular-letter-4-2025/?lang=id)
- Otoritas Jasa Keuangan. Siaran Pers — "Pembiayaan UMKM Lewat Pinjaman Online terus Berkembang, Pinjaman Masyarakat masih Terkendali." [ojk.go.id](https://ojk.go.id/id/berita-dan-kegiatan/siaran-pers/Pages/Pembiayaan-UMKM-Lewat-Pinjaman-Online-terus-Berkembang,-Pinjaman-Masyarakat-masih-Terkendali.aspx)
- Wikipedia. "CASA Ratio." [en.wikipedia.org/wiki/CASA_ratio](https://en.wikipedia.org/wiki/CASA_ratio)
- pandas Development Team. "pandas documentation." [pandas.pydata.org/docs](https://pandas.pydata.org/docs/)
- DuckDB Foundation. "DuckDB Documentation." [duckdb.org/docs](https://duckdb.org/docs/)
- Plotly Technologies. "Plotly Python Graphing Library." [plotly.com/python](https://plotly.com/python/)
- Streamlit Inc. "Streamlit Documentation." [docs.streamlit.io](https://docs.streamlit.io/)
