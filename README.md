# 🇮🇳 AI Strategy Hub Pro: E-Ticaret Büyüme & Fiyatlandırma Zekası

Bu proje, **büyük e-ticaret veri setlerini** işleyerek stratejik karar destek mekanizmalarına dönüştüren, **Data Engineering ve Advanced Machine Learning** disiplinlerini harmanlayan uçtan uca bir analitik platformdur.

Başlangıçta temel bir analiz aracı olarak tasarlanan sistem; **Multi-Page App (Çok Sayfalı Uygulama)** mimarisine geçirilmiş, **K-Means Kümeleme**, **Gelişmiş Zaman Serisi Ayrışımı** ve **A/B Senaryo Simülatörü** ile donatılarak bir **"CEO Dashboard"** seviyesine taşınmıştır.


## 🏗️ Proje Mimarisi ve Dosya Yapısı

Proje, yönetilebilirliği artırmak ve performansı optimize etmek için modüler bir yapıya sahiptir:

📁 AI_Strategy_Hub_Pro
│
├── 🏠 Home.py                 # Ana Giriş ve Ayar Merkezi (Session State Yönetimi)
├── ⚙️ utils.py                # ETL Motoru (Veri Yükleme, Temizleme, Önişleme)
│
└── 📂 pages
    ├── 1_📊_Veri_Analizi.py        # İleri Seviye EDA & KPI Dashboard
    ├── 2_🤖_AutoML_ve_Kumeleme.py  # Model Eğitimi & Müşteri Segmentasyonu (3D)
    ├── 3_📈_Zaman_Serisi.py        # Trend Analizi & Gelecek Tahmini
    └── 4_🎯_Strateji_Simulatoru.py # A/B Testi & Fiyat Optimizasyonu



# 🚀 Modüller & Analitik Yetenekler

## 1️⃣ 📊 İleri Seviye Veri Analizi (Advanced EDA & BI)

Ham veriyi, **doğrudan iş kararına dönüşebilecek stratejik içgörülere** çeviren gelişmiş BI katmanıdır.

### Sunulan Analitikler

### 🔹 Pareto Analizi (80/20 Kuralı)
- Gelirin %80’ini oluşturan:
  - Kritik ürün grupları  
  - En değerli müşteri segmentleri  
- Gelir yoğunluk optimizasyonu

### 🔹 Coğrafi Pazar Analizi
Bölge & eyalet bazlı:
- Ciro
- Sipariş yoğunluğu
- Pazar payı

### 🔹 Korelasyon Isı Haritası (Heatmap)
- Fiyat – talep – satış – kâr ilişkilerinin:
  - Gizli bağıntıları
  - Nedensel sinyalleri

### 🔹 Outlier Detection (Aykırı Değer Analizi)
IQR tabanlı anomali tespiti ile:
- Hatalı veri girişleri
- Fraud sinyalleri
- Operasyonel risk alanları

---

## 2️⃣ 🤖 AutoML & 3D K-Means Segmentasyon Laboratuvarı

Makine öğrenmesini demokratikleştiren ve **karar destek altyapısının çekirdeğini oluşturan AI motoru**.

### 🧠 AutoML Yarış Motoru

Aynı anda eğitilen modeller:
- Linear Regression  
- Random Forest  
- XGBoost  
- LightGBM  

### 🔎 Otomatik Model Seçimi

Performans kriterleri:
- R²
- RMSE
- MAE

En iyi modeli **tam otomatik olarak seçer.**

### 📈 Model Analitiği

- Gerçek vs Tahmin grafikleri
- Residual dağılım analizi
- Hata yoğunluk haritaları

---

### 🎯 Unsupervised Learning: 3D K-Means Müşteri Segmentasyonu

Müşterileri şu değişkenlere göre **davranışsal kümelere ayırır:**

- Satın alma davranışı  
- Sepet büyüklüğü  
- Harcama frekansı  

### Kullanılan Teknikler

- **Elbow Method** → Optimum K seçimi  
- **3D Scatter Plot** → Segmentlerin uzaysal görselleştirilmesi  

> Bu yapı, doğrudan **kişiselleştirilmiş pazarlama, kampanya hedefleme ve churn analizi** altyapısı sağlar.

---

## 3️⃣ 📈 Zaman Serisi & Forecasting Motoru

Sadece tahmin yapmaz, **verinin zaman içindeki hikayesini çözer.**

### 🔄 Çift Motorlu Tahmin Sistemi

- **Prophet (Meta AI)**
  - Tatil etkileri
  - Karmaşık trend
  - Düzensiz periyotlar

- **Holt-Winters (Exponential Smoothing)**
  - Hızlı
  - Stabil
  - İstatistiksel tahminleme

### 🧩 Zaman Serisi Ayrıştırma

- Trend  
- Seasonality  
- Residual  

### 📊 Güven Aralığı (Confidence Interval)

Tahminlerin:
- Alt sınır
- Üst sınır

değerlerini göstererek **risk kontrollü planlama** sağlar.

---

## 4️⃣ 🎯 Next-Gen Strateji Simülatörü (A/B Testing & Pricing AI)

Platformun **karar destek zirvesi**.

### 🧪 A/B Senaryo Simülasyonu

Mevcut durum ile hedeflenen senaryo:

- Ciro  
- Kâr  
- Sipariş  
- Marj  

bazında **yan yana karşılaştırılır.**

### 📊 Görsel Karar Destek Araçları

- **Gauge Charts** → KPI hedef yakınlığı  
- **Delta Metrikleri** → Yüzdesel değişim & etki analizi  

### 🤖 AI Fiyat Optimizasyon Eğrisi

Makine öğrenmesi destekli algoritma:

> **"Maksimum kâr için ideal fiyat noktası nedir?"**

sorusuna **otomatik zirve tespiti** yaparak cevap verir.

---

# 🛠 Teknik Altyapı (Tech Stack)

| Katman | Teknolojiler |
|---------|----------------|
| Framework | Streamlit (Multi-Page SaaS Architecture) |
| Data Engineering | Pandas, NumPy, Scikit-Learn |
| Machine Learning | XGBoost, LightGBM, Random Forest, K-Means |
| Time Series | Prophet, Statsmodels (Holt-Winters) |
| Visualization | Plotly Express & Graph Objects |
| UI/UX | Custom CSS, Metric Cards, Modern Dashboard UI |

stremlit uygulaması:https://an-open-source-real-time-financial-lakehouse-project-yjuh6geau.streamlit.app
<img width="2379" height="1167" alt="Ekran görüntüsü 2026-02-18 181546" src="https://github.com/user-attachments/assets/755fde05-40a0-4924-8a83-92067b0a6616" />
<img width="1192" height="796" alt="Ekran görüntüsü 2026-02-18 181824" src="https://github.com/user-attachments/assets/27c5c47f-eef3-4241-89db-17072ea4d3e0" />
<img width="2322" height="1305" alt="Ekran görüntüsü 2026-02-18 181813" src="https://github.com/user-attachments/assets/5a63f314-89d5-4b7d-bca3-73db956b4a56" /><img width="2366" height="1439" alt="Ekran görüntüsü 2026-02-18 181603" src="https://github.com/user-attachments/assets/231ba4d6-9898-4e34-907e-5b26d4d04b5c" />
<img width="2352" height="1106" alt="Ekran görüntüsü 2026-02-18 181617" src="https://github.com/user-attachments/assets/178389e0-1468-44ad-9c48-87b0c0a38ccb" />
<img width="2333" height="1051" alt="Ekran görüntüsü 2026-02-18 181628" src="https://github.com/user-attachments/assets/fd1f7848-22c7-49f0-8ce3-2d97fd4e3661" />
<img width="2330" height="1376" alt="Ekran görüntüsü 2026-02-18 181638" src="https://github.com/user-attachments/assets/ede066fa-c0b9-48f6-860c-d2de6d8a7f3f" />
<img width="149" height="80" alt="Ekran görüntüsü 2026-02-18 181755" src="https://github.com/user-attachments/assets/bcf3a217-702e-4e36-b6f2-e03f7354d37f" />
<img width="2365" height="1173" alt="Ekran görüntüsü 2026-02-18 181526" src="https://github.com/user-attachments/assets/10060644-71ee-4e92-86fe-00c9d9a0809e" />
<img width="2362" height="1199" alt="Ekran görüntüsü 2026-02-18 181538" src="https://github.com/user-attachments/assets/9b3d5719-c2ee-479c-9bf4-7b0f4db4f848" />
<img width="2314" height="1319" alt="Ekran görüntüsü 2026-02-18 181507" src="https://github.com/user-attachments/assets/5d7e3647-e437-4558-9cce-9f1dcec8388b" />
<img width="2342" height="1210" alt="Ekran görüntüsü 2026-02-18 181517" src="https://github.com/user-attachments/assets/a2bc6b42-bb54-45e3-b7b0-2222fb4c0519" />
<img width="2416" height="1395" alt="Ekran görüntüsü 2026-02-18 181851" src="https://github.com/user-attachments/assets/e4484330-15a3-41c1-895f-0a9df7c3e819" />
<img width="2427" height="1425" alt="Ekran görüntüsü 2026-02-18 181924" src="https://github.com/user-attachments/assets/9c7c67d2-7d23-4388-afc9-738ea2f3dae1" />


https://an-open-source-real-time-financial-lakehouse-project-yjuh6geau.streamlit.app/
---
