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

---
