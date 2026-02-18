Bu proje; **perakende sektöründe büyük veri (Big Data) setlerini stratejik karar destek mekanizmalarına dönüştürmek** amacıyla tasarlanmış, **Data Engineering ve Machine Learning disiplinlerini harmanlayan ileri seviye bir analitik platformdur**.

Ramazan serisinin **2. haftasında**, ilk projenin üzerine **Explainable AI (XAI)** ve **Time Series Forecasting** katmanları eklenerek geliştirilmiştir.

---

# 🏗️ Sistem Mimarisi ve Pipeline Akışı

Platform, ham veriyi alıp **aksiyon alınabilir stratejik içgörülere** dönüştürürken aşağıdaki katmanlardan geçer:

---

## 1️⃣ Veri Mühendisliği & Preprocessing (Data Engine)

### 🎯 Target Leakage Guard
Modelin gelecekteki bilgileri (**Revenue, Units Sold, Final Price**) kullanarak *hile yapmasını* engellemek için **dinamik feature exclusion mekanizması** kurulmuştur.

### 🔡 Label Encoding & Optimization
Kategorik değişkenler (**Brand Type, Zone, Category**) modelin **hiyerarşik ilişkileri anlayabileceği** şekilde encode edilirken **bellek ve hız optimizasyonu** sağlanmıştır.

### 🧹 Statik & Dinamik Temizlik
- Eksik değerler (**NaN**)  
- Aykırı değerler (**Outliers**)  

**IQR yöntemi** ile analiz edilerek temizlenmiştir.

---

## 2️⃣ AutoML & Model Yarışması (Model Laboratory)

Proje, tek bir algoritmaya güvenmek yerine **Model Yarışması (Leaderboard)** mimarisi ile çalışır.

### 🔬 Kullanılan Algoritmalar
- XGBoost  
- LightGBM  
- Random Forest  
- Voting Ensemble (Hibrit Model)

### 📏 Değerlendirme Metrikleri
- **R²** → Model başarısı  
- **MAE** → Ortalama mutlak hata  
- **RMSE** → Hata kareler ortalaması  

### 🏆 Champion Model Selection
Sistem, **en yüksek R² değerine sahip modeli otomatik olarak "Şampiyon Model"** ilan eder ve **tüm dashboard analizleri bu beyin üzerine inşa edilir**.

---

## 3️⃣ Explainable AI (XAI) & Karar Şeffaflığı

Yapay zekanın **neden bu sonucu verdiğini açıklamak** için iki güçlü metodoloji entegre edilmiştir:

### 🌍 Global Explanation — SHAP
Modelin genel stratejisinde **hangi değişkenlerin daha etkili olduğunu** gösterir.  
*(Örn: İndirim oranı, Marka tipi, Bölge, Kampanya etkisi)*

### 🔍 Local Explanation — LIME
Tekil işlem ve ürün bazında, **modelin verdiği kararın gerekçelerini** açıklar.

---

## 4️⃣ Zaman Serisi & Forecasting Katmanı

Geçmiş **36 aylık veri** kullanılarak geleceğe yönelik projeksiyonlar üretilir:

### 🤖 AI Tabanlı Tahmin
- **Facebook Prophet**
- Mevsimsellik + Tatil + Kampanya etkisi

### 📈 İstatistiksel Tahmin
- **Holt-Winters (Exponential Smoothing)**
- Trend + Sezonsallık analizi

---

# 🎯 Dinamik Persona Simülatörü (What-If Analysis)

Bu modül, stratejik karar senaryolarının **gerçek zamanlı simülasyonunu** sağlar.

> "Fiyatı %10 artırırsak ne olur?"  
> "Bu markada indirimi %5 düşürürsek satışlar nasıl etkilenir?"

### ⚡ Anlık Tahmin Motoru
- Girilen parametrelere göre **saniyeler içinde fiyat & talep tahmini**

### 📉 Sensitivity Curve (Hassasiyet Eğrisi)
- Fiyat değişimlerinin hedef metrikler üzerindeki **elastikiyetini** gösterir.

---

# 📊 Gelişmiş Görselleştirme Katmanı

### 🌍 Hierarchical Treemap
- Bölge → Eyalet → Kategori bazlı **ciro akış analizi**

### 🔵 Bubble & Hexbin Maps
- Fiyat, indirim ve talep arasındaki **çok boyutlu korelasyon görselleştirmesi**

### 🧪 Residual Analysis
- Model hata dağılımının **normal dağılıma uygunluğunu test eden regresyon diyagramları**

---

## 🛠 Kullanılan Teknolojiler

| Katman | Teknolojiler |
|-----------|----------------|
| Arayüz & UI | Streamlit, Custom CSS |
| Veri İşleme | Pandas, NumPy, Scikit-Learn |
| Görselleştirme | Plotly (Interactive), Seaborn, Matplotlib |
| Yapay Zekâ & Forecast | XGBoost, LightGBM, Prophet, LIME, SHAP |
| İstatistik | Statsmodels (Holt-Winters), SciPy |

---
## 📌 Not

Bu platform, **yüksek hacimli ticari veriler üzerinde gerçek zamanlı analiz, karar destek ve fiyat optimizasyonu sağlamak amacıyla geliştirilmiş ileri düzey bir yapay zekâ tabanlı analitik sistemdir.**
# 🚀 Kurulum & Çalıştırma

```bash
# Bağımlılıkları yükle
pip install -r requirements.txt

# Uygulamayı çalıştır
streamlit run indian.py
