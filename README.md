# 🧠 AI Retail Strategy and Pricing Intelligence Platform

Bu platform, **perakende sektöründe veriye dayalı stratejik karar alma süreçlerini optimize etmek** ve **fiyatlandırma zekâsı oluşturmak** amacıyla geliştirilmiş **uçtan uca bir analitik çözümdür**. Karar mekanizmalarını yalnızca tahminlerle değil, **Açıklanabilir Yapay Zekâ (XAI)** ve **dinamik senaryo simülasyonları** ile destekler.

---

## 🚀 Proje Özellikleri

### 🔄 Dinamik Veri Ön İşleme
- Gerçek zamanlı veri temizleme  
- Kategorik dönüşüm  
- Veri sızıntısı (leakage) korumalı pipeline mimarisi  

### 🤖 AutoML Mimarisi
- 6 yüksek performanslı algoritmanın otomatik yarışması:
  - Linear Regression  
  - Decision Tree  
  - Random Forest  
  - XGBoost  
  - LightGBM  
  - Ensemble (Voting)

- Değerlendirme metrikleri:
  - **R²**
  - **MAE**
  - **RMSE**

### 🔍 Explainable AI (XAI)
- **LIME** ile lokal açıklamalar  
- **SHAP** ile global açıklamalar  
- Model kararlarının hem tekil hem de stratejik seviyede yorumlanabilirliği

### 📊 Zenginleştirilmiş Keşifsel Veri Analizi (EDA)
- 15+ interaktif görselleştirme:
  - Zaman serisi trend analizi  
  - Pazar payı dağılımı  
  - Fiyat elastikiyeti  
  - Hiyerarşik veri haritaları  

### 🧑‍💼 Persona Simülatörü
- Müşteri profili + ürün segmenti bazlı:
  - Fiyat tahmini  
  - Dinamik hassasiyet analizi  

---

## 🏗️ Veri Ön İşleme ve Pipeline Mimarisi

### 🧹 Veri Temizleme
- Eksik değerler (NaN), istatistiksel sapmaları önlemek adına **satır bazlı** olarak temizlenmiştir.

### 🔡 Kategorik Kodlama
- `State`, `Zone`, `Category`, `Brand Type` gibi değişkenler **Label Encoding** yöntemi ile dönüştürülmüştür.

### 🛡 Leakage Guard (Sızıntı Korumasi)
Modelin **ezber yapmasını (overfitting)** önlemek için, tahmin anında bilinemeyecek aşağıdaki değişkenler otomatik olarak özellik setinden çıkarılır:

- Revenue  
- Units Sold  
- Final Price  

### 💾 Feature Persistence
- **Session State yönetimi** sayesinde:
  - Kullanıcının hariç tuttuğu özellikler
  - Model parametreleri
  - Pipeline ayarları  
uygulama boyunca korunur.

---

## 🏆 AutoML Model Havuzu ve Şampiyon Model Seçimi

Platform, **en yüksek R² skoruna sahip modeli otomatik olarak “Şampiyon Model”** ilan eder ve tüm analizleri bu model üzerinden yürütür.

| Tip | Yöntem | Avantaj |
|------|----------|------------|
| XGBoost / LightGBM | Boosting | Yüksek tahmin gücü & hız |
| Random Forest | Bagging | Düşük varyans & stabilite |
| Voting Ensemble | Hybrid | Modellerin kolektif zekâsı |
| Linear Regression | İstatistiksel | Katsayı yorumlanabilirliği |

---

## 📈 Analitik Kapasite ve Karar Destek Mekanizmaları

### ⏳ Zaman Serisi Analizi
- **Facebook Prophet**
- **Holt-Winters**
- Gelecek **30 günlük satış projeksiyonu**

### 💰 Fiyat Hassasiyeti Analizi
- Fiyat değişimlerinin talep üzerindeki etkisini gösteren **elastikiyet eğrileri**

### 🌍 Hiyerarşik Analiz
- Bölge → Eyalet → Kategori bazlı:
  - İnteraktif **Treemap** ciro analizi

### 🧪 Hata Teşhisi
- Residuals (artık hata) dağılım grafikleri  
- Gerçek vs tahmin regresyon diyagramları  

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

Bu platform, **yüksek hacimli ticari veriler üzerinde gerçek zamanlı analiz, karar destek ve fiyat optimizasyonu sağlamak amacıyla geliştirilmiş ileri düzey bir yapay zekâ tabanlı analitik sistemdir.**
