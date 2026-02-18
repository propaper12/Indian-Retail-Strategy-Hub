import streamlit as st
import pandas as pd
from utils import load_data, set_design

# Sayfa ayari
st.set_page_config(page_title="AI Strategy Hub", layout="wide")
set_design()

st.title("AI Strategy Hub Pro")
st.markdown("### E-Ticaret Fiyatlandirma ve Buyume Analitigi")

st.markdown("""
Bu proje, veri analizi ve yapay zeka kullanarak sirket stratejilerini belirlemek icin yapilmistir.
Soldaki menuden istediginiz sayfaya gidebilirsiniz.

**Neler Yapabiliriz?**
1.  **Veri Analizi:** Satislarimiz ne durumda?
2.  **AutoML:** Fiyat tahmini yapan yapay zeka modelleri.
3.  **Zaman Serisi:** Gelecek ay satislar ne olacak?
4.  **Strateji:** Fiyati degistirirsek karimiz ne olur?
""")

# Veriyi yukluyorum
df = load_data()

if df is None:
    st.error("Veri dosyasi bulunamadi! Lutfen 'indian_ecommerce...csv' dosyasini ana dizine koyun.")
    st.stop()

# --- SIDEBAR (AYARLAR) ---
# Buradaki ayarlari 'session_state' icine atiyorum ki diger sayfalarda da kullanabileyim
with st.sidebar:
    st.header("Genel Ayarlar")
    
    all_cols = df.columns.tolist()
    
    # Hedef degisken
    default_target = "base_price" if "base_price" in all_cols else all_cols[0]
    target_col = st.selectbox("Hedef Degisken (Target)", all_cols, index=all_cols.index(default_target))
    
    # Gereksiz kolonlari cikariyorum
    exclude_list = [target_col, "order_id", "order_date", "month", "day_of_week", "year"]
    potential_leakage = ["final_price", "revenue", "units_sold", "discount_percent"]
    
    feature_candidates = [c for c in all_cols if c not in exclude_list and c not in potential_leakage]
    
    # Secilen ozellikler
    selected_features = st.multiselect(
        "Model Ozellikleri", 
        feature_candidates,
        default=feature_candidates[:5]
    )
    
    split_ratio = st.slider("Test Verisi Orani", 10, 40, 20) / 100
    
    # Session State'e kaydetme (Diger sayfalar buradan okuyacak)
    st.session_state['df_raw'] = df
    st.session_state['target_col'] = target_col
    st.session_state['selected_features'] = selected_features
    st.session_state['split_ratio'] = split_ratio
    
    st.success("Ayarlar Kaydedildi! Diger sayfalara gecebilirsiniz.")