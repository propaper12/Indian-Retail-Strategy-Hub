import streamlit as st
import pandas as pd
import os
from utils import set_design

# Sayfa ayarları
st.set_page_config(page_title="E-Ticaret Analiz Merkezi", layout="wide")
set_design()

st.title(" E-Ticaret Strateji ve Gelir Analizi")

def load_data_from_file():
    file_name = "indian_ecommerce_pricing_revenue_growth_36_months.csv"
    
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
        df['order_date'] = pd.to_datetime(df['order_date'])
        return df
    else:
        return None

if 'df_raw' not in st.session_state:
    df = load_data_from_file()
    if df is not None:
        st.session_state['df_raw'] = df
        st.success(f" Veri seti başarıyla yüklendi: {len(df)} satır işleme hazır.")
    else:
        st.error(" Veri seti dosyası bulunamadı! Lütfen GitHub deponuzun ana dizininde 'indian_ecommerce_pricing_revenue_growth_36_months.csv' dosyasının olduğundan emin olun.")
        st.stop()

df = st.session_state['df_raw']

st.divider()
c1, c2, c3 = st.columns(3)
c1.metric("Toplam Sipariş", f"{len(df):,}")
c2.metric("Toplam Ciro (Revenue)", f"₹ {df['revenue'].sum():,.0f}")
c3.metric("Ortalama İndirim", f"% {df['discount_percent'].mean():.1f}")

st.info("👈 Diğer analiz ve ML modülleri için soldaki menüyü kullanabilirsiniz.")
