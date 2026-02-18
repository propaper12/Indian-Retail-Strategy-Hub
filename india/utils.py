import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Dosya adini buraya yazdim, degisirse buradan degistiririz
FILE_NAME = "indian_ecommerce_pricing_revenue_growth_36_months.csv"

# Veriyi yukleyen fonksiyon
# cache_data kullandim ki her seferinde dosyayi okuyup vakit kaybetmesin
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(FILE_NAME)
        # Tarih formatini duzeltiyorum
        if "order_date" in df.columns:
            df["order_date"] = pd.to_datetime(df["order_date"], errors='coerce')
            df['month'] = df['order_date'].dt.month_name()
            df['day_of_week'] = df['order_date'].dt.day_name()
            df['year'] = df['order_date'].dt.year
        return df
    except FileNotFoundError:
        return None

# Veriyi modele hazirlayan fonksiyon
def run_preprocessing(df, target, features):
    # Sadece lazim olan kolonlari aliyorum
    cols = list(set(features + [target]))
    working_df = df[cols].copy().dropna()
    label_encoders = {}
    
    # Yazilari sayiya ceviriyorum (Label Encoding)
    categorical_cols = working_df.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        working_df[col] = le.fit_transform(working_df[col].astype(str))
        label_encoders[col] = le
        
    return working_df, label_encoders

# Sayfa stilini ayarlayan fonksiyon (Her sayfada cagiracagiz)
def set_design():
    st.markdown("""
        <style>
        .main { background-color: #f8f9fa; }
        div[data-testid="stMetric"] {
            background-color: #1e293b;
            color: #ffffff;
            border: 1px solid #334155;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        div[data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.9em; }
        div[data-testid="stMetricValue"] { color: #f8fafc !important; font-size: 1.6em; font-weight: 700; }
        .explanation-box {
            background-color: #e0f2fe;
            border-left: 5px solid #0284c7;
            padding: 15px;
            border-radius: 5px;
            margin-top: 10px;
            margin-bottom: 25px;
            color: #0c4a6e;
            font-family: 'Segoe UI', sans-serif;
            font-size: 0.95em;
        }
        h1, h2, h3 { color: #0f172a; font-family: 'Segoe UI', sans-serif; font-weight: 700; }
        </style>
        """, unsafe_allow_html=True)