import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from utils import set_design

st.set_page_config(page_title="Veri Analizi", layout="wide")
set_design()

st.title("Kesifci Veri Analizi (EDA)")

# Veriyi Home.py'den aliyorum
if 'df_raw' not in st.session_state:
    st.warning("Lutfen once Ana Sayfa (Home) uzerinden veriyi yukleyin.")
    st.stop()

df_raw = st.session_state['df_raw']

# 1. KPI Metrikleri
col1, col2, col3, col4 = st.columns(4)
col1.metric("Toplam Gelir", f"{df_raw['revenue'].sum()/1e6:.1f}M")
col2.metric("Toplam Satis", f"{df_raw['units_sold'].sum():,}")
col3.metric("Ort. Indirim", f"%{df_raw['discount_percent'].mean():.1f}")
col4.metric("Kategori Sayisi", df_raw['category'].nunique())

st.divider()

# --- YENI EKLEME: BOLGESEL PAZAR PAYI (PASTA GRAFIGI) ---
st.subheader("Bolgesel Pazar Payi (Revenue by Zone)")
col_pie1, col_pie2 = st.columns([2, 1])

with col_pie1:
    # Bolgelere gore ciro dagilimi
    zone_df = df_raw.groupby("zone")["revenue"].sum().reset_index()
    fig_pie = px.pie(zone_df, values='revenue', names='zone', hole=0.4)
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)

with col_pie2:
    st.markdown("""
    <div class="explanation-box">
        <b>Analiz Notu:</b> Bu pasta grafigini, satislarimizin hangi cografi bolgelerde yogunlastigini bir bakista anlamak icin ekledim. 
        Mesela 'North' veya 'West' bolgeleri pastanin buyuk dilimini olusturuyorsa, lojistik ve pazarlama butcemizi oraya kaydirmamiz gerekebilir. 
        Kucuk dilimler ise henuz kesfedilmemis firsat pazarlari olabilir.
    </div>
    """, unsafe_allow_html=True)

st.divider()

# --- YENI EKLEME: EN COK HARCAMA YAPAN EYALETLER ---
st.subheader("En Cok Gelir Getiren Eyaletler (Top States)")
# Eyalet bazli ciro toplami
state_rev = df_raw.groupby("state")["revenue"].sum().reset_index().sort_values("revenue", ascending=False).head(10)

fig_bar_state = px.bar(state_rev, x="revenue", y="state", orientation='h', text_auto='.2s', color="revenue", color_continuous_scale='Viridis')
fig_bar_state.update_layout(yaxis=dict(autorange="reversed")) # En yuksegi en uste al
st.plotly_chart(fig_bar_state, use_container_width=True)

st.markdown("""
<div class="explanation-box">
    <b>Analiz Notu:</b> Burada 'Pareto Prensibi'ni test ediyorum. Genelde gelirlerin %80'i, sehirlerin %20'sinden gelir. 
    Bu grafik, hangi eyaletlerin bizim 'Yildiz Oyuncularimiz' oldugunu gosteriyor. 
    Eger pazarlama yapacaksak, butceyi buradaki ilk 3-4 eyalete harcamak, sonlardaki eyaletlere harcamaktan cok daha karli olacaktir.
</div>
""", unsafe_allow_html=True)

st.divider()

# --- KORELASYON ISI HARITASI ---
st.subheader("Korelasyon Isi Haritasi (Heatmap)")
numeric_df = df_raw.select_dtypes(include=[np.number])
if not numeric_df.empty:
    corr_matrix = numeric_df.corr()
    fig_heat = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r")
    st.plotly_chart(fig_heat, use_container_width=True)
    
    st.markdown("""
    <div class="explanation-box">
        <b>Analiz Notu:</b> Degiskenler arasindaki gizli iliskileri ortaya cikarmak icin bu haritayi kullaniyorum. 
        Koyu kirmizi kutucuklar, iki verinin birlikte arttigini (pozitif iliski) gosterir. 
        Koyu mavi kutucuklar ise biri artarken digerinin azaldigini (negatif iliski) soyler. 
        Ornegin; 'Fiyat' artarken 'Satis Adedi' azaliyorsa (Mavi), urunlerimiz fiyata duyarli demektir.
    </div>
    """, unsafe_allow_html=True)

st.divider()

# --- AYKIRI DEGER ANALIZI ---
st.subheader("Aykiri Deger Analizi (Outlier Detection)")
col_box_sel = st.selectbox("Incelemek istediginiz veri:", numeric_df.columns, index=0)
fig_iqr = px.box(df_raw, y=col_box_sel)
st.plotly_chart(fig_iqr, use_container_width=True)

st.markdown("""
<div class="explanation-box">
    <b>Analiz Notu:</b> Veri setindeki 'normal disi' durumlari yakalamak icin bu kutu grafigini (Box Plot) cizdim. 
    Kutunun ustundeki ve altindaki cizgilerin disinda kalan noktalar 'Outlier' yani aykiri degerlerdir. 
    Mesela ortalama fiyat 1000 TL iken bir urun 50.000 TL ise burada gorunur. 
    Bu veriler analizi bozabilir, o yuzden onlari tespit edip belki de temizlememiz gerekir.
</div>
""", unsafe_allow_html=True)

st.divider()

# --- STOK VE REKABET ---
col_st1, col_st2 = st.columns(2)
with col_st1:
    st.subheader("Stok Baskisi vs Indirim")
    fig_box = px.box(df_raw, x="inventory_pressure", y="discount_percent", color="inventory_pressure", color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'})
    st.plotly_chart(fig_box, use_container_width=True)
    st.markdown("""
    <div class="explanation-box">
        <b>Analiz Notu:</b> Depoda mal sisince (High Inventory Pressure), elimizden cikarmak icin daha fazla indirim yapiyor muyuz? 
        Bu grafik stratejimizin tutup tutmadigini gosteriyor. Kirmizi kutu daha yukaridaysa, stok eritebilmek icin fiyat kiriyoruz demektir.
    </div>
    """, unsafe_allow_html=True)

with col_st2:
    st.subheader("Rekabet Analizi")
    comp_df = df_raw.groupby("competition_intensity")[["base_price", "final_price"]].mean().reset_index()
    fig_comp = px.bar(comp_df, x="competition_intensity", y=["base_price", "final_price"], barmode="group")
    st.plotly_chart(fig_comp, use_container_width=True)
    st.markdown("""
    <div class="explanation-box">
        <b>Analiz Notu:</b> Rakiplerin cok oldugu (High Competition) pazarlarda fiyatlarimiz ne durumda? 
        Mavi cubuk (Baz Fiyat) ile Kirmizi cubuk (Son Fiyat) arasindaki fark, rekabet yuzunden ne kadar indirim yapmak zorunda kaldigimizi gosterir.
    </div>
    """, unsafe_allow_html=True)

st.divider()

# --- TREEMAP ---
st.subheader("Pazar Haritasi (Treemap)")
fig_tree = px.treemap(df_raw, path=['zone', 'state', 'category'], values='revenue', color='units_sold', color_continuous_scale='RdYlGn')
st.plotly_chart(fig_tree, use_container_width=True)
st.markdown("""
<div class="explanation-box">
    <b>Analiz Notu:</b> Bu harita buyuk resmi gormek icin. Hangi bolgede, hangi eyalette, hangi kategori daha cok ciro yapiyor? 
    Kutunun buyuklugu parayi (Ciro), rengi ise satilan adedi gosterir. Yesil olanlar surumden kazandiranlar, kirmizi olanlar ise az satan ama cirosu yuksek olanlardir.
</div>
""", unsafe_allow_html=True)