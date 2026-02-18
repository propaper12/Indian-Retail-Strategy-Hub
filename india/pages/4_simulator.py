import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils import set_design

# Sayfa ayarlarini yaptim, genis ekran olsun ki grafikler sigsin
st.set_page_config(page_title="Strateji Simulatoru", layout="wide")
set_design()

# Buraya CSS kodlari ekledim cunku duz hali cok cirkin duruyordu.
# Kart gorunumu verdim ki senaryolar birbirinden ayrilsin.
st.markdown("""
<style>
    .scenario-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
        margin-bottom: 20px;
    }
    .scenario-header {
        font-size: 1.2em;
        font-weight: bold;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
    }
    .metric-big {
        font-size: 2.5em;
        font-weight: 800;
        color: #1e293b;
    }
    .metric-delta {
        font-size: 1.2em;
        font-weight: 600;
    }
    .stSlider > div > div > div > div {
        background-color: #3b82f6; 
    }
</style>
""", unsafe_allow_html=True)

st.title("Strateji ve Senaryo Yonetimi")
st.markdown("Yapay zeka modelini kullanarak Mevcut Durum ile Hedeflenen Senaryoyu karsilastiriyoruz.")

# Eger model egitilmediyse hata vermesin diye kontrol koydum
if 'trained_model' not in st.session_state:
    st.warning("Model henuz egitilmedi! Lutfen 'AutoML' sayfasina gidip bir model egitin.")
    st.stop()

# Diger sayfalardan gelen verileri burada kullaniyorum
model = st.session_state['trained_model']
encoders = st.session_state['encoders']
feature_cols = st.session_state['feature_cols']
target_col = st.session_state.get('target_col', 'base_price')
df_raw = st.session_state['df_raw']

# Ekrani ikiye boldum, araya da bosluk koydum ki yapisik durmasinlar
col1, col_space, col2 = st.columns([1, 0.1, 1])

# === SOL TARAF: MEVCUT DURUM ===
with col1:
    # HTML ile cerceve cizdim
    st.markdown('<div class="scenario-card">', unsafe_allow_html=True)
    st.markdown('<div class="scenario-header">Senaryo A (Mevcut Durum)</div>', unsafe_allow_html=True)
    
    input_a = {}
    # Dongu ile butun ozellikleri tek tek ekrana basiyorum
    for col in feature_cols:
        if col in encoders:
            # Eger yaziysa (kategorik) selectbox koydum
            val = st.selectbox(f"{col}", encoders[col].classes_.tolist(), key=f"a_{col}")
            input_a[col] = encoders[col].transform([val])[0]
        else:
            # Sayiysa slider koydum, daha guzel duruyor
            min_v = float(df_raw[col].min())
            max_v = float(df_raw[col].max())
            mean_v = float(df_raw[col].mean())
            # Sliderin adimini ayarladim, buyuk sayilarda 1, kucuklerde 0.1 artsin
            step = 1.0 if max_v > 100 else 0.1
            val = st.slider(f"{col}", min_value=min_v, max_value=max_v, value=mean_v, step=step, key=f"a_{col}")
            input_a[col] = val
    st.markdown('</div>', unsafe_allow_html=True)

# === SAG TARAF: HEDEF DURUM ===
with col2:
    # Buranin ust cizgisini mavi yaptim farkli oldugu belli olsun
    st.markdown('<div class="scenario-card" style="border-top: 5px solid #3b82f6;">', unsafe_allow_html=True)
    st.markdown('<div class="scenario-header">Senaryo B (Hedeflenen)</div>', unsafe_allow_html=True)
    
    input_b = {}
    for col in feature_cols:
        if col in encoders:
            # A senaryosunda ne secildiyse aynisi buraya da gelsin, kullanici ugrasmasin
            default_val = encoders[col].inverse_transform([int(input_a[col])])[0]
            val_idx = encoders[col].classes_.tolist().index(default_val)
            val = st.selectbox(f"{col} (Hedef)", encoders[col].classes_.tolist(), index=val_idx, key=f"b_{col}")
            input_b[col] = encoders[col].transform([val])[0]
        else:
            # Sayilari da A'dan kopyaladim
            min_v = float(df_raw[col].min())
            max_v = float(df_raw[col].max())
            val = st.slider(f"{col} (Hedef)", min_value=min_v, max_value=max_v, value=input_a[col], step=step, key=f"b_{col}")
            input_b[col] = val
    st.markdown('</div>', unsafe_allow_html=True)

# Modeli cagirip tahminleri aliyorum
pred_a = model.predict(pd.DataFrame([input_a]))[0]
pred_b = model.predict(pd.DataFrame([input_b]))[0]

# Aradaki farki hesapladim
delta = pred_b - pred_a
delta_pct = (delta / pred_a) * 100 if pred_a != 0 else 0

st.divider()

# --- SONUC GOSTERGESI ---
st.header("Simulasyon Sonuclari")

c_res1, c_res2, c_res3 = st.columns([1, 1, 2])

with c_res1:
    st.markdown(f"**Tahmin A**")
    st.markdown(f"<div class='metric-big'>{pred_a:,.2f}</div>", unsafe_allow_html=True)
    st.caption("Mevcut Parametrelerle")

with c_res2:
    st.markdown(f"**Tahmin B**")
    # Eger sonuc arttiysa yesil, azaldiysa kirmizi yapiyorum
    color = "#10b981" if delta > 0 else "#ef4444"
    st.markdown(f"<div class='metric-big' style='color:{color}'>{pred_b:,.2f}</div>", unsafe_allow_html=True)
    
    # Ok isaretini ayarladim
    arrow = "UKARI" if delta > 0 else "ASAGI"
    st.markdown(f"<div class='metric-delta' style='color:{color}'>{arrow} %{delta_pct:.2f}</div>", unsafe_allow_html=True)

with c_res3:
    # Buraya bir hiz gostergesi (Gauge Chart) ekledim, patronlar seviyor
    max_gauge = df_raw[target_col].max() * 1.2
    
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = pred_b,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Hedef: {target_col}", 'font': {'size': 20}},
        delta = {'reference': pred_a, 'increasing': {'color': "#10b981"}, 'decreasing': {'color': "#ef4444"}},
        gauge = {
            'axis': {'range': [None, max_gauge], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#3b82f6"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, pred_a], 'color': '#e2e8f0'},
                {'range': [pred_a, max_gauge], 'color': '#ffffff'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': pred_b}}))
    
    fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

st.divider()

# --- OPTIMIZASYON KISMI ---
st.subheader("Yapay Zeka Optimizasyonu")

# Burada hangi degiskeni optimize edecegimize karar veriyorum
optim_feat = None
possible_optim_feats = ["base_price", "discount_percent", "customer_age", "units_sold"]
# Listeden ilk buldugunu sec
for feat in possible_optim_feats:
    if feat in feature_cols:
        optim_feat = feat
        break

if optim_feat:
    st.markdown(f"""
    <div style='background-color:#eff6ff; padding:15px; border-radius:10px; border:1px solid #bfdbfe;'>
        <b>AI Onerisi:</b> Asagidaki grafik, <b>{optim_feat}</b> degiskenini degistirdiginizde sonucun nasil degisecegini gosteriyor. 
        Grafigin en yuksek noktasi, en iyi sonucu alacaginiz yerdir.
    </div>
    """, unsafe_allow_html=True)
    
    # Mevcut degerin yarisindan baslayip 2 katina kadar deniyorum
    base_v = input_a[optim_feat]
    test_values = np.linspace(base_v * 0.5, base_v * 2.0, 50)
    
    results = []
    # Donguyle tek tek tahmin aliyorum
    for v in test_values:
        temp = input_a.copy()
        temp[optim_feat] = v
        results.append(model.predict(pd.DataFrame([temp]))[0])
    
    # Grafigi cizdiriyorum
    fig_opt = go.Figure()
    
    # Alan grafigi sectim (Area Chart) daha dolgun duruyor
    fig_opt.add_trace(go.Scatter(x=test_values, y=results, fill='tozeroy', mode='lines', 
                                 line=dict(color='#3b82f6', width=3), name='Tahmin Egrisi'))
    
    # Mevcut durumu kirmizi nokta ile isaretledim
    fig_opt.add_trace(go.Scatter(x=[input_a[optim_feat]], y=[pred_a], mode='markers', 
                                 marker=dict(color='red', size=12), name='Mevcut Durum (A)'))
    
    # Hedef durumu yesil yildiz ile isaretledim
    fig_opt.add_trace(go.Scatter(x=[input_b[optim_feat]], y=[pred_b], mode='markers', 
                                 marker=dict(color='green', size=12, symbol='star'), name='Hedef Durum (B)'))
    
    # En iyi noktayi bulup uzerine yazi yazdiriyorum
    max_y = max(results)
    max_x = test_values[results.index(max_y)]
    fig_opt.add_annotation(x=max_x, y=max_y, text=f"Optimum: {max_x:.2f}", showarrow=True, arrowhead=2, ax=0, ay=-40)
    
    fig_opt.update_layout(
        title=f"{optim_feat} Degisiminin {target_col} Uzerindeki Etkisi",
        xaxis_title=optim_feat,
        yaxis_title=target_col,
        hovermode="x unified"
    )
    
    st.plotly_chart(fig_opt, use_container_width=True)

else:
    st.info("Optimizasyon grafigini cizmem icin Fiyat veya Indirim gibi sayisal bir veri secmelisiniz.")