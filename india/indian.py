import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# Makine Ogrenmesi kutuphanelerini buraya import ettim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
# Kumeleme icin bunu ekledim (KMeans)
from sklearn.cluster import KMeans

# Zaman serisi icin lazim olan kutuphane
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Bazen bu kutuphaneler sunucuda olmuyor, kod patlamasin diye try-except yaptim
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except:
    PROPHET_AVAILABLE = False

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except:
    LIME_AVAILABLE = False

# ---------------------------------------------------------
# 1. SAYFA AYARLARI
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Strategy Hub Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tasarim kodlari. Emojisiz, sade ve temiz olsun istedim.
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    
    /* Metrik kartlarinin tasarimi */
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
    
    /* Aciklama kutulari icin stil */
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
    
    /* Baslik fontlari */
    h1, h2, h3 { color: #0f172a; font-family: 'Segoe UI', sans-serif; font-weight: 700; }
    </style>
    """, unsafe_allow_html=True)

FILE_NAME = "indian_ecommerce_pricing_revenue_growth_36_months.csv"

# ---------------------------------------------------------
# 2. VERI YUKLEME VE ISLEME
# ---------------------------------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(FILE_NAME)
        # Tarih kolonunu duzeltiyorum, object olarak kalirsa islem yapamiyoruz
        if "order_date" in df.columns:
            df["order_date"] = pd.to_datetime(df["order_date"], errors='coerce')
            df['month'] = df['order_date'].dt.month_name()
            df['day_of_week'] = df['order_date'].dt.day_name()
            df['year'] = df['order_date'].dt.year
        return df
    except FileNotFoundError:
        return None

def run_preprocessing(df, target, features):
    # Secilen kolonlari alip bos verileri (NaN) siliyorum
    cols = list(set(features + [target]))
    working_df = df[cols].copy().dropna()
    label_encoders = {}
    
    # Kategorik (yazi) olan kolonlari bulup sayiya ceviriyorum
    categorical_cols = working_df.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        working_df[col] = le.fit_transform(working_df[col].astype(str))
        label_encoders[col] = le
        
    return working_df, label_encoders

df_raw = load_data()

if df_raw is None:
    st.error(f"Dosya bulunamadi: {FILE_NAME}")
    st.stop()

# ---------------------------------------------------------
# 3. SIDEBAR (SOL MENU)
# ---------------------------------------------------------
with st.sidebar:
    st.title("AI Strategy Hub")
    st.info("E-Ticaret Fiyatlandirma ve Buyume Analitigi")
    
    all_cols = df_raw.columns.tolist()
    
    # Hedef degisken secimi (Genelde base_price tahmin edilir)
    target_options = ["base_price", "units_sold", "revenue"]
    default_target = "base_price" if "base_price" in all_cols else all_cols[0]
    target_col = st.selectbox("Hedef Degisken (Target)", all_cols, index=all_cols.index(default_target))
    
    # Gelecegi tahmin ederken bilmemizin imkansiz oldugu kolonlari cikariyorum (Data Leakage onlemi)
    exclude_list = [target_col, "order_id", "order_date", "month", "day_of_week", "year"]
    potential_leakage = ["final_price", "revenue", "units_sold", "discount_percent"]
    
    feature_candidates = [c for c in all_cols if c not in exclude_list and c not in potential_leakage]
    
    selected_features = st.multiselect(
        "Model Ozellikleri (Features)", 
        feature_candidates,
        default=feature_candidates[:5]
    )
    
    split_ratio = st.slider("Test Verisi Orani", 10, 40, 20) / 100
    
    st.divider()
    st.caption("Designed by Data Engineer")

# ---------------------------------------------------------
# 4. ANA DASHBOARD
# ---------------------------------------------------------
st.title("E-Ticaret Strateji ve Fiyatlandirma Zekasi")
st.markdown("""
Bu proje, verileri analiz etmek ve fiyat tahmini yapmak icin gelistirildi. 
AutoML, Zaman Serisi ve Kumeleme algoritmalarini icerir.
""")

# Sekmeler
tabs = st.tabs(["Veri Analizi (EDA)", "AutoML & Kumeleme", "Zaman Serisi", "Strateji Simulatoru"])

# --- TAB 1: VERI ANALIZI (EDA) ---
with tabs[0]:
    st.header("Kesifci Veri Analizi (Exploratory Data Analysis)")
    
    # 1. KPI Metrikleri
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Toplam Gelir", f"{df_raw['revenue'].sum()/1e6:.1f}M")
    col2.metric("Toplam Satis Adedi", f"{df_raw['units_sold'].sum():,}")
    col3.metric("Ortalama Indirim", f"%{df_raw['discount_percent'].mean():.1f}")
    col4.metric("Aktif Kategori", df_raw['category'].nunique())
    
    st.divider()

    # YENI GRAFIK: HEATMAP
    st.subheader("Degiskenler Arasi Iliski (Correlation Heatmap)")
    # Sadece sayisal kolonlari aliyorum yoksa kod hata veriyor
    numeric_df = df_raw.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        corr_matrix = numeric_df.corr()
        fig_heat = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", title="Korelasyon Matrisi")
        st.plotly_chart(fig_heat, use_container_width=True)
        st.markdown("""
        <div class="explanation-box">
            <b>Analiz Notu:</b> Bu haritayi degiskenlerin birbirini nasil etkiledigini gormek icin ekledim. 
            Kirmizi renkler pozitif iliskiyi, mavi renkler negatif iliskiyi gosterir. Mesela fiyat artinca satis dusuyor mu?
        </div>
        """, unsafe_allow_html=True)
    
    # YENI GRAFIK: IQR BOX PLOT
    st.subheader("Aykiri Deger Analizi (IQR Box Plot)")
    col_box_sel = st.selectbox("Incelemek istediginiz veri:", numeric_df.columns, index=0)
    fig_iqr = px.box(df_raw, y=col_box_sel, title=f"{col_box_sel} Icin Aykiri Deger Kontrolu")
    st.plotly_chart(fig_iqr, use_container_width=True)
    st.markdown("""
    <div class="explanation-box">
        <b>Analiz Notu:</b> Kutunun disinda kalan noktalar 'Outlier' yani aykiri degerlerdir. 
        Bunlar veri setini bozabilir, o yuzden kontrol etmek istedim.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # 2. Pareto Analizi (Eski grafik, oldugu gibi biraktim)
    st.subheader("1. Pareto Analizi (80/20 Kurali)")
    pareto_df = df_raw.groupby("category")["revenue"].sum().reset_index().sort_values("revenue", ascending=False)
    pareto_df["cumulative_percentage"] = pareto_df["revenue"].cumsum() / pareto_df["revenue"].sum() * 100
    
    fig_pareto = go.Figure()
    fig_pareto.add_trace(go.Bar(x=pareto_df["category"], y=pareto_df["revenue"], name="Gelir", marker_color='rgb(55, 83, 109)'))
    fig_pareto.add_trace(go.Scatter(x=pareto_df["category"], y=pareto_df["cumulative_percentage"], name="Kumulatif %", yaxis="y2", marker_color='rgb(26, 118, 255)'))
    fig_pareto.update_layout(
        title="Kategori Bazli Gelir Dagilimi (Pareto)",
        yaxis=dict(title="Gelir"),
        yaxis2=dict(title="Kumulatif %", overlaying="y", side="right", showgrid=False, range=[0, 110]),
        legend=dict(x=0.8, y=1.1)
    )
    st.plotly_chart(fig_pareto, use_container_width=True)
    st.markdown("""
    <div class="explanation-box">
        <b>Analiz Notu:</b> Gelirimizin cogu hangi kategorilerden geliyor ona bakiyoruz. Az sayida kategori cironun cogunu yapiyor olabilir.
    </div>
    """, unsafe_allow_html=True)

    # 3. Stok Baskisi ve Indirim (Eski grafik)
    col_st1, col_st2 = st.columns(2)
    with col_st1:
        st.subheader("2. Stok Baskisi vs Indirim")
        fig_box = px.box(df_raw, x="inventory_pressure", y="discount_percent", color="inventory_pressure",
                         title="Stok Baskisinin Indirime Etkisi",
                         color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'})
        st.plotly_chart(fig_box, use_container_width=True)
        st.markdown("""
        <div class="explanation-box">
            <b>Analiz Notu:</b> Stok baskisi (Inventory Pressure) yuksek oldugunda daha fazla indirim yapiyor muyuz? Bunu kontrol ediyorum.
        </div>
        """, unsafe_allow_html=True)
        
    with col_st2:
        st.subheader("3. Rekabet Yogunlugu Analizi")
        comp_df = df_raw.groupby("competition_intensity")[["base_price", "final_price"]].mean().reset_index()
        fig_comp = px.bar(comp_df, x="competition_intensity", y=["base_price", "final_price"], barmode="group",
                          title="Rekabete Gore Fiyatlandirma", color_discrete_sequence=['#636EFA', '#EF553B'])
        st.plotly_chart(fig_comp, use_container_width=True)
        st.markdown("""
        <div class="explanation-box">
            <b>Analiz Notu:</b> Rekabetin (Competition) yuksek oldugu yerlerde fiyatlari ne kadar kiriyoruz, burada onu goruyoruz.
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # 4. Mevsimsellik Isi Haritasi (Eski grafik)
    st.subheader("4. Satis Isi Haritasi (Mevsimsellik)")
    if "month" in df_raw.columns and "day_of_week" in df_raw.columns:
        heatmap_data = df_raw.pivot_table(index="day_of_week", columns="month", values="revenue", aggfunc="sum")
        # Siralamayi duzeltiyorum yoksa karisik geliyor
        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        months_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        
        heatmap_data = heatmap_data.reindex(days_order)
        heatmap_data = heatmap_data.reindex(columns=months_order)
        
        fig_heat = px.imshow(heatmap_data, text_auto=".2s", aspect="auto", color_continuous_scale="Viridis",
                             title="Gun ve Ay Bazli Gelir Yogunlugu")
        st.plotly_chart(fig_heat, use_container_width=True)
        st.markdown("""
        <div class="explanation-box">
            <b>Analiz Notu:</b> Hangi ayin hangi gunu satislar patliyor? Kampanya planlamak icin buna bakmamiz lazim.
        </div>
        """, unsafe_allow_html=True)

    # 5. Hiyerarsik Treemap (Eski grafik)
    st.subheader("5. Hiyerarsik Pazar Haritasi")
    fig_tree = px.treemap(df_raw, path=['zone', 'state', 'category'], values='revenue', color='units_sold',
                          color_continuous_scale='RdYlGn', title="Bolge > Eyalet > Kategori Ciro Akisi")
    st.plotly_chart(fig_tree, use_container_width=True)
    st.markdown("""
    <div class="explanation-box">
        <b>Analiz Notu:</b> Bolgeden kategoriye kadar paraning nereden geldigini gosteren harita. Yesil olanlar cok satanlar.
    </div>
    """, unsafe_allow_html=True)

# --- TAB 2: AUTOML & TAHMIN & KUMELEME ---
with tabs[1]:
    st.header("AutoML Sampiyon Model Yarismasi")
    st.markdown("Veriyi egitip en iyi sonucu veren modeli otomatik seciyoruz.")

    if len(selected_features) > 0:
        if st.button("Modelleri Yaristir"):
            with st.spinner("Modeller egitiliyor... Biraz surebilir."):
                # Veri Hazirligi
                df_p, encoders = run_preprocessing(df_raw, target_col, selected_features)
                X = df_p.drop(columns=[target_col])
                y = df_p[target_col]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=42)
                
                # Modelleri tanimliyorum
                models = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42),
                    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1),
                    "LightGBM": LGBMRegressor(n_estimators=100, verbose=-1)
                }
                
                results = []
                best_score = -np.inf
                best_model = None
                best_name = ""
                
                progress_bar = st.progress(0)
                
                for idx, (name, model) in enumerate(models.items()):
                    try:
                        model.fit(X_train, y_train)
                        preds = model.predict(X_test)
                        r2 = r2_score(y_test, preds)
                        mae = mean_absolute_error(y_test, preds)
                        rmse = np.sqrt(mean_squared_error(y_test, preds))
                        
                        results.append({"Model": name, "R2": r2, "MAE": mae, "RMSE": rmse})
                        
                        if r2 > best_score:
                            best_score = r2
                            best_model = model
                            best_name = name
                    except Exception as e:
                        st.warning(f"{name} modelinde hata oldu: {e}")
                    
                    progress_bar.progress((idx + 1) / len(models))
                
                # Sonuclari kaydediyorum
                st.session_state['trained_model'] = best_model
                st.session_state['encoders'] = encoders
                st.session_state['feature_cols'] = X.columns.tolist()
                st.session_state['automl_results'] = pd.DataFrame(results).sort_values("R2", ascending=False)
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                
                st.success(f"En Iyi Model: {best_name} (R2: {best_score:.4f})")

        if 'trained_model' in st.session_state:
            res_df = st.session_state['automl_results']
            st.dataframe(res_df.style.format({"R2": "{:.4f}", "MAE": "{:.2f}", "RMSE": "{:.2f}"}), use_container_width=True)
            
            # Feature Importance
            model = st.session_state['trained_model']
            if hasattr(model, 'feature_importances_'):
                imp_df = pd.DataFrame({
                    'Feature': st.session_state['feature_cols'],
                    'Importance': model.feature_importances_
                }).sort_values(by='Importance', ascending=True)
                
                fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h', title="Model Karar Agirliklari")
                st.plotly_chart(fig_imp, use_container_width=True)
                st.markdown("""
                <div class="explanation-box">
                    <b>Analiz Notu:</b> Yapay zeka tahmin yaparken en cok hangi veriye bakiyor? Bunu gormek onemli.
                </div>
                """, unsafe_allow_html=True)
                
            # Gercek vs Tahmin
            y_test = st.session_state['y_test']
            preds = model.predict(st.session_state['X_test'])
            
            fig_scat = px.scatter(x=y_test, y=preds, labels={'x': 'Gercek Deger', 'y': 'AI Tahmini'}, 
                                  title="Dogruluk Testi (Gercek vs Tahmin)", opacity=0.6)
            fig_scat.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(), line=dict(color="Red", dash="dash"))
            st.plotly_chart(fig_scat, use_container_width=True)
            st.markdown("""
            <div class="explanation-box">
                <b>Analiz Notu:</b> Kirmizi cizgiye ne kadar yakinsak o kadar iyi tahmin yapiyoruz demektir.
            </div>
            """, unsafe_allow_html=True)
            
    else:
        st.warning("Lutfen sol menuden en az bir ozellik secin.")

    st.divider()

    # --- KUMELEME (CLUSTERING) BOLUMU (YENI) ---
    st.header("Musteri/Urun Kumeleme Analizi (K-Means)")
    st.markdown("Regresyonla tahmin yaptik ama bir de verileri gruplara ayiralim istedim.")
    
    # Sadece sayisal verileri aliyorum
    numeric_for_cluster = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    
    col_clus1, col_clus2 = st.columns(2)
    with col_clus1:
        cluster_feats = st.multiselect("Kumeleme icin kullanilacak veriler:", numeric_for_cluster, default=numeric_for_cluster[:2])
    with col_clus2:
        k_val = st.slider("Kac gruba ayiralim? (K Degeri)", 2, 8, 3)
    
    if st.button("Gruplara Ayir"):
        if len(cluster_feats) >= 2:
            try:
                # Veriyi hazirlama
                X_cluster = df_raw[cluster_feats].dropna()
                
                # Verileri olcekleme (StandardScaler) - KMeans icin onemli
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_cluster)
                
                # Modeli kurma
                kmeans = KMeans(n_clusters=k_val, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X_scaled)
                
                # Sonuclari tabloya ekleme
                X_cluster['Cluster'] = clusters.astype(str)
                
                st.success("Veriler basariyla gruplandirildi!")
                
                # Grafigi cizdirme
                fig_cluster = px.scatter(X_cluster, x=cluster_feats[0], y=cluster_feats[1], color='Cluster',
                                       title=f"K-Means Sonuclari (K={k_val})")
                st.plotly_chart(fig_cluster, use_container_width=True)
                
                st.markdown("""
                <div class="explanation-box">
                    <b>Analiz Notu:</b> Verileri benzerliklerine gore grupladik. Ayni renkteki noktalar birbirine benzeyen urunleri veya musterileri temsil ediyor.
                    Bunu pazarlama stratejisi belirlerken kullanabiliriz.
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Kumeleme yaparken hata oldu: {e}")
        else:
            st.warning("Kumeleme yapmak icin en az 2 tane veri secmeniz lazim.")

# --- TAB 3: ZAMAN SERISI ---
with tabs[2]:
    st.header("Zaman Serisi Projeksiyonu")
    
    if "order_date" in df_raw.columns:
        ts_data = df_raw.groupby("order_date")[target_col].mean().reset_index()
        ts_data.columns = ['ds', 'y'] # Prophet bu formatta istiyor
        
        col_t1, col_t2 = st.columns([1, 3])
        with col_t1:
            method = st.radio("Metod Secimi", ["Prophet (AI Tabanli)", "Holt-Winters (Istatistiksel)"])
            forecast_days = st.slider("Tahmin Gun Sayisi", 7, 90, 30)
            run_forecast = st.button("Gelecegi Tahminle")
            
        with col_t2:
            if run_forecast:
                if method == "Prophet (AI Tabanli)" and PROPHET_AVAILABLE:
                    with st.spinner("Prophet motoru calisiyor..."):
                        m = Prophet()
                        m.fit(ts_data)
                        future = m.make_future_dataframe(periods=forecast_days)
                        forecast = m.predict(future)
                        
                        fig_prophet = px.line(forecast, x='ds', y='yhat', title=f"{forecast_days} Gunluk AI Projeksiyonu")
                        # Guven araligini ekledim
                        fig_prophet.add_traces(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
                        fig_prophet.add_traces(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)', fillcolor='rgba(0, 100, 80, 0.2)', name='Guven Araligi'))
                        st.plotly_chart(fig_prophet, use_container_width=True)
                else:
                    # Prophet yoksa Holt-Winters kullaniyoruz
                    try:
                        ts_indexed = ts_data.set_index('ds').asfreq('D').fillna(method='ffill')
                        model_hw = ExponentialSmoothing(ts_indexed['y'], trend='add', seasonal='add', seasonal_periods=7).fit()
                        pred = model_hw.forecast(forecast_days)
                        
                        fig_hw = go.Figure()
                        fig_hw.add_trace(go.Scatter(x=ts_indexed.index[-60:], y=ts_indexed['y'][-60:], name="Gecmis Veri"))
                        fig_hw.add_trace(go.Scatter(x=pred.index, y=pred.values, name="Tahmin", line=dict(color='orange', dash='dot')))
                        fig_hw.update_layout(title="Holt-Winters Istatistiksel Tahmin")
                        st.plotly_chart(fig_hw, use_container_width=True)
                    except Exception as e:
                        st.error(f"Holt-Winters hatasi: {e}")
                
                st.markdown("""
                <div class="explanation-box">
                    <b>Analiz Notu:</b> Gecmis verideki trendleri kullanarak gelecekte satislar nasil olacak onu gormeye calisiyoruz.
                </div>
                """, unsafe_allow_html=True)
    else:
        st.error("Tarih verisi bulunamadi.")

# ... (Üst kısımlar, importlar ve önceki tablar AYNI kalsın) ...

# --- TAB 4: STRATEJI SIMULATORU (BAŞTAN AŞAĞI YENİLENDİ) ---
with tabs[3]:
    st.header("Strateji ve Senaryo Yonetimi")
    st.markdown("Yapay zeka ile farkli senaryolari (A/B Testi) karsilastirin ve en karli stratejiyi bulun.")
    
    if 'trained_model' not in st.session_state:
        st.warning("Lutfen once 'AutoML' sekmesinden bir model egitin.")
    else:
        model = st.session_state['trained_model']
        encoders = st.session_state['encoders']
        feature_cols = st.session_state['feature_cols']
        
        # Ekranı ikiye boluyorum: Senaryo A ve Senaryo B
        col_a, col_b = st.columns(2)
        
        # --- SENARYO A AYARLARI ---
        with col_a:
            st.subheader("Senaryo A (Mevcut Durum)")
            input_a = {}
            for col in feature_cols:
                if col in encoders:
                    val = st.selectbox(f"A - {col}", encoders[col].classes_.tolist(), key=f"a_{col}")
                    input_a[col] = encoders[col].transform([val])[0]
                else:
                    min_v = float(df_raw[col].min())
                    max_v = float(df_raw[col].max())
                    mean_v = float(df_raw[col].mean())
                    val = st.number_input(f"A - {col}", min_value=min_v, max_value=max_v, value=mean_v, key=f"a_{col}")
                    input_a[col] = val
            
            # Tahmin A
            pred_a = model.predict(pd.DataFrame([input_a]))[0]
            st.metric(f"Senaryo A Tahmini ({target_col})", f"{pred_a:.2f}")

        # --- SENARYO B AYARLARI ---
        with col_b:
            st.subheader("Senaryo B (Hedeflenen Durum)")
            input_b = {}
            for col in feature_cols:
                if col in encoders:
                    # Varsayilan olarak A ile ayni olsun, kullanici degistirsin
                    default_idx = encoders[col].classes_.tolist().index(encoders[col].inverse_transform([int(input_a[col])])[0])
                    val = st.selectbox(f"B - {col}", encoders[col].classes_.tolist(), index=default_idx, key=f"b_{col}")
                    input_b[col] = encoders[col].transform([val])[0]
                else:
                    val = st.number_input(f"B - {col}", min_value=min_v, max_value=max_v, value=input_a[col], key=f"b_{col}")
                    input_b[col] = val
            
            # Tahmin B
            pred_b = model.predict(pd.DataFrame([input_b]))[0]
            delta = pred_b - pred_a
            st.metric(f"Senaryo B Tahmini ({target_col})", f"{pred_b:.2f}", delta=f"{delta:.2f}")

        st.divider()

        # --- OPTIMAL FIYAT BULUCU (PRICE OPTIMIZER) ---
        st.subheader("Optimal Fiyat Analizi (Price Elasticity)")
        
        # Eger 'base_price' veya 'discount_percent' özellikler arasindaysa analiz yapabiliriz
        optim_feat = None
        if "base_price" in feature_cols: optim_feat = "base_price"
        elif "discount_percent" in feature_cols: optim_feat = "discount_percent"
        
        if optim_feat:
            st.markdown(f"**{optim_feat}** degiskenini degistirerek en iyi sonucu bulmaya calisiyoruz.")
            
            # Senaryo A'yi baz alarak bir aralik olusturuyorum
            base_v = input_a[optim_feat]
            test_values = np.linspace(base_v * 0.5, base_v * 1.5, 50)
            
            optim_results = []
            for v in test_values:
                temp_input = input_a.copy()
                temp_input[optim_feat] = v
                pred = model.predict(pd.DataFrame([temp_input]))[0]
                optim_results.append(pred)
            
            # Grafigi cizdiriyorum
            fig_opt = go.Figure()
            fig_opt.add_trace(go.Scatter(x=test_values, y=optim_results, mode='lines', name='Tahmin Eğrisi', line=dict(color='#2ecc71', width=4)))
            
            # Zirve noktayi isaretliyorum
            max_y = max(optim_results)
            max_x = test_values[optim_results.index(max_y)]
            
            fig_opt.add_annotation(x=max_x, y=max_y, text=f"Zirve Noktası: {max_x:.2f}", showarrow=True, arrowhead=1)
            
            fig_opt.update_layout(title=f"Optimizasyon Egrisi: {optim_feat} vs {target_col}", xaxis_title=optim_feat, yaxis_title=f"Tahmini {target_col}")
            st.plotly_chart(fig_opt, use_container_width=True)
            
            st.markdown("""
            <div class="explanation-box">
                <b>Stratejik İpucu:</b> Bu grafik, fiyati veya indirimi ne kadar artirirsaniz gelirin (veya hedef degiskenin) nerede zirve yaptigini gosterir. 
                Egrinin dondugu nokta, sizin icin en verimli noktadir.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Optimizasyon icin modelinizde 'base_price' veya 'discount_percent' ozelliklerinin secili olmasi gerekir.")