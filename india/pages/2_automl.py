import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from utils import set_design, run_preprocessing

st.set_page_config(page_title="AutoML & AI", layout="wide")
set_design()

st.title("AutoML ve Kumeleme Laboratuvari")

if 'df_raw' not in st.session_state:
    st.warning("Lutfen once Ana Sayfa'ya gidip verileri yukleyin.")
    st.stop()

# Verileri Session State'den aliyorum
df_raw = st.session_state['df_raw']
target_col = st.session_state.get('target_col', 'base_price')
selected_features = st.session_state.get('selected_features', [])
split_ratio = st.session_state.get('split_ratio', 0.2)

#AutoML Bolumu
st.header("1. Sampiyon Model Yarismasi")
st.markdown("Veriyi farkli zekalarla egitiyoruz ve hangisi daha iyi tahmin ediyor diye kapistiriyoruz.")

if len(selected_features) > 0:
    # Egitim Butonu
    if st.button("Modelleri Yaristir"):
        with st.spinner("Modeller egitiliyor... Islemci isiniyor..."):
            # Veri Hazirligi (Preprocessing)
            df_p, encoders = run_preprocessing(df_raw, target_col, selected_features)
            X = df_p.drop(columns=[target_col])
            y = df_p[target_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=42)
            
            # Modelleri listeye koydum
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
            
            # Ilerleme cubugu (Progress Bar)
            progress = st.progress(0)
            
            for idx, (name, model) in enumerate(models.items()):
                try:
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    r2 = r2_score(y_test, preds)
                    rmse = np.sqrt(mean_squared_error(y_test, preds))
                    mae = mean_absolute_error(y_test, preds)
                    
                    results.append({"Model": name, "R2": r2, "RMSE": rmse, "MAE": mae})
                    
                    if r2 > best_score:
                        best_score = r2
                        best_model = model
                        best_name = name
                except Exception as e:
                    st.error(f"{name} hatasi: {e}")
                
                progress.progress((idx + 1) / len(models))
            
            # Sonuclari kaydetme (Session State)
            st.session_state['trained_model'] = best_model
            st.session_state['encoders'] = encoders
            st.session_state['feature_cols'] = X.columns.tolist()
            st.session_state['automl_results'] = pd.DataFrame(results).sort_values("R2", ascending=False)
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
            st.session_state['best_name'] = best_name
            
            st.success(f"Sampiyon Belli Oldu: {best_name} (R2 Skor: {best_score:.4f})")

    #MODEL SONUCLARI VE GRAFIKLER
    if 'trained_model' in st.session_state:
        res_df = st.session_state['automl_results']
        best_model = st.session_state['trained_model']
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        
        # 1. Tablo ve Skor Karsilastirma Grafigi
        col_res1, col_res2 = st.columns([1, 1])
        
        with col_res1:
            st.subheader("Skor Tablosu")
            st.dataframe(res_df.style.format({"R2": "{:.4f}", "RMSE": "{:.2f}", "MAE": "{:.2f}"}), use_container_width=True)
        
        with col_res2:
            st.subheader("Modellerin Karsilastirilmasi")
            # R2 Skorlarina gore bar grafik
            fig_comp = px.bar(res_df, x='R2', y='Model', orientation='h', color='R2', title="Hangi Model Daha Zeki?", color_continuous_scale='Viridis')
            st.plotly_chart(fig_comp, use_container_width=True)
            st.markdown('<div class="explanation-box">R2 skoru 1\'e ne kadar yakinsa model o kadar basarilidir. Burada hangi modelin kazandigini gorsellestirdim.</div>', unsafe_allow_html=True)

        st.divider()

        # 2. Gercek vs Tahmin ve Hata Dagilimi
        col_g1, col_g2 = st.columns(2)
        
        preds = best_model.predict(X_test)
        
        with col_g1:
            st.subheader("Gercek vs Tahmin (Dogruluk)")
            fig_scat = px.scatter(x=y_test, y=preds, labels={'x': 'Gercek Deger', 'y': 'AI Tahmini'}, opacity=0.6)
            # Mukemmel tahmin cizgisi (Kirmizi)
            fig_scat.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(), line=dict(color="Red", dash="dash"))
            st.plotly_chart(fig_scat, use_container_width=True)
            st.markdown('<div class="explanation-box">Noktalar kirmizi cizginin uzerindeyse tam isabet demektir. Dagilmissa model bazi yerlerde zorlaniyor.</div>', unsafe_allow_html=True)
            
        with col_g2:
            st.subheader("Hata Dagilimi (Residuals)")
            residuals = y_test - preds
            fig_res = px.histogram(x=residuals, nbins=50, title="Hata Paylari")
            st.plotly_chart(fig_res, use_container_width=True)
            st.markdown('<div class="explanation-box">Hatalarin sifir etrafinda toplanmasi lazim. Eger saga veya sola cok kaymissa model yanli (bias) ogrenmis olabilir.</div>', unsafe_allow_html=True)

        # 3. Feature Importance (Eger varsa)
        if hasattr(best_model, 'feature_importances_'):
            st.subheader("Modelin Karar Mekanizmasi")
            imp_df = pd.DataFrame({'Feature': X_test.columns, 'Importance': best_model.feature_importances_}).sort_values('Importance', ascending=True)
            fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h', title="Model En Cok Neye Bakiyor?")
            st.plotly_chart(fig_imp, use_container_width=True)
            st.markdown('<div class="explanation-box">Yapay zeka fiyati tahmin ederken en cok hangi veriye onem vermis? Bunu gormek strateji icin cok onemli.</div>', unsafe_allow_html=True)

else:
    st.info("Lutfen sol menuden (Sidebar) model icin ozellik secin.")

st.divider()

# 2. BOLUM: KUMELEME (CLUSTERING)

st.header("2. Musteri Kumeleme (K-Means)")
st.markdown("Verileri benzerliklerine gore gruplara ayiriyoruz. Artik 3 boyutlu gorebiliyoruz!")

numeric_for_cluster = df_raw.select_dtypes(include=[np.number]).columns.tolist()

# 3 tane ozellik sectiriyorum (3D Grafik icin)
c1, c2 = st.columns([3, 1])
with c1:
    cluster_feats = st.multiselect("Kumeleme Verileri (En az 3 tane secin):", numeric_for_cluster, default=numeric_for_cluster[:3])
with c2:
    k_val = st.slider("Grup Sayisi (K)", 2, 8, 4)

if len(cluster_feats) >= 3:
    # Veriyi hazirlama
    X_cluster = df_raw[cluster_feats].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    #ELBOW Methodu (K sayısını seçmek için)
    col_elb, col_3d = st.columns([1, 2])
    
    with col_elb:
        st.subheader("Optimum Grup Sayisi (Elbow)")
        inertia = []
        K_range = range(1, 10)
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_scaled)
            inertia.append(km.inertia_)
        
        fig_elb = px.line(x=list(K_range), y=inertia, markers=True, labels={'x': 'Grup Sayisi (K)', 'y': 'Hata (Inertia)'}, title="Dirsek Yontemi")
        st.plotly_chart(fig_elb, use_container_width=True)
        st.markdown('<div class="explanation-box">Bu grafik, kac gruba ayirmaniz gerektigini soyler. Kirilmanin (dirsegin) oldugu yer en iyi sayidir.</div>', unsafe_allow_html=True)

    # Modeli calistir
    kmeans = KMeans(n_clusters=k_val, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    X_cluster['Cluster'] = clusters.astype(str)
    
    with col_3d:
        st.subheader(f"3D Kumeleme Sonuclari (K={k_val})")
        # 3D Scatter Plot
        fig_3d = px.scatter_3d(X_cluster, x=cluster_feats[0], y=cluster_feats[1], z=cluster_feats[2],
                               color='Cluster', opacity=0.7, size_max=10,
                               title=f"3D Gruplandirma: {cluster_feats[0]} x {cluster_feats[1]} x {cluster_feats[2]}")
        st.plotly_chart(fig_3d, use_container_width=True)
        st.markdown('<div class="explanation-box">Veriyi 3 boyutlu uzayda dondurerek gruplarin nasil ayristigini inceleyebilirsiniz. Ayni renkler benzer musterilerdir.</div>', unsafe_allow_html=True)

else:
    st.warning("3 Boyutlu grafik icin lutfen yukaridan en az 3 tane veri secin.")