import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from utils import set_design

# Prophet kütüphanesi biraz nazlıdır, bazen sunucularda yüklü olmaz.
# O yüzden kod patlamasın diye try-except bloğu içine aldım.
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except:
    PROPHET_AVAILABLE = False

# Sayfa ayarlarını yapıyorum, geniş ekran olsun ki grafikler sığsın
st.set_page_config(page_title="Zaman Serisi Analizi", layout="wide")
set_design()

st.title("Zaman Serisi ve Gelecek Tahmini")
st.markdown("""
Burada sadece geleceği tahmin etmiyoruz; aynı zamanda geçmişte neler olduğunu, 
hangi günlerde satışların arttığını ve genel trendin nereye gittiğini analiz ediyoruz.
""")

# Veriyi Session State'den (Hafızadan) çekiyorum. 
# Eğer kullanıcı ana sayfadan veri yüklemediyse hata verip durduruyorum.
if 'df_raw' not in st.session_state:
    st.warning("Veriler yüklenmemiş. Lütfen önce Ana Sayfa'ya gidip verileri yükleyin.")
    st.stop()

df_raw = st.session_state['df_raw']
target_col = st.session_state.get('target_col', 'base_price')

# Tarih kolonu var mı diye kontrol ediyorum, yoksa zaman serisi yapamayız çünkü :)
if "order_date" in df_raw.columns:
    
    # Veriyi tarihe göre grupluyorum. Günlük ortalama değerleri alıyorum.
    # Prophet kütüphanesi kolon isimlerini 'ds' (tarih) ve 'y' (değer) olarak istiyor, o yüzden rename yaptım.
    ts_data = df_raw.groupby("order_date")[target_col].mean().reset_index()
    ts_data.columns = ['ds', 'y']
    ts_data = ts_data.sort_values('ds') # Tarihleri sıraya dizdim, karışık olmasın.

    # --- AYARLAR PANELI ---
    col_set1, col_set2, col_set3 = st.columns([1, 1, 1])
    
    with col_set1:
        # Kullanıcıya model seçtiriyorum. Prophet daha zeki ama Holt-Winters daha hızlıdır.
        method = st.selectbox("Tahmin Modeli:", ["Prophet (Yapay Zeka)", "Holt-Winters (İstatistiksel)"])
    
    with col_set2:
        # Kaç gün sonrasını merak ediyoruz?
        days = st.slider("Tahmin Ufku (Gün):", 7, 180, 30)
        
    with col_set3:
        # Butonu biraz aşağı indirmek için boşluk bıraktım
        st.write("") 
        btn_run = st.button("Analizi Başlat", use_container_width=True)

    if btn_run:
        st.divider()
        
        # --- SENARYO 1: PROPHET (AI TABANLI) ---
        if method == "Prophet (Yapay Zeka)" and PROPHET_AVAILABLE:
            with st.spinner("Yapay zeka veriyi öğreniyor..."):
                # Modeli başlatıyorum
                m = Prophet()
                # Modeli eğitiyorum (fit)
                m.fit(ts_data)
                
                # Gelecek için boş bir tablo oluşturuyorum
                future = m.make_future_dataframe(periods=days)
                # Tahminleri yapıyorum
                forecast = m.predict(future)
                
                # 1. ANA GRAFIK (TAHMIN)
                st.subheader(f"🔮 Gelecek {days} Günlük Tahmin")
                
                fig_main = go.Figure()
                
                # Geçmiş veriyi siyah nokta olarak ekliyorum
                fig_main.add_trace(go.Scatter(x=ts_data['ds'], y=ts_data['y'], name='Gerçek Veri', mode='markers', marker=dict(color='gray', size=4)))
                
                # Tahmin çizgisini ekliyorum
                fig_main.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Tahmin', line=dict(color='#2ecc71', width=3)))
                
                # Güven aralığı (Confidence Interval) - Yani AI diyor ki: "Tam burası olmayabilir ama şu aralıkta olacağından eminim"
                fig_main.add_trace(go.Scatter(
                    x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
                    y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
                    fill='toself',
                    fillcolor='rgba(46, 204, 113, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Güven Aralığı'
                ))
                
                # Grafiğe zum yapabilme özelliği ekliyorum (Rangeslider)
                fig_main.update_layout(xaxis_rangeslider_visible=True, height=500)
                st.plotly_chart(fig_main, use_container_width=True)
                
                st.markdown("""
                <div class="explanation-box">
                    <b>Junior Notu:</b> Yeşil çizgi yapay zekanın tahmini. Etrafındaki açık yeşil alan ise 'Hata Payı'. 
                    Geleceğe gittikçe bu alanın genişlediğini görebilirsiniz, çünkü zaman ilerledikçe belirsizlik artar.
                </div>
                """, unsafe_allow_html=True)

                # 2. BILESENLER (TREND VE MEVSIMSELLIK)
                st.subheader("📊 Verinin Röntgeni (Trend ve Mevsimsellik)")
                col_c1, col_c2 = st.columns(2)
                
                with col_c1:
                    # Genel Trend Grafiği
                    fig_trend = px.line(forecast, x='ds', y='trend', title="Genel Trend (Yükseliyor mu?)")
                    fig_trend.update_traces(line_color='orange')
                    st.plotly_chart(fig_trend, use_container_width=True)
                    st.info("Bu grafik, günlük dalgalanmaları yok sayarak işlerin genel olarak iyiye mi kötüye mi gittiğini gösterir.")

                with col_c2:
                    # Haftalık Mevsimsellik
                    if 'weekly' in forecast.columns:
                        # Haftanın günlerini sıralamak için biraz pandas büyüsü yapıyorum
                        weekly_data = forecast.groupby(forecast['ds'].dt.day_name())['weekly'].mean().reindex(
                            ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                        ).reset_index()
                        
                        fig_week = px.bar(weekly_data, x='ds', y='weekly', title="Haftalık Etki (Hangi Gün Daha İyi?)", 
                                          color='weekly', color_continuous_scale='RdBu')
                        st.plotly_chart(fig_week, use_container_width=True)
                        st.info("Sıfırın üzerindeki günler satışların arttığı, altındakiler ise düştüğü günlerdir.")

        # --- SENARYO 2: HOLT-WINTERS (ISTATISTIKSEL) ---
        else:
            try:
                # Veriyi zaman serisi formatına çeviriyorum (index tarih olmalı)
                ts_indexed = ts_data.set_index('ds').asfreq('D').fillna(method='ffill')
                
                # Modeli kuruyorum. Trend ve Mevsimselliği (Seasonal) toplamsal (add) olarak ayarladım.
                # seasonal_periods=7 yaptım çünkü haftalık döngü olduğunu varsayıyorum.
                model_hw = ExponentialSmoothing(ts_indexed['y'], trend='add', seasonal='add', seasonal_periods=7).fit()
                pred = model_hw.forecast(days)
                
                # 1. TAHMIN GRAFIGI
                st.subheader(f"📈 Holt-Winters Tahmini ({days} Gün)")
                fig_hw = go.Figure()
                # Son 90 günü gösteriyorum ki grafik çok sıkışmasın
                fig_hw.add_trace(go.Scatter(x=ts_indexed.index[-90:], y=ts_indexed['y'][-90:], name="Geçmiş (Son 3 Ay)"))
                fig_hw.add_trace(go.Scatter(x=pred.index, y=pred.values, name="Tahmin", line=dict(color='orange', width=3, dash='dot')))
                st.plotly_chart(fig_hw, use_container_width=True)
                
                # 2. AYRISIM (DECOMPOSITION)
                st.subheader("🧩 Zaman Serisi Ayrışımı")
                # Statsmodels kütüphanesi ile veriyi parçalarına ayırıyorum
                decomposition = seasonal_decompose(ts_indexed['y'].dropna(), model='additive', period=7)
                
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    # Trend Bileşeni
                    fig_dec_trend = px.line(decomposition.trend, title="Trend Bileşeni")
                    st.plotly_chart(fig_dec_trend, use_container_width=True)
                
                with col_d2:
                    # Mevsimsellik Bileşeni
                    # Sadece son 30 günü alıyorum ki döngü net görünsün
                    season_last_30 = decomposition.seasonal[-30:]
                    fig_dec_seas = px.line(season_last_30, title="Mevsimsellik (Son 1 Ay)")
                    st.plotly_chart(fig_dec_seas, use_container_width=True)

                st.markdown("""
                <div class="explanation-box">
                    <b>Junior Notu:</b> Holt-Winters klasik ama güçlü bir yöntemdir. Prophet kütüphanesi yüklü değilse devreye girer. 
                    Veriyi Trend (Eğilim) ve Seasonality (Tekrarlayan Hareketler) olarak ikiye ayırıp analiz ettim.
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Holt-Winters modelinde bir hata oluştu: {e}")
                st.warning("Veri setinizdeki tarih aralığı çok kısa olabilir veya çok fazla eksik veri (NaN) olabilir.")

else:
    st.error("Veri setinde 'order_date' (Tarih) kolonu bulunamadı. Zaman serisi analizi yapılamaz.")