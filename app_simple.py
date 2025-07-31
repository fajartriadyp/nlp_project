import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from datetime import datetime, timedelta
import yfinance as yf
import statsmodels.api as sm
from scipy.stats import pearsonr
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Crypto Sentiment Analysis - Notebook Results",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .phase-header {
        font-size: 1.8rem;
        color: #2c3e50;
        border-left: 4px solid #3498db;
        padding-left: 1rem;
        margin: 2rem 0 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-metric {
        border-left-color: #28a745;
    }
    .warning-metric {
        border-left-color: #ffc107;
    }
    .danger-metric {
        border-left-color: #dc3545;
    }
    .info-box {
        background-color: #e3f2fd;
        border: 1px solid #2196f3;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_process_data():
    """FASE 1: Load dan process data seperti di notebook"""
    st.markdown('<h2 class="phase-header">FASE 1: DATA PREPROCESSING DAN EKSPLORASI</h2>', unsafe_allow_html=True)
    
    st.info("📥 Memuat dataset cryptocurrency tweets...")
    
    try:
        # Load dataset
        raw_dataset = load_dataset("StephanAkkerman/financial-tweets-crypto")
        df = raw_dataset['train'].to_pandas()
        
        st.success(f"✅ Dataset berhasil dimuat dengan {len(df)} tweets")
        st.write(f"📊 Kolom yang tersedia: {list(df.columns)}")
        
        # Data cleaning
        st.info("🧹 Melakukan pembersihan data...")
        df_original_size = len(df)
        df.dropna(subset=['description', 'sentiment'], inplace=True)
        st.write(f"📝 Menghapus {df_original_size - len(df)} baris dengan nilai kosong")
        
        # Filter valid sentiments
        valid_sentiments = ['Bullish', 'Bearish', 'Neutral']
        df = df[df['sentiment'].isin(valid_sentiments)]
        st.write(f"📊 Data setelah filtering: {len(df)} tweets")
        
        # Text cleaning function
        def clean_tweet(text):
            if not isinstance(text, str):
                return ""
            text = text.lower()
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            text = re.sub(r'\@\w+|\#','', text)
            text = re.sub(r'[^a-z\s]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        df['cleaned_text'] = df['description'].apply(clean_tweet)
        
        st.success("✅ Pembersihan teks selesai")
        
        return df, df_original_size
        
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        return None, 0

@st.cache_data
def train_baseline_model(df):
    """FASE 2: Train baseline model seperti di notebook"""
    st.markdown('<h2 class="phase-header">FASE 2: MODEL BASELINE (TF-IDF + NAIVE BAYES)</h2>', unsafe_allow_html=True)
    
    st.info("🤖 Membangun model baseline menggunakan TF-IDF + Multinomial Naive Bayes...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'],
        df['sentiment'],
        test_size=0.2,
        random_state=42,
        stratify=df['sentiment']
    )
    
    st.write(f"📊 Data training: {len(X_train)} tweets")
    st.write(f"📊 Data testing: {len(X_test)} tweets")
    
    # TF-IDF Vectorization
    st.info("🔢 Membuat representasi TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    st.write(f"✅ TF-IDF matrix shape: {X_train_tfidf.shape}")
    
    # Train model
    st.info("🎯 Melatih model Naive Bayes...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    
    # Predictions
    st.info("📈 Melakukan prediksi dan evaluasi...")
    y_pred = nb_model.predict(X_test_tfidf)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    st.success(f"✅ Model training selesai. F1-Score (Macro): {f1_macro:.4f}")
    
    return {
        'model': nb_model,
        'vectorizer': vectorizer,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'f1_score': f1_macro,
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }

@st.cache_data
def get_crypto_price_data():
    """FASE 5: Download data harga cryptocurrency seperti di notebook"""
    st.markdown('<h2 class="phase-header">FASE 5: ANALISIS KORELASI DENGAN VOLATILITAS HARGA</h2>', unsafe_allow_html=True)
    
    st.info("📈 Mengunduh data harga historis BTC-USD...")
    
    try:
        btc_price = yf.download('BTC-USD', start='2023-01-01', end='2024-02-01')
        btc_price = btc_price.reset_index()
        btc_price.set_index('Date', inplace=True)
        
        # Flatten MultiIndex columns
        if isinstance(btc_price.columns, pd.MultiIndex):
            btc_price.columns = ['_'.join(col).strip() for col in btc_price.columns.values]
        
        st.success(f"✅ Data harga berhasil diunduh: {len(btc_price)} hari")
        st.write(f"📊 Kolom data harga: {list(btc_price.columns)}")
        
        # Calculate volatility
        st.info("📊 Menghitung volatilitas harga...")
        btc_price['log_return'] = np.log(btc_price['Close_BTC-USD'] / btc_price['Close_BTC-USD'].shift(1))
        btc_price['volatility'] = btc_price['log_return'].rolling(window=7).std()
        btc_price.dropna(inplace=True)
        
        st.success(f"✅ Volatilitas berhasil dihitung untuk {len(btc_price)} hari")
        return btc_price
        
    except Exception as e:
        st.error(f"❌ Error downloading price data: {e}")
        return None

def calculate_daily_sentiment(df):
    """Hitung sentimen harian seperti di notebook"""
    st.info("📊 Mengagregasi sentimen per hari...")
    
    # Parse timestamp
    df['date'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True).dt.date
    df['date'] = pd.to_datetime(df['date'])
    
    # Hitung sentiment score harian
    daily_sentiment = pd.crosstab(df['date'], df['sentiment'])
    daily_sentiment['sentiment_score'] = (
        daily_sentiment.get('Bullish', 0) - daily_sentiment.get('Bearish', 0)
    ) / (
        daily_sentiment.get('Bullish', 0) +
        daily_sentiment.get('Bearish', 0) +
        daily_sentiment.get('Neutral', 0)
    )
    
    st.success(f"✅ Sentimen harian berhasil dihitung untuk {len(daily_sentiment)} hari")
    return daily_sentiment

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Crypto Sentiment Analysis - Notebook Results</h1>', unsafe_allow_html=True)
    st.markdown("**Social Media Sentiment Analysis to Predict Cryptocurrency Asset Price Volatility**")
    st.markdown("*Penulis: Fajar Triady Putra - Fakultas Sains dan Teknologi, Universitas Al Azhar Indonesia*")
    
    # Sidebar
    st.sidebar.title("⚙️ Pengaturan")
    
    # Load dan process data
    df_processed, df_original_size = load_and_process_data()
    
    if df_processed is None:
        st.error("Gagal memuat dataset. Silakan coba lagi.")
        return
    
    # Analisis Distribusi Sentimen (FASE 1)
    st.subheader("📊 Distribusi sentimen dalam dataset:")
    sentiment_counts = df_processed['sentiment'].value_counts()
    st.write(sentiment_counts)
    
    # Visualisasi distribusi sentimen
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            title="Distribusi Sentimen Tweets",
            color=sentiment_counts.index,
            color_discrete_map={'Bullish': '#28a745', 'Bearish': '#dc3545', 'Neutral': '#6c757d'}
        )
        fig.update_layout(xaxis_title="Sentimen", yaxis_title="Jumlah Tweets")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Proporsi Sentimen Tweets",
            color_discrete_map={'Bullish': '#28a745', 'Bearish': '#dc3545', 'Neutral': '#6c757d'}
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    # Sample data
    st.subheader("📋 Contoh data setelah dibersihkan:")
    sample_data = df_processed[['description', 'cleaned_text', 'sentiment']].head()
    st.dataframe(sample_data)
    
    # Train baseline model
    model_results = train_baseline_model(df_processed)
    
    # Hasil evaluasi model baseline
    st.subheader("HASIL EVALUASI MODEL BASELINE")
    st.write("="*50)
    
    # Classification Report
    report_df = pd.DataFrame(model_results['classification_report']).transpose()
    st.dataframe(report_df)
    
    st.write(f"🎯 F1-Score (Macro) Baseline: {model_results['f1_score']:.4f}")
    st.write("="*50)
    
    # Confusion Matrix
    st.subheader("📊 Confusion Matrix")
    cm = confusion_matrix(model_results['y_test'], model_results['y_pred'])
    cm_df = pd.DataFrame(
        cm, 
        index=['Actual Bullish', 'Actual Bearish', 'Actual Neutral'],
        columns=['Predicted Bullish', 'Predicted Bearish', 'Predicted Neutral']
    )
    
    fig = px.imshow(
        cm_df,
        text_auto=True,
        aspect="auto",
        title="Confusion Matrix - Baseline Model"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Get price data
    price_data = get_crypto_price_data()
    
    if price_data is not None:
        # Calculate daily sentiment
        daily_sentiment = calculate_daily_sentiment(df_processed)
        
        # Merge data
        st.info("🔗 Menggabungkan data harga dan skor sentimen...")
        df_merged = price_data[['Close_BTC-USD', 'volatility']].merge(
            daily_sentiment[['sentiment_score']],
            left_index=True,
            right_index=True,
            how='inner'
        )
        
        st.success(f"✅ Data berhasil digabungkan: {len(df_merged)} hari")
        
        if len(df_merged) > 0:
            st.subheader("📋 Contoh data gabungan:")
            st.dataframe(df_merged.head())
            
            # Correlation analysis
            st.subheader("ANALISIS KORELASI SENTIMEN DAN VOLATILITAS")
            st.write("="*50)
            
            # Calculate correlation
            corr, p_value = pearsonr(df_merged['sentiment_score'], df_merged['volatility'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Korelasi Skor Original", f"{corr:.4f}")
                st.metric("P-value", f"{p_value:.4f}")
                significance = "signifikan secara statistik" if p_value < 0.05 else "tidak signifikan secara statistik"
                st.metric("Signifikansi Statistik", significance)
            
            with col2:
                # Scatter plot
                fig = px.scatter(
                    df_merged,
                    x='sentiment_score',
                    y='volatility',
                    title=f"Sentiment vs Volatility Correlation (r={corr:.4f})",
                    trendline="ols"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Regression analysis
            st.subheader("ANALISIS REGRESI PREDIKTIF")
            st.write("="*50)
            
            # Prepare data for regression
            df_merged['sentiment_lagged'] = df_merged['sentiment_score'].shift(1)
            df_reg = df_merged.dropna(subset=['volatility', 'sentiment_lagged'])
            
            if len(df_reg) > 0:
                Y = df_reg['volatility']
                X = sm.add_constant(df_reg['sentiment_lagged'])
                model = sm.OLS(Y, X).fit()
                
                st.write("**Hasil Regresi Linear (Skor Original):**")
                st.text(str(model.summary()))
                
                # Residual plot
                fig = px.scatter(
                    x=model.fittedvalues,
                    y=model.resid,
                    title="Residual Plot",
                    labels={'x': 'Fitted Values', 'y': 'Residuals'}
                )
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
            
            # Time series comparison
            st.subheader("📈 Time Series Comparison")
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Sentiment Score Over Time', 'Volatility Over Time'),
                vertical_spacing=0.1
            )
            
            fig.add_trace(
                go.Scatter(x=df_merged.index, y=df_merged['sentiment_score'], name='Sentiment Score'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=df_merged.index, y=df_merged['volatility'], name='Volatility'),
                row=2, col=1
            )
            
            fig.update_layout(height=600, title_text="Sentiment and Volatility Time Series")
            st.plotly_chart(fig, use_container_width=True)
    
    # Kesimpulan dan Ringkasan
    st.markdown('<h2 class="phase-header">KESIMPULAN DAN RINGKASAN PROYEK</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>🎯 RINGKASAN HASIL PROYEK:</h3>
        
        <h4>1. DATA PREPROCESSING:</h4>
        <ul>
            <li>✅ Total tweets diproses: {:,}</li>
            <li>✅ Tweets dengan sentimen valid: {:,}</li>
        </ul>
        
        <h4>2. MODEL PERFORMANCE:</h4>
        <ul>
            <li>📊 Baseline F1-Score (TF-IDF + Naive Bayes): {:.4f}</li>
        </ul>
        
        <h4>3. ANALISIS PREDIKTIF:</h4>
        <ul>
            <li>📊 Korelasi Sentimen-Volatilitas: {:.4f}</li>
            <li>📈 Signifikansi Statistik: {}</li>
            <li>📅 Periode analisis gabungan: {} hari</li>
        </ul>
        
        <h4>4. KONTRIBUSI ILMIAH:</h4>
        <ul>
            <li>✅ Implementasi analisis sentimen untuk domain cryptocurrency</li>
            <li>✅ Metodologi yang dapat direplikasi untuk aset crypto lainnya</li>
        </ul>
    </div>
    """.format(
        df_original_size,
        len(df_processed),
        model_results['f1_score'],
        corr if 'corr' in locals() else 0,
        significance if 'significance' in locals() else "N/A",
        len(df_merged) if 'df_merged' in locals() else 0
    ), unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎓 Academic Context
    Proyek ini dikembangkan sebagai tugas akhir mata kuliah Natural Language Processing (NLP)
    yang fokus pada aplikasi analisis sentimen dalam pasar finansial.
    
    ### 👨‍💻 Author
    **Fajar Triady Putra**  
    Fakultas Sains dan Teknologi  
    Universitas Al Azhar Indonesia
    """)
    
    st.success("🎉 PROYEK SELESAI DENGAN SUKSES!")

if __name__ == "__main__":
    main() 