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

import sys
st.write("Python version:", sys.version)

warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Crypto Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.8rem;
        border-left: 6px solid #1f77b4;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-card h3 {
        color: #333333;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .metric-card h2 {
        color: #1f77b4;
        font-size: 2rem;
        font-weight: bold;
        margin: 0;
    }
    .success-metric {
        border-left-color: #28a745;
        background: linear-gradient(135deg, #ffffff 0%, #f8fff9 100%);
    }
    .success-metric h2 {
        color: #28a745;
    }
    .warning-metric {
        border-left-color: #ffc107;
        background: linear-gradient(135deg, #ffffff 0%, #fffef8 100%);
    }
    .warning-metric h2 {
        color: #ffc107;
    }
    .danger-metric {
        border-left-color: #dc3545;
        background: linear-gradient(135deg, #ffffff 0%, #fff8f8 100%);
    }
    .danger-metric h2 {
        color: #dc3545;
    }
    .info-metric {
        border-left-color: #17a2b8;
        background: linear-gradient(135deg, #ffffff 0%, #f8fcff 100%);
    }
    .info-metric h2 {
        color: #17a2b8;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_crypto_dataset():
    """Load dataset cryptocurrency tweets"""
    try:
        raw_dataset = load_dataset("StephanAkkerman/financial-tweets-crypto")
        df = raw_dataset['train'].to_pandas()
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

def clean_tweet(text):
    """Fungsi pembersihan teks tweet"""
    if not isinstance(text, str):
        return ""
    
    # Lowercase conversion
    text = text.lower()
    
    # Removal of URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Removal of user mentions dan hashtag
    text = re.sub(r'\@\w+|\#','', text)
    
    # Removal of special characters, hanya menyisakan huruf dan spasi
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Menghapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

@st.cache_data
def preprocess_data(df):
    """Preprocess data untuk analisis"""
    # Hapus baris dengan nilai kosong
    df_clean = df.dropna(subset=['description', 'sentiment'])
    
    # Filter hanya sentimen yang valid
    valid_sentiments = ['Bullish', 'Bearish', 'Neutral']
    df_clean = df_clean[df_clean['sentiment'].isin(valid_sentiments)]
    
    # Bersihkan teks
    df_clean['cleaned_text'] = df_clean['description'].apply(clean_tweet)
    
    return df_clean

@st.cache_data
def train_baseline_model(df):
    """Train baseline model TF-IDF + Naive Bayes"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'],
        df['sentiment'],
        test_size=0.2,
        random_state=42,
        stratify=df['sentiment']
    )
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train model
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    
    # Predictions
    y_pred = nb_model.predict(X_test_tfidf)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
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
def get_crypto_price_data(symbol='BTC-USD', start_date='2023-01-01', end_date='2024-02-01'):
    """Download data harga cryptocurrency"""
    try:
        price_data = yf.download(symbol, start=start_date, end=end_date)
        price_data = price_data.reset_index()
        price_data.set_index('Date', inplace=True)
        
        # Flatten MultiIndex columns
        if isinstance(price_data.columns, pd.MultiIndex):
            price_data.columns = ['_'.join(col).strip() for col in price_data.columns.values]
        
        # Calculate volatility
        price_data['log_return'] = np.log(price_data['Close_BTC-USD'] / price_data['Close_BTC-USD'].shift(1))
        price_data['volatility'] = price_data['log_return'].rolling(window=7).std()
        price_data.dropna(inplace=True)
        
        return price_data
    except Exception as e:
        st.error(f"Error downloading price data: {e}")
        return None

def calculate_daily_sentiment(df):
    """Hitung sentimen harian"""
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
    
    return daily_sentiment

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Crypto Sentiment Analysis</h1>', unsafe_allow_html=True)
    st.markdown("**Social Media Sentiment Analysis to Predict Cryptocurrency Asset Price Volatility**")
    st.markdown("*Penulis: Fajar Triady Putra - Fakultas Sains dan Teknologi, Universitas Al Azhar Indonesia*")
    
    # Sidebar
    st.sidebar.title("⚙️ Pengaturan")
    
    # Load data
    with st.spinner("Memuat dataset cryptocurrency..."):
        df_raw = load_crypto_dataset()
    
    if df_raw is None:
        st.error("Gagal memuat dataset. Silakan coba lagi.")
        return
    
    # Preprocess data
    with st.spinner("Memproses data..."):
        df_processed = preprocess_data(df_raw)
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Dashboard", 
        "🧹 Data Preprocessing", 
        "🤖 Model Analysis", 
        "📈 Price Analysis", 
        "🔗 Correlation Analysis",
        "📝 About"
    ])
    
    # Tab 1: Dashboard
    with tab1:
        st.header("📊 Dashboard Overview")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card info-metric">
                <h3>Total Tweets</h3>
                <h2>{len(df_raw):,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card warning-metric">
                <h3>Valid Tweets</h3>
                <h2>{len(df_processed):,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            sentiment_counts = df_processed['sentiment'].value_counts()
            st.markdown(f"""
            <div class="metric-card success-metric">
                <h3>Bullish Tweets</h3>
                <h2>{sentiment_counts.get('Bullish', 0):,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card danger-metric">
                <h3>Bearish Tweets</h3>
                <h2>{sentiment_counts.get('Bearish', 0):,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Sentiment Distribution Chart
        st.subheader("📊 Distribusi Sentimen")
        
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Distribusi Sentimen Tweets",
            color_discrete_map={'Bullish': '#28a745', 'Bearish': '#dc3545', 'Neutral': '#6c757d'}
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
        # Sample Data
        st.subheader("📋 Sample Data")
        st.dataframe(df_processed[['description', 'sentiment', 'timestamp']].head(10))
    
    # Tab 2: Data Preprocessing
    with tab2:
        st.header("🧹 Data Preprocessing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Statistik Data")
            
            # Data quality metrics
            missing_data = df_raw.isnull().sum()
            st.write("**Missing Values:**")
            st.dataframe(missing_data[missing_data > 0])
            
            st.write("**Data Types:**")
            st.dataframe(df_raw.dtypes)
        
        with col2:
            st.subheader("🧹 Text Cleaning")
            
            # Show before and after cleaning
            sample_data = df_processed[['description', 'cleaned_text']].head(5)
            st.write("**Before vs After Cleaning:**")
            st.dataframe(sample_data)
            
            # Text length distribution
            df_processed['text_length'] = df_processed['cleaned_text'].str.len()
            
            fig = px.histogram(
                df_processed, 
                x='text_length',
                title="Distribusi Panjang Teks Setelah Cleaning",
                nbins=50
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Model Analysis
    with tab3:
        st.header("🤖 Model Analysis")
        
        # Train baseline model
        with st.spinner("Training baseline model..."):
            model_results = train_baseline_model(df_processed)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Model Performance")
            
            # F1 Score
            st.metric("F1 Score (Macro)", f"{model_results['f1_score']:.4f}")
            
            # Classification Report
            st.write("**Classification Report:**")
            report_df = pd.DataFrame(model_results['classification_report']).transpose()
            st.dataframe(report_df)
        
        with col2:
            st.subheader("📊 Confusion Matrix")
            
            # Confusion Matrix
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
                title="Confusion Matrix"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Sample predictions
        st.subheader("🔮 Sample Predictions")
        sample_predictions = pd.DataFrame({
            'Text': model_results['X_test'].head(10),
            'Actual': model_results['y_test'].head(10),
            'Predicted': model_results['y_pred'][:10]
        })
        st.dataframe(sample_predictions)
    
    # Tab 4: Price Analysis
    with tab4:
        st.header("📈 Price Analysis")
        
        # Crypto selection
        crypto_symbol = st.selectbox(
            "Pilih Cryptocurrency:",
            ['BTC-USD', 'ETH-USD', 'ADA-USD', 'DOT-USD', 'LINK-USD']
        )
        
        # Date range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=datetime(2023, 1, 1))
        with col2:
            end_date = st.date_input("End Date", value=datetime(2024, 2, 1))
        
        # Load price data
        with st.spinner(f"Memuat data harga {crypto_symbol}..."):
            price_data = get_crypto_price_data(crypto_symbol, start_date, end_date)
        
        if price_data is not None:
            # Price chart
            st.subheader(f"📈 {crypto_symbol} Price Chart")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=price_data.index,
                y=price_data['Close_BTC-USD'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue')
            ))
            fig.update_layout(
                title=f"{crypto_symbol} Price Over Time",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Volatility chart
            st.subheader("📊 Volatility Analysis")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=price_data.index,
                y=price_data['volatility'],
                mode='lines',
                name='Volatility',
                line=dict(color='red')
            ))
            fig.update_layout(
                title="Price Volatility Over Time",
                xaxis_title="Date",
                yaxis_title="Volatility",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Price statistics
            st.subheader("📊 Price Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"${price_data['Close_BTC-USD'].iloc[-1]:.2f}")
            with col2:
                st.metric("Max Price", f"${price_data['Close_BTC-USD'].max():.2f}")
            with col3:
                st.metric("Min Price", f"${price_data['Close_BTC-USD'].min():.2f}")
            with col4:
                st.metric("Avg Volatility", f"{price_data['volatility'].mean():.4f}")
    
    # Tab 5: Correlation Analysis
    with tab5:
        st.header("🔗 Correlation Analysis")
        
        # Calculate daily sentiment
        with st.spinner("Menghitung sentimen harian..."):
            daily_sentiment = calculate_daily_sentiment(df_processed)
        
        # Get price data for correlation
        price_data = get_crypto_price_data('BTC-USD', '2023-01-01', '2024-02-01')
        
        if price_data is not None and len(daily_sentiment) > 0:
            # Merge data
            df_merged = price_data[['Close_BTC-USD', 'volatility']].merge(
                daily_sentiment[['sentiment_score']],
                left_index=True,
                right_index=True,
                how='inner'
            )
            
            if len(df_merged) > 0:
                # Correlation analysis
                st.subheader("📊 Correlation Analysis")
                
                # Calculate correlation
                corr, p_value = pearsonr(df_merged['sentiment_score'], df_merged['volatility'])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Correlation Coefficient", f"{corr:.4f}")
                    st.metric("P-value", f"{p_value:.4f}")
                    st.metric("Significance", "Significant" if p_value < 0.05 else "Not Significant")
                
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
                
                # Regression analysis
                st.subheader("📊 Regression Analysis")
                
                # Prepare data for regression
                df_merged['sentiment_lagged'] = df_merged['sentiment_score'].shift(1)
                df_reg = df_merged.dropna(subset=['volatility', 'sentiment_lagged'])
                
                if len(df_reg) > 0:
                    Y = df_reg['volatility']
                    X = sm.add_constant(df_reg['sentiment_lagged'])
                    model = sm.OLS(Y, X).fit()
                    
                    st.write("**Regression Results:**")
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
            else:
                st.warning("Tidak ada data yang tumpang tindih untuk analisis korelasi.")
        else:
            st.error("Gagal memuat data harga atau sentimen untuk analisis korelasi.")
    
    # Tab 6: About
    with tab6:
        st.header("📝 About This Project")
        
        st.markdown("""
        ## 🎯 Project Overview
        
        **Social Media Sentiment Analysis to Predict Cryptocurrency Asset Price Volatility**
        
        ### 📋 Objectives
        - Menganalisis sentimen dari social media tweets terkait cryptocurrency
        - Membangun model machine learning untuk klasifikasi sentimen
        - Menganalisis korelasi antara sentimen dan volatilitas harga cryptocurrency
        - Mengembangkan prediktor volatilitas harga berdasarkan sentimen
        
        ### 🛠️ Technologies Used
        - **Python**: Primary programming language
        - **Streamlit**: Web application framework
        - **Pandas & NumPy**: Data manipulation and analysis
        - **Scikit-learn**: Machine learning models
        - **Transformers**: BERTweet fine-tuning
        - **Plotly**: Interactive visualizations
        - **YFinance**: Cryptocurrency price data
        
        ### 📊 Dataset
        - **Source**: StephanAkkerman/financial-tweets-crypto (Hugging Face)
        - **Size**: ~58,000 cryptocurrency-related tweets
        - **Features**: Tweet text, sentiment labels, timestamps, tweet types
        
        ### 🤖 Models
        1. **Baseline Model**: TF-IDF + Multinomial Naive Bayes
        2. **Advanced Model**: Fine-tuned BERTweet
        3. **Evaluation Metrics**: F1-Score, Classification Report, Confusion Matrix
        
        ### 📈 Analysis Features
        - Real-time sentiment analysis
        - Price volatility calculation
        - Correlation analysis
        - Time series visualization
        - Statistical significance testing
        
        ### 👨‍💻 Author
        **Fajar Triady Putra**  
        Fakultas Sains dan Teknologi  
        Universitas Al Azhar Indonesia
        
        ### 📚 Academic Context
        This project was developed as part of the Natural Language Processing (NLP) course final assignment, 
        focusing on the application of sentiment analysis in financial markets.
        """)
        
        # Contact information
        st.subheader("📧 Contact Information")
        st.markdown("""
        - **Email**: fajar.triady@outlook.com
        - **LinkedIn**: [https://www.linkedin.com/in/fajartriadyp/]
        - **GitHub**: [https://github.com/fajartriadyp]
        """)

if __name__ == "__main__":
    main() 