#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script untuk Crypto Sentiment Analysis
Menunjukkan cara menggunakan fitur-fitur utama aplikasi

Author: Fajar Triady Putra
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

def demo_sentiment_analysis():
    """Demo untuk analisis sentimen"""
    st.header("🔮 Demo Sentiment Analysis")
    
    # Sample tweets
    sample_tweets = [
        "Bitcoin is going to the moon! 🚀 $BTC",
        "Crypto market is crashing, sell everything! 📉",
        "Ethereum looks stable today, holding my position",
        "This is the end of cryptocurrency as we know it",
        "Great news for crypto investors! Bull run incoming! 🐂"
    ]
    
    # Predicted sentiments (simulasi)
    predicted_sentiments = ["Bullish", "Bearish", "Neutral", "Bearish", "Bullish"]
    
    # Create demo dataframe
    demo_df = pd.DataFrame({
        'Tweet': sample_tweets,
        'Predicted Sentiment': predicted_sentiments
    })
    
    st.write("**Sample Tweets dan Prediksi Sentimen:**")
    st.dataframe(demo_df)
    
    # Sentiment distribution
    sentiment_counts = demo_df['Predicted Sentiment'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Distribusi Sentimen Demo",
            color_discrete_map={'Bullish': '#28a745', 'Bearish': '#dc3545', 'Neutral': '#6c757d'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Statistik Sentimen:**")
        for sentiment, count in sentiment_counts.items():
            st.metric(sentiment, count)

def demo_price_analysis():
    """Demo untuk analisis harga"""
    st.header("📈 Demo Price Analysis")
    
    # Generate sample price data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)
    
    # Simulate price movement
    base_price = 30000
    price_changes = np.random.normal(0, 0.02, len(dates))
    prices = [base_price]
    
    for change in price_changes[1:]:
        prices.append(prices[-1] * (1 + change))
    
    # Calculate volatility
    returns = np.diff(np.log(prices))
    volatility = pd.Series(returns).rolling(window=7).std()
    
    # Create price dataframe
    price_df = pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'Volatility': [np.nan] + volatility.tolist()
    })
    
    # Price chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price_df['Date'],
        y=price_df['Price'],
        mode='lines',
        name='BTC Price',
        line=dict(color='blue')
    ))
    fig.update_layout(
        title="Demo Bitcoin Price Chart",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Volatility chart
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=price_df['Date'],
        y=price_df['Volatility'],
        mode='lines',
        name='Volatility',
        line=dict(color='red')
    ))
    fig2.update_layout(
        title="Demo Volatility Chart",
        xaxis_title="Date",
        yaxis_title="Volatility",
        height=400
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # Price statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Price", f"${price_df['Price'].iloc[-1]:.2f}")
    with col2:
        st.metric("Max Price", f"${price_df['Price'].max():.2f}")
    with col3:
        st.metric("Min Price", f"${price_df['Price'].min():.2f}")
    with col4:
        st.metric("Avg Volatility", f"{price_df['Volatility'].mean():.4f}")

def demo_correlation():
    """Demo untuk analisis korelasi"""
    st.header("🔗 Demo Correlation Analysis")
    
    # Generate sample sentiment and volatility data
    np.random.seed(42)
    n_days = 100
    
    # Simulate sentiment scores
    sentiment_scores = np.random.normal(0, 0.3, n_days)
    sentiment_scores = np.clip(sentiment_scores, -1, 1)
    
    # Simulate volatility with some correlation to sentiment
    volatility = np.random.normal(0.02, 0.01, n_days) + 0.1 * np.abs(sentiment_scores)
    
    # Create correlation dataframe
    corr_df = pd.DataFrame({
        'Sentiment Score': sentiment_scores,
        'Volatility': volatility
    })
    
    # Calculate correlation
    correlation = corr_df['Sentiment Score'].corr(corr_df['Volatility'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Correlation Coefficient", f"{correlation:.4f}")
        st.metric("Sample Size", n_days)
    
    with col2:
        # Scatter plot
        fig = px.scatter(
            corr_df,
            x='Sentiment Score',
            y='Volatility',
            title=f"Demo Sentiment vs Volatility (r={correlation:.4f})",
            trendline="ols"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Time series
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    time_df = pd.DataFrame({
        'Date': dates,
        'Sentiment Score': sentiment_scores,
        'Volatility': volatility
    })
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=time_df['Date'],
        y=time_df['Sentiment Score'],
        mode='lines',
        name='Sentiment Score',
        line=dict(color='blue')
    ))
    fig2.add_trace(go.Scatter(
        x=time_df['Date'],
        y=time_df['Volatility'],
        mode='lines',
        name='Volatility',
        line=dict(color='red'),
        yaxis='y2'
    ))
    fig2.update_layout(
        title="Demo Time Series Comparison",
        yaxis2=dict(overlaying='y', side='right'),
        height=400
    )
    st.plotly_chart(fig2, use_container_width=True)

def main():
    """Main demo function"""
    st.set_page_config(
        page_title="Crypto Sentiment Analysis - Demo",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 Crypto Sentiment Analysis - Demo")
    st.markdown("**Demonstrasi fitur-fitur utama aplikasi**")
    
    # Demo tabs
    tab1, tab2, tab3 = st.tabs([
        "🔮 Sentiment Analysis",
        "📈 Price Analysis", 
        "🔗 Correlation Analysis"
    ])
    
    with tab1:
        demo_sentiment_analysis()
    
    with tab2:
        demo_price_analysis()
    
    with tab3:
        demo_correlation()
    
    # Instructions
    st.sidebar.title("📋 Demo Instructions")
    st.sidebar.markdown("""
    ### Cara Menggunakan Demo:
    
    1. **Sentiment Analysis**: Melihat contoh analisis sentimen dari tweets
    2. **Price Analysis**: Visualisasi harga dan volatilitas cryptocurrency
    3. **Correlation Analysis**: Analisis korelasi antara sentimen dan volatilitas
    
    ### Untuk Aplikasi Lengkap:
    Jalankan `streamlit run app.py` untuk menggunakan aplikasi dengan data real.
    """)

if __name__ == "__main__":
    main() 