# 📊 Crypto Sentiment Analysis

**Social Media Sentiment Analysis to Predict Cryptocurrency Asset Price Volatility**

Aplikasi web interaktif untuk menganalisis sentimen dari social media tweets terkait cryptocurrency dan memprediksi volatilitas harga aset kripto.

## 🎯 Fitur Utama

- 📊 **Dashboard Overview**: Visualisasi data tweets dan sentimen
- 🧹 **Data Preprocessing**: Pembersihan dan preprocessing data
- 🤖 **Model Analysis**: Analisis performa model machine learning
- 📈 **Price Analysis**: Analisis harga dan volatilitas cryptocurrency
- 🔗 **Correlation Analysis**: Analisis korelasi sentimen vs volatilitas
- 📝 **About**: Informasi lengkap tentang proyek

## 🛠️ Teknologi yang Digunakan

- **Python 3.8+**
- **Streamlit**: Framework web application
- **Pandas & NumPy**: Data manipulation dan analysis
- **Scikit-learn**: Machine learning models
- **Plotly**: Interactive visualizations
- **YFinance**: Cryptocurrency price data
- **Hugging Face Datasets**: Dataset tweets cryptocurrency

## 🚀 Cara Menjalankan Lokal

### 1. Clone Repository
```bash
git clone <repository-url>
cd nlp_project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Jalankan Aplikasi
```bash
streamlit run streamlit_app.py
```

Aplikasi akan berjalan di `http://localhost:8501`

## 🌐 Deployment Online

### Streamlit Cloud (Gratis)
1. Upload kode ke GitHub
2. Kunjungi [share.streamlit.io](https://share.streamlit.io)
3. Connect dengan repository GitHub
4. Deploy otomatis

### Render (Gratis)
1. Buat akun di [render.com](https://render.com)
2. Connect dengan repository GitHub
3. Pilih "Web Service"
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `streamlit run streamlit_app.py`

### Railway (Gratis)
1. Buat akun di [railway.app](https://railway.app)
2. Connect dengan repository GitHub
3. Deploy otomatis

## 📊 Dataset

- **Source**: StephanAkkerman/financial-tweets-crypto (Hugging Face)
- **Size**: ~58,000 cryptocurrency-related tweets
- **Features**: Tweet text, sentiment labels, timestamps

## 🤖 Model Machine Learning

1. **Baseline Model**: TF-IDF + Multinomial Naive Bayes
2. **Evaluation Metrics**: F1-Score, Classification Report, Confusion Matrix

## 📈 Analisis yang Tersedia

- Real-time sentiment analysis
- Price volatility calculation
- Correlation analysis
- Time series visualization
- Statistical significance testing

## 👨‍💻 Author

**Fajar Triady Putra**  
Fakultas Sains dan Teknologi  
Universitas Al Azhar Indonesia

### 📧 Contact Information
- **Email**: fajar.triady@outlook.com
- **LinkedIn**: [https://www.linkedin.com/in/fajartriadyp/](https://www.linkedin.com/in/fajartriadyp/)
- **GitHub**: [https://github.com/fajartriadyp](https://github.com/fajartriadyp)

## 📚 Academic Context

Proyek ini dikembangkan sebagai bagian dari tugas akhir mata kuliah Natural Language Processing (NLP), dengan fokus pada aplikasi sentiment analysis di pasar keuangan.

## 📄 License

MIT License - lihat file [LICENSE](LICENSE) untuk detail lebih lanjut.

## 🤝 Contributing

Kontribusi selalu diterima! Silakan buat pull request atau buka issue untuk saran dan perbaikan.

---

⭐ **Jika proyek ini membantu, jangan lupa untuk memberikan star!** 