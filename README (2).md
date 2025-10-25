# 🎥 YouTube Monetization Modeler

A Streamlit-based machine learning dashboard for analyzing, modeling, and predicting YouTube ad revenue.

## 📘 Overview
The YouTube Monetization Modeler enables data analysts and ML practitioners to:
- Perform interactive EDA on YouTube datasets.
- Train regression models (Linear, Ridge, Lasso, Random Forest, Gradient Boosting).
- Predict ad revenue and visualize feature importances.
- Export cleaned data and trained models.

## ⚙️ Installation
```bash
git clone https://github.com/yourusername/youtube-monetization-modeler.git
cd youtube-monetization-modeler
pip install -r requirements.txt
streamlit run app.py
```

Access the app locally at [http://localhost:8501](http://localhost:8501)

## 🚀 Features
- Automated preprocessing and feature engineering
- Interactive visualizations with Plotly
- Model performance evaluation (R², RMSE, MAE)
- One-click CSV and model export

## 📊 Tech Stack
- **Python 3.9+**
- **Streamlit**
- **scikit-learn**
- **pandas, numpy**
- **Plotly Express**
- **joblib**

## 📈 Evaluation Metrics
| Metric | Description |
|---------|-------------|
| R² | Variance explained by the model |
| RMSE | Root Mean Squared Error |
| MAE | Mean Absolute Error |

## 🧠 Example Use
1. Upload your dataset or use sample.
2. Explore trends (views vs revenue, daily totals).
3. Train and evaluate regression models.
4. Predict new video revenue interactively.
5. Export cleaned data or trained model.

## 📜 License
MIT License © 2025 Kishore Kumar
