import io
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance

st.set_page_config(page_title="YouTube Monetization Modeler", layout="wide")

# ------------------------------
# Helpers
# ------------------------------
NUM_COLS_BASE = [
    "views", "likes", "comments", "watch_time_minutes",
    "video_length_minutes", "subscribers"
]
CAT_COLS = ["category", "device", "country"]
TARGET = "ad_revenue_usd"

def logical_caps(df: pd.DataFrame) -> pd.DataFrame:
    # Likes/comments cannot exceed views
    df["likes"] = np.minimum(df["likes"], df["views"])
    df["comments"] = np.minimum(df["comments"], df["views"])
    # Avg watch per view cannot exceed video length
    avg_watch = df["watch_time_minutes"] / df["views"].replace(0, np.nan)
    mask = avg_watch > df["video_length_minutes"]
    df.loc[mask, "watch_time_minutes"] = (
        df.loc[mask, "video_length_minutes"] * df.loc[mask, "views"]
    )
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["engagement_rate"] = (df["likes"] + df["comments"]) / df["views"].replace(0, np.nan)
    df["engagement_rate"] = df["engagement_rate"].clip(0, 1)
    df["avg_watch_time_per_view_min"] = df["watch_time_minutes"] / df["views"].replace(0, np.nan)
    df["rpm_usd"] = df[TARGET] / (df["views"] / 1000).replace(0, np.nan)
    return df

def clean_dataset(df: pd.DataFrame, imputation="median", do_caps=True) -> pd.DataFrame:
    df = df.copy()
    # Dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    # Deduplicate (video_id+date)
    if {"video_id", "date"}.issubset(df.columns):
        df = df.sort_values(["video_id", "date"]).drop_duplicates(["video_id", "date"], keep="last")
    # Numeric coercion
    for c in NUM_COLS_BASE + [TARGET, "rpm_usd"] if "rpm_usd" in df.columns else []:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Impute likes/comments/watch_time
    imp_val = "median" if imputation == "median" else "mean"
    for col in ["likes", "comments", "watch_time_minutes"]:
        if col in df.columns:
            fill = df[col].median() if imp_val == "median" else df[col].mean()
            df[col] = df[col].fillna(fill)
    # Clip negatives
    for c in ["views","likes","comments","watch_time_minutes","video_length_minutes",TARGET]:
        if c in df.columns:
            df[c] = df[c].clip(lower=0)
    # Avoid zero length for division safety
    if "video_length_minutes" in df.columns:
        df["video_length_minutes"] = df["video_length_minutes"].where(df["video_length_minutes"] > 0, 0.01)
    if do_caps:
        df = logical_caps(df)
    # Features
    df = engineer_features(df)
    # Replace infinities
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

def build_pipeline(model_name: str):
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.001),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }
    model = models[model_name]
    numeric_features = NUM_COLS_BASE + ["engagement_rate", "avg_watch_time_per_view_min"]
    categorical_features = CAT_COLS

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numeric_features),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical_features),
        ],
        remainder="drop"
    )

    pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
    return pipe, numeric_features, categorical_features

def kpi(label, value, help_text=""):
    st.metric(label, value, help=help_text)

# ------------------------------
# Sidebar: Data + Options
# ------------------------------
st.sidebar.title("⚙️ Controls")

data_src = st.sidebar.radio("Data source", ["Use sample from session", "Upload CSV"])
impute_choice = st.sidebar.selectbox("Imputation for likes/comments/watch time", ["median", "mean"])
do_caps = st.sidebar.checkbox("Apply logical caps (recommended)", value=True)

if data_src == "Upload CSV":
    up = st.sidebar.file_uploader("Upload your CSV", type=["csv"])
    if up is not None:
        raw_df = pd.read_csv(up)
    else:
        st.sidebar.info("Upload a CSV to get started.")
        st.stop()
else:
    # Default path from your session
    try:
        raw_df = pd.read_csv("content_monetization/youtube_ad_revenue_dataset.csv")
    except Exception:
        st.error("Sample CSV not found. Please upload a CSV.")
        st.stop()

df = clean_dataset(raw_df, imputation=impute_choice, do_caps=do_caps)

# Sidebar filters (derived)
min_date, max_date = df["date"].min(), df["date"].max()
date_range = st.sidebar.date_input("Date range", value=(min_date.date(), max_date.date()),
                                   min_value=min_date.date(), max_value=max_date.date())

cats = sorted(df["category"].dropna().unique().tolist())
devs = sorted(df["device"].dropna().unique().tolist())
ctrs = sorted(df["country"].dropna().unique().tolist())

sel_cats = st.sidebar.multiselect("Category", cats, default=cats[: min(6,len(cats))])
sel_devs = st.sidebar.multiselect("Device", devs, default=devs)
sel_ctrs = st.sidebar.multiselect("Country", ctrs, default=ctrs[: min(10,len(ctrs))])

# Apply filters
mask = (
    (df["date"].dt.date >= date_range[0]) &
    (df["date"].dt.date <= date_range[1]) &
    (df["category"].isin(sel_cats)) &
    (df["device"].isin(sel_devs)) &
    (df["country"].isin(sel_ctrs))
)
fdf = df.loc[mask].copy()

# ------------------------------
# Header & Tabs
# ------------------------------
st.title("🎥 Content Monetization Modeler — Interactive")
st.caption("Predict YouTube ad revenue, explore trends, and build models — all in one place.")

tab_overview, tab_eda, tab_model, tab_predict, tab_export = st.tabs(
    ["Overview", "EDA", "Train Models", "Predict", "Export"]
)

# ------------------------------
# Overview
# ------------------------------
with tab_overview:
    st.subheader("Dataset Snapshot")
    c1, c2, c3, c4 = st.columns(4)
    kpi("Rows (filtered)", f"{len(fdf):,}")
    kpi("Date span", f"{date_range[0]} → {date_range[1]}")
    kpi("Avg RPM (USD/1k views)", f"{fdf['rpm_usd'].mean():.2f}")
    kpi("Total Revenue (USD)", f"{fdf[TARGET].sum():,.0f}")

    st.dataframe(fdf.head(200), use_container_width=True)

# ------------------------------
# EDA
# ------------------------------
with tab_eda:
    st.subheader("Interactive Charts")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Ad Revenue vs Views (sample)**")
        sample = fdf.sample(min(3000, len(fdf)), random_state=42)
        fig_sc = px.scatter(sample, x="views", y=TARGET, trendline="ols",
                            hover_data=["video_id","category","device","country","date"])
        st.plotly_chart(fig_sc, use_container_width=True)

    with colB:
        st.markdown("**Daily Total Revenue**")
        daily = fdf.groupby(fdf["date"].dt.date)[TARGET].sum().reset_index()
        fig_line = px.line(daily, x="date", y=TARGET)
        st.plotly_chart(fig_line, use_container_width=True)

    colC, colD = st.columns(2)
    with colC:
        st.markdown("**Average Revenue by Category**")
        cat_rev = fdf.groupby("category")[TARGET].mean().sort_values(ascending=False).reset_index()
        fig_cat = px.bar(cat_rev.head(20), x="category", y=TARGET)
        st.plotly_chart(fig_cat, use_container_width=True)

    with colD:
        st.markdown("**Total Revenue by Device**")
        dev_rev = fdf.groupby("device")[TARGET].sum().sort_values(ascending=False).reset_index()
        fig_dev = px.bar(dev_rev, x="device", y=TARGET)
        st.plotly_chart(fig_dev, use_container_width=True)

    st.markdown("**Correlation Heatmap (key metrics)**")
    corr_cols = [
        "views","likes","comments","watch_time_minutes","video_length_minutes","subscribers",
        TARGET,"engagement_rate","avg_watch_time_per_view_min","rpm_usd"
    ]
    corr = fdf[corr_cols].corr(numeric_only=True)
    fig_hm = px.imshow(corr, text_auto=False, aspect="auto", color_continuous_scale="Viridis")
    st.plotly_chart(fig_hm, use_container_width=True)

# ------------------------------
# Train Models
# ------------------------------
with tab_model:
    st.subheader("Model Training")

    model_name = st.selectbox("Choose model", ["LinearRegression","Ridge","Lasso","RandomForest","GradientBoosting"])
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    do_cv = st.checkbox("5-fold Cross-validation", value=True)
    seed = st.number_input("Random seed", value=42)

    X_cols = NUM_COLS_BASE + ["engagement_rate","avg_watch_time_per_view_min"] + CAT_COLS
    # Keep only rows without target NaN
    train_df = fdf.dropna(subset=[TARGET])
    X = train_df[X_cols]
    y = train_df[TARGET]

    pipe, num_feats, cat_feats = build_pipeline(model_name)

    if st.button("Train model"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
        if do_cv:
            cv_r2 = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2").mean()
            st.info(f"CV R² (5-fold): {cv_r2:.4f}")

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        r2 = r2_score(y_test, preds)
        rmse = mean_squared_error(y_test, preds, squared=True)
        mae = mean_absolute_error(y_test, preds)

        c1, c2, c3 = st.columns(3)
        c1.metric("R² (test)", f"{r2:.4f}")
        c2.metric("RMSE", f"{rmse:,.2f}")
        c3.metric("MAE", f"{mae:,.2f}")

        st.session_state["trained_pipe"] = pipe
        st.session_state["feature_cols"] = X_cols

        # Feature importance (if tree-based)
        model = pipe.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            # Get one-hot feature names
            ohe = pipe.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
            cat_names = ohe.get_feature_names_out(cat_feats)
            num_names = num_feats
            all_names = np.concatenate([num_names, cat_names])
            importances = pd.Series(model.feature_importances_, index=all_names).sort_values(ascending=False)
            fig_imp = px.bar(importances.head(25), title="Top Feature Importances")
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            # Permutation importance (slower but model-agnostic)
            st.caption("Model has no native importances; showing permutation importance (may take a moment).")
            result = permutation_importance(pipe, X_test, y_test, n_repeats=5, random_state=seed, n_jobs=-1)
            imp = pd.Series(result.importances_mean, index=pipe.named_steps["prep"].get_feature_names_out()).sort_values(ascending=False)
            fig_perm = px.bar(imp.head(25), title="Top Permutation Importances")
            st.plotly_chart(fig_perm, use_container_width=True)

# ------------------------------
# Predict (Single)
# ------------------------------
with tab_predict:
    st.subheader("Single Prediction")
    if "trained_pipe" not in st.session_state:
        st.info("Train a model in the **Train Models** tab first.")
    else:
        pipe = st.session_state["trained_pipe"]
        # Build small form using ranges from filtered data
        c1, c2, c3 = st.columns(3)
        with c1:
            views = st.number_input("Views", min_value=0, value=int(fdf["views"].median()))
            likes = st.number_input("Likes", min_value=0, value=int(fdf["likes"].median()))
            comments = st.number_input("Comments", min_value=0, value=int(fdf["comments"].median()))
            wt = st.number_input("Watch time (minutes)", min_value=0.0, value=float(fdf["watch_time_minutes"].median()))
        with c2:
            vlen = st.number_input("Video length (minutes)", min_value=0.0, value=float(fdf["video_length_minutes"].median()))
            subs = st.number_input("Subscribers", min_value=0, value=int(fdf["subscribers"].median()))
            category = st.selectbox("Category", sorted(fdf["category"].unique()))
        with c3:
            device = st.selectbox("Device", sorted(fdf["device"].unique()))
            country = st.selectbox("Country", sorted(fdf["country"].unique()))

        # Compute engineered features like the training step
        engagement_rate = (likes + comments) / (views if views > 0 else 1)
        engagement_rate = min(max(engagement_rate, 0), 1)
        avg_wpv = (wt / views) if views > 0 else 0.0

        if st.button("Predict revenue"):
            row = pd.DataFrame([{
                "views": views, "likes": likes, "comments": comments, "watch_time_minutes": wt,
                "video_length_minutes": vlen, "subscribers": subs,
                "category": category, "device": device, "country": country,
                "engagement_rate": engagement_rate, "avg_watch_time_per_view_min": avg_wpv
            }])
            pred = pipe.predict(row)[0]
            st.success(f"💰 Estimated Ad Revenue: **${pred:,.2f}**")

# ------------------------------
# Export
# ------------------------------
with tab_export:
    st.subheader("Export & Downloads")

    # Cleaned CSV
    cleaned = fdf.copy()
    buf = io.StringIO()
    cleaned.to_csv(buf, index=False)
    st.download_button("Download cleaned CSV", data=buf.getvalue(), file_name="youtube_ad_revenue_cleaned.csv", mime="text/csv")

    # Model
    if "trained_pipe" in st.session_state:
        model_bytes = io.BytesIO()
        joblib.dump(st.session_state["trained_pipe"], model_bytes)
        st.download_button("Download trained model (.pkl)", data=model_bytes.getvalue(),
                           file_name="youtube_revenue_model.pkl", mime="application/octet-stream")
    else:
        st.caption("Train a model to enable model download.")