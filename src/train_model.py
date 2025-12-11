import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import joblib
from pathlib import Path

url = "https://raw.githubusercontent.com/Uvirgil/products-category-prediction/main/data/products_modified.csv"
df = pd.read_csv(url)
print("Dataset loaded successfully:", df.shape)

# Column cleaning
df["product_title"] = df["product_title"].fillna("").astype(str).str.strip()
df["category_label"] = df["category_label"].astype(str).str.strip()

# Feature engineering
df["views_log"] = np.log1p(df["number_of_views"])
df["year"] = pd.to_datetime(df["listing_date"]).dt.year
df["month"] = pd.to_datetime(df["listing_date"]).dt.month

# Features and label
X = df[["product_title", "views_log", "merchant_rating", "year", "month"]]
y = df["category_label"]

# Preprocessing: TF-IDF per title and numerical scaling
preprocessor = ColumnTransformer(
    transformers=[
        ("title", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=100000), "product_title"),
        ("num", StandardScaler(), ["views_log", "merchant_rating", "year", "month"])
    ]
)

# Pipeline with LinearSVC
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LinearSVC(max_iter=5000))
])

# Train the model
pipeline.fit(X, y)

# Save model
joblib.dump(pipeline, "products-category-prediction/models/category_model.pkl")
print(" Model trained and saved as 'models/category_model.pkl'")