import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import joblib
from pathlib import Path

url = "https://raw.githubusercontent.com/Uvirgil/products-category-prediction/main/data/products_modified.csv"
df = pd.read_csv(url)
print("Dataset loaded successfully: ",df.shape)


# Column cleaning
df["product_title"] = df["product_title"].fillna("").astype(str).str.strip()
df["category_label"] = df["category_label"].astype(str).str.strip()

# Feature engineering
df["title_length"] = df["product_title"].apply(len)
df["word_count"] = df["product_title"].apply(lambda x: len(x.split()))
df["has_number"] = df["product_title"].apply(lambda x: int(any(ch.isdigit() for ch in x)))

# Features and labels
X = df[["product_title", "title_length", "word_count", "has_number", "number_of_views", "merchant_rating"]]
y = df["category_label"]

# Preprocessing: TF-IDF per title and numerical scaling
preprocessor = ColumnTransformer(
    transformers=[
        ("title", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=100000), "product_title"),
        ("num", StandardScaler(), ["title_length", "word_count", "has_number", "number_of_views", "merchant_rating"])
    ]
)

# Pipeline with Logistic Regression (most stable for multi-class)
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LinearSVC())
])

# Training on the entire dataset
pipeline.fit(X, y)

# Save model
joblib.dump(pipeline, "products-category-prediction/models/category_model.pkl")
print(" Model trained and saved as 'models/category_model.pkl'")
