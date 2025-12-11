#  Product Category Prediction

This is a simple machine learning project where we try to predict the category of a product based on its title and a few numeric features (views, merchant rating, listing date).  
The project is designed as a learning exercise, step by step.


##  Project Structure
products-category-classifier/
├── data/
│   ├── products.csv
|   ├── products_modified.csv
├── notebooks/
│   ├── explorare_and_cleaning.ipynb
│   ├── feature_engineering_and_modeling.ipynb
├── src/
│   ├── train_model.py
│   ├── predict_category.py
├── models/
│   └── category_model.pkl
├── README.md


##  Main Steps

### 1. Load and inspect the dataset
- Read the CSV file with **Pandas**
- Check dataset size (`df.shape`) and column names (`df.columns`)
- Preview the first rows (`df.head()`) to understand the structure

### 2. Clean the data
- Standardize column names (lowercase, no spaces)
- Clean product titles (lowercase, remove extra spaces)
- Drop rows with missing values (`dropna`)

### 3. Feature engineering
- Create new columns:
  - `views_log` → log(1 + number_of_views) to reduce skew
  - `year` and `month` → extracted from `listing_date`
- These features help the model capture more useful patterns

### 4. Preprocessing
- Use **ColumnTransformer**:
  - `TfidfVectorizer` for text (`product_title`)
  - `StandardScaler` for numeric features (`views_log`, `merchant_rating`, `year`, `month`)

### 5. Build the pipeline
- Combine preprocessing with a classifier in a **Pipeline**
- Models tested:
  - Logistic Regression
  - Naive Bayes
  - Decision Tree
  - Random Forest
  - Linear SVM (`LinearSVC`)

### 6. Train and evaluate
- Split data into train/validation (80/20)
- Train each model and calculate:
  - **Accuracy**
  - **Precision, Recall, F1** (classification report)
- Compare results to choose the best model

### 7. Save the model
- Use `joblib.dump` to save the trained pipeline in `models/category_model.pkl`
- This allows reusing the model without retraining

### 8. Make predictions
- `predict_category.py` loads the saved model
- You can enter a product title (and numeric features if needed)
- The model outputs the predicted category