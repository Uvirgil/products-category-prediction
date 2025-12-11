import joblib
import pandas as pd

# Load the saved model
model = joblib.load("products-category-prediction/models/category_model.pkl")

print("Model loaded successfully!")
print("Type 'exit' at any point to stop.\n")

while True:
    # Read the product title
    title = input("Enter product title: ")
    if title.lower() == "exit":
        print("Exiting...")
        break

    # Create a DataFrame with user input
    # If your model only uses the title, the rest of the columns can be filled with default values
    user_input = pd.DataFrame([{
        "product_title": title,
        "views_log": 0,
        "merchant_rating": 0,
        "year": 2024,
        "month": 1
    }])

    # Predict the category
    prediction = model.predict(user_input)[0]
    print(f"Predicted category: {prediction}\n" + "-" * 60)
