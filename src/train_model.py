import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import os

def train_mushroom_model():
    # 1. Load the dataset
    data_path = os.path.join('data', 'mushroom_data.csv')
    df = pd.read_csv(data_path)
    print("--- 1. Data Loaded Successfully ---")
    print(df.head(3))
    
    # 2. Split into features (Inputs) and target (Output)
    X = df[['Temperature', 'Humidity', 'CO2']] # What the model looks at
    y = df['Yield']                            # What the model tries to predict
    
    # 3. Split into training set (80%) and testing set (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Initialize and Train the Machine Learning Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("\n--- 2. Model Training Complete ---")
    
    # 5. Evaluate the model
    predictions = model.predict(X_test)
    error = mean_squared_error(y_test, predictions)
    print(f"Model Error (Mean Squared Error): {error:.4f}")
    
    # 6. Save the trained brain into the 'models' folder
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', 'mushroom_model.pkl')
    joblib.dump(model, model_path)
    print(f"\n--- 3. Saved Trained Model to: {model_path} ---")

if __name__ == '__main__':
    train_mushroom_model()