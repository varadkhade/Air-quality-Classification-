# air_quality_simple.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
def load_data():
    df = pd.read_csv('air_quality_dataset.csv')
    print("Data loaded. Shape:", df.shape)
    return df

# Show visualizations
def show_visualizations(df):
    # Create a figure with subplots
    plt.figure(figsize=(15, 5))
    
    # 1. Bar Chart
    plt.subplot(1, 3, 1)
    quality_counts = df['Quality'].value_counts().sort_index()
    quality_labels = ['Hazardous', 'Poor', 'Moderate', 'Good', 'Excellent']
    colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
    
    plt.bar(quality_labels, quality_counts.values, color=colors, alpha=0.7)
    plt.title('Air Quality Distribution - Bar Chart')
    plt.xlabel('Quality Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # 2. Pie Chart
    plt.subplot(1, 3, 2)
    plt.pie(quality_counts.values, labels=quality_labels, colors=colors, autopct='%1.1f%%')
    plt.title('Air Quality Distribution - Pie Chart')
    
    # 3. Heat Map
    plt.subplot(1, 3, 3)
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heat Map')
    
    plt.tight_layout()
    plt.show()

# Train model
def train_air_quality_model(df):
    X = df.drop('Quality', axis=1)
    y = df['Quality']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

# Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Predict function
def predict_air_quality(model):
    print("\nEnter air quality parameters:")
    pm25 = float(input("PM2.5: "))
    pm10 = float(input("PM10: "))
    no2 = float(input("NO2: "))
    so2 = float(input("SO2: "))
    co = float(input("CO: "))
    o3 = float(input("O3: "))
    
    input_data = np.array([[pm25, pm10, no2, so2, co, o3]])
    prediction = model.predict(input_data)[0]
    
    quality_map = {1: 'Hazardous', 2: 'Poor', 3: 'Moderate', 4: 'Good', 5: 'Excellent'}
    print(f"\nPredicted Air Quality: {prediction} ({quality_map[prediction]})")

# Main execution
def main():
    # Load data
    df = load_data()
    
    # Show visualizations
    show_visualizations(df)
    
    # Train model
    model, X_test, y_test = train_air_quality_model(df)
    
    # Evaluate
    evaluate_model(model, X_test, y_test)
    
    # Interactive prediction
    while True:
        predict_air_quality(model)
        cont = input("\nPredict again? (y/n): ").lower()
        if cont != 'y':
            break

if __name__ == "__main__":
    main()
