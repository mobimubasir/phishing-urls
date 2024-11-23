import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Step 1: Load the dataset
# Assuming the CSV contains two columns: 'url' (the URL) and 'label' (0 for safe, 1 for phishing)
data = pd.read_csv('phishing_dataset.csv')

# Step 2: Preprocess the data
# Ensure no missing values
data.dropna(inplace=True)

# Extract features (URLs) and labels
urls = data['url']
labels = data['label']

# Step 3: Convert URLs into numerical features using TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    stop_words='english',  # Remove common stop words
    token_pattern=r'(?u)\b[A-Za-z0-9\-\.]+\b',  # Tokenize URL parts
    max_features=5000  # Use top 5000 features to avoid overfitting
)
X = vectorizer.fit_transform(urls)

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Step 5: Train a machine learning model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 7: Save the trained model and vectorizer
with open('phishing_url_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model and vectorizer saved successfully!")
