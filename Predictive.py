# Predicting Type (Movie vs TV Show)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


df = pd.read_csv(r"D:\Project 234\netflix_titles.csv")


df['country'] = df['country'].fillna("Unknown")
df['rating'] = df['rating'].fillna("Unknown")
df['duration'] = df['duration'].fillna("0 min")

def convert_duration(value):
    if "min" in value:
        return int(value.split()[0])
    return 0

df['duration_minutes'] = df['duration'].apply(convert_duration)
df['genre_count'] = df['listed_in'].apply(lambda x: len(x.split(",")))
df['cast_size'] = df['cast'].fillna("").apply(lambda x: len(x.split(",")))


X = df[['country', 'rating', 'release_year', 'duration_minutes', 'genre_count', 'cast_size']]
y = df['type']

label = LabelEncoder()
y = label.fit_transform(y)

categorical_features = ['country', 'rating']
numerical_features = ['release_year', 'duration_minutes', 'genre_count', 'cast_size']

preprocess = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('num', 'passthrough', numerical_features)
    ]
)

# ==========================
# TRAIN-TEST SPLIT
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================
# STORAGE FOR GRAPH
# ==========================
model_names = []
model_accuracies = []

# ==========================
# MODEL TRAINING FUNCTION
# ==========================
def evaluate_model(model, name):
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocess),
        ('model', model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    # store accuracy for comparison graph
    model_names.append(name)
    model_accuracies.append(acc)

    print(f"\n===== {name} =====")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", acc)

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# ==========================
# RUN MODELS
# ==========================
evaluate_model(LogisticRegression(max_iter=2000), "Logistic Regression")
evaluate_model(DecisionTreeClassifier(), "Decision Tree")
evaluate_model(KNeighborsClassifier(n_neighbors=5), "KNN Classifier")
evaluate_model(GaussianNB(), "Naive Bayes")

# ==========================
# ACCURACY COMPARISON GRAPH
# ==========================
plt.figure(figsize=(8, 5))
sns.barplot(x=model_names, y=model_accuracies)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xticks(rotation=25)
plt.ylim(0, 1)
plt.show()
