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








# *************************************
# Objective 2: Regression
# Predicting Duration (Minutes) of Netflix Titles
# *************************************

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ==========================
# LOAD DATA
# ==========================
df = pd.read_csv(r"D:\Project 234\netflix_titles.csv")

# ==========================
# DATA PREPROCESSING
# ==========================
df['duration'] = df['duration'].fillna("0 min")

def convert_duration(value):
    if "min" in value:
        return int(value.split()[0])
    return 0   # TV shows have seasons; ignored here

df['duration_minutes'] = df['duration'].apply(convert_duration)

df['genre_count'] = df['listed_in'].apply(lambda x: len(x.split(",")))
df['cast_size'] = df['cast'].fillna("").apply(lambda x: len(x.split(",")))

# ========= Select Features =========
X = df[['release_year', 'genre_count', 'cast_size']]
y = df['duration_minutes']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Evaluation Function
def evaluate_regression(model, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n===== {name} =====")
    print("MAE :", mean_absolute_error(y_test, y_pred))
    print("MSE :", mean_squared_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R² Score:", r2_score(y_test, y_pred))

# ==========================
# 1. Simple Linear Regression
# Using only release_year as feature
# ==========================
print("\n\n### SIMPLE LINEAR REGRESSION ###")
simple_model = LinearRegression()
simple_model.fit(X_train[['release_year']], y_train)
y_pred = simple_model.predict(X_test[['release_year']])

print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))

# ==========================
# 2. Multiple Linear Regression
# ==========================
print("\n\n### MULTIPLE LINEAR REGRESSION ###")
multi_model = LinearRegression()
evaluate_regression(multi_model, "Multiple Linear Regression")

# ==========================
# 3. Polynomial Regression (Degree = 2)
# ==========================
print("\n\n### POLYNOMIAL REGRESSION (Degree 2) ###")

poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_poly_train, y_train)
y_pred_poly = poly_model.predict(X_poly_test)

print("MAE :", mean_absolute_error(y_test, y_pred_poly))
print("MSE :", mean_squared_error(y_test, y_pred_poly))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_poly)))
print("R² Score:", r2_score(y_test, y_pred_poly))

# ==========================
# 4. Decision Tree Regression
# ==========================
print("\n\n### DECISION TREE REGRESSION ###")
tree_model = DecisionTreeRegressor(random_state=42)
evaluate_regression(tree_model, "Decision Tree Regression")

# ==========================
# OPTIONAL: Scatter Plot
# ==========================
plt.scatter(df['release_year'], df['duration_minutes'], alpha=0.4)
plt.title("Release Year vs Duration (Minutes)")
plt.xlabel("Release Year")
plt.ylabel("Duration")
plt.show()








# ---------------------------------------------
# OBJECTIVE 3: CONTENT TREND ANALYSIS
# ---------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(r"D:\Project 234\netflix_titles.csv")

# ---------------------------------------------
# DATA PREPROCESSING
# ---------------------------------------------
# Remove rows where date_added is missing
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
df = df.dropna(subset=['date_added'])

# Extract year and month
df['year_added'] = df['date_added'].dt.year
df['month_added'] = df['date_added'].dt.month_name()

# ---------------------------------------------
# 1. Trend of total content added per year
# ---------------------------------------------
plt.figure(figsize=(12,5))
sns.countplot(x='year_added', data=df)
plt.title("Total Content Added on Netflix Over the Years", fontsize=14)
plt.xlabel("Year")
plt.ylabel("Number of Titles")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ---------------------------------------------
# 2. Movies vs TV Shows added each year
# ---------------------------------------------
plt.figure(figsize=(12,5))
sns.countplot(x='year_added', data=df, hue='type')
plt.title("Movies vs TV Shows Added per Year", fontsize=14)
plt.xlabel("Year")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.legend(title="Type")
plt.tight_layout()
plt.show()

# ---------------------------------------------
# 3. Heatmap of content release by Month & Year
# ---------------------------------------------
pivot = df.pivot_table(index='month_added', columns='year_added', values='show_id', aggfunc='count')

# Order months correctly
months = ["January","February","March","April","May","June","July","August","September","October","November","December"]
pivot = pivot.reindex(months)

plt.figure(figsize=(15,8))
sns.heatmap(pivot, cmap="YlOrRd", linewidths=.5)
plt.title("Heatmap of Content Added per Month & Year", fontsize=14)
plt.xlabel("Year")
plt.ylabel("Month")
plt.show()






# *************************************
# Objective 6: MODEL PERFORMANCE
# Cross-validation, Bias-Variance, Bagging, Boosting, Random Forest
# Using Netflix Dataset
# *************************************

import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier

# ======================================
# LOAD DATA
# ======================================
df = pd.read_csv(r"D:\Project 234\netflix_titles.csv")

# Fill missing values
df['country'] = df['country'].fillna("Unknown")
df['rating'] = df['rating'].fillna("Unknown")
df['duration'] = df['duration'].fillna("0 min")

# Convert duration into minutes
def convert_duration(value):
    if "min" in value:
        return int(value.split()[0])
    return 0

df['duration_minutes'] = df['duration'].apply(convert_duration)
df['genre_count'] = df['listed_in'].apply(lambda x: len(x.split(",")))
df['cast_size'] = df['cast'].fillna("").apply(lambda x: len(x.split(",")))

# ======================================
# SELECT FEATURES & TARGET
# ======================================
X = df[['country', 'rating', 'release_year', 'duration_minutes', 'genre_count', 'cast_size']]
y = df['type']

# Encode target
label = LabelEncoder()
y = label.fit_transform(y)

# Preprocessing
categorical = ['country', 'rating']
numerical = ['release_year', 'duration_minutes', 'genre_count', 'cast_size']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical),
    ('num', 'passthrough', numerical)
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)


# ======================================
# FUNCTION TO EVALUATE MODELS
# ======================================
def evaluate_model(model, model_name):
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Fit model
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    print(f"\n=========== {model_name} ===========")
    print("Accuracy on Test Data:", accuracy_score(y_test, y_pred))

    # K-Fold Cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=kfold, scoring='accuracy')

    print("Cross-Validation Accuracy:", cv_scores.mean())
    print("Bias-Variance Check →")
    print("   Variance:", cv_scores.var())
    print("   Std Dev:", cv_scores.std())


# ======================================
# MODELS FOR OBJECTIVE 6
# ======================================

# 1) Decision Tree (Base Model)
evaluate_model(DecisionTreeClassifier(), "Decision Tree")

# 2) Bagging (Reduces Variance)
evaluate_model(BaggingClassifier(
    estimator=DecisionTreeClassifier(), 
    n_estimators=50, random_state=42),
    "Bagging Classifier"
)

# 3) Boosting (Reduces Bias)
evaluate_model(AdaBoostClassifier(
    estimator=DecisionTreeClassifier(), 
    n_estimators=50, random_state=42),
    "AdaBoost Classifier"
)

# 4) Random Forest (Bagging + Feature Randomness)
evaluate_model(RandomForestClassifier(
    n_estimators=100, random_state=42),
    "Random Forest"
)





# 7th Objective: Hierarchical Clustering with Dendrogram

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("D:/Project 234/netflix_titles.csv")

# Clean genre column
df['listed_in'] = df['listed_in'].fillna("Unknown")
df['genre_list'] = df['listed_in'].apply(lambda x: x.split(", "))

# Convert genres into numerical vectors using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(df['genre_list'])

# Perform hierarchical clustering using Ward linkage
Z = linkage(genre_encoded[:50], method='ward')  
# (Using first 50 items to avoid overcrowded dendrogram)

# Plot Dendrogram
plt.figure(figsize=(14, 7))
dendrogram(Z, labels=df['title'].values[:50], leaf_rotation=90)
plt.title("Hierarchical Clustering Dendrogram of Netflix Titles")
plt.xlabel("Movie/TV Show Title")
plt.ylabel("Euclidean Distance")
plt.tight_layout()
plt.show()

