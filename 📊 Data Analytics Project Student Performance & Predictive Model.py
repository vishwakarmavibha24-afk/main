import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set a random seed for reproducibility
np.random.seed(42)

print("--- 1. Data Sourcing (Simulation) ---")

# Simulate a dataset of 1000 students
n_samples = 1000

# Feature: previous_scores (0-100)
previous_scores = np.random.normal(loc=65, scale=15, size=n_samples)
previous_scores = np.clip(previous_scores, 0, 100)

# Feature: study_hours (1-30 hours/week)
study_hours = np.random.normal(loc=12, scale=5, size=n_samples)
study_hours = np.clip(study_hours, 1, 30)

# Feature: parental_education (Categorical)
parental_education = np.random.choice(
    ['High School', 'Bachelors', 'Masters', 'PhD'],
    n_samples,
    p=[0.3, 0.4, 0.2, 0.1]
)

# Feature: socio_economic_status (Categorical, correlated with education)
ses_choices = ['Low', 'Medium', 'High']
ses_probs = [0.5, 0.3, 0.2]  # Base probabilities
ses = [np.random.choice(ses_choices, p=ses_probs) for _ in range(n_samples)]

# Create the target variable (Pass/Fail)
# We'll create a 'score' based on features and then apply a sigmoid to get probability
# This makes the data realistically predictable
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define weights for each feature
score = -4.0  # Base intercept (making 'Pass' harder)
score += (previous_scores / 20)  # Weight for scores
score += (study_hours / 5)       # Weight for study hours

# Add weights for categorical features
edu_map = {'High School': -0.5, 'Bachelors': 0.2, 'Masters': 0.8, 'PhD': 1.2}
ses_map = {'Low': -0.8, 'Medium': 0.1, 'High': 0.7}

score += np.array([edu_map[edu] for edu in parental_education])
score += np.array([ses_map[s] for s in ses])

# Add some random noise
score += np.random.normal(0, 1.0, n_samples)

# Convert score to probability and then to binary outcome (0 or 1)
probabilities = sigmoid(score)
pass_fail = (probabilities > 0.55).astype(int)  # 1 = Pass, 0 = Fail

# Create DataFrame
df = pd.DataFrame({
    'previous_scores': previous_scores,
    'study_hours': study_hours,
    'parental_education': parental_education,
    'socio_economic_status': ses,
    'pass_fail': pass_fail
})

print(f"Generated dataset with {len(df)} samples.")
print(df.head())

print("\n--- 2. Data Cleaning ---")
# Introduce some missing values to clean (as per project description)
for col, prop in [('study_hours', 0.05), ('socio_economic_status', 0.03)]:
    idx = np.random.choice(df.index, size=int(df.shape[0] * prop), replace=False)
    df.loc[idx, col] = np.nan

print(f"Introduced missing values. Null counts:\n{df.isnull().sum()}")

# NOTE: The actual cleaning (imputation) will be handled inside the
# scikit-learn pipeline (Step 5) to prevent data leakage.
# This is the modern, best-practice approach.

print("\n--- 3. Exploratory Data Analysis (EDA) ---")

# Set up the plotting style
sns.set_style("whitegrid")
plt.figure(figsize=(18, 12))

# Plot 1: Target Variable Distribution
plt.subplot(2, 3, 1)
sns.countplot(x='pass_fail', data=df, palette='pastel')
plt.title('Distribution of Pass/Fail Outcomes')
plt.xticks([0, 1], ['Fail (0)', 'Pass (1)'])

# Plot 2: Study Hours vs. Outcome
plt.subplot(2, 3, 2)
sns.histplot(data=df, x='study_hours', hue='pass_fail', kde=True, bins=20, palette='viridis')
plt.title('Study Hours vs. Pass/Fail')


# Plot 3: Previous Scores vs. Outcome
plt.subplot(2, 3, 3)
sns.histplot(data=df, x='previous_scores', hue='pass_fail', kde=True, bins=20, palette='plasma')
plt.title('Previous Scores vs. Pass/Fail')


# Plot 4: Parental Education vs. Outcome
plt.subplot(2, 3, 4)
sns.countplot(data=df, x='parental_education', hue='pass_fail', order=['High School', 'Bachelors', 'Masters', 'PhD'], palette='coolwarm')
plt.title('Parental Education vs. Pass/Fail')

# Plot 5: Socio-Economic Status vs. Outcome
plt.subplot(2, 3, 5)
sns.countplot(data=df, x='socio_economic_status', hue='pass_fail', order=['Low', 'Medium', 'High'], palette='ocean_r')
plt.title('Socio-Economic Status vs. Pass/Fail')

# Plot 6: Correlation Matrix (for numeric features)
plt.subplot(2, 3, 6)
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Numeric Feature Correlation Matrix')

plt.tight_layout()
plt.show()

print("EDA plots generated. Check the output window.")

print("\n--- 4. Feature Engineering & Preprocessing ---")

# Define features (X) and target (y)
X = df.drop('pass_fail', axis=1)
y = df['pass_fail']

# Identify numeric and categorical features
numeric_features = ['previous_scores', 'study_hours']
categorical_features = ['parental_education', 'socio_economic_status']

# Create the preprocessing pipeline for numeric data:
# 1. Impute missing values with the median
# 2. Standardize data (as per project description)
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Create the preprocessing pipeline for categorical data:
# 1. Impute missing values with the most frequent value (mode)
# 2. Apply One-Hot Encoding
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

print("Preprocessing pipeline created successfully.")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Data split: {len(X_train)} training samples, {len(X_test)} testing samples.")


print("\n--- 5. Model Implementation (Logistic Regression) ---")

# Create the full pipeline: Preprocessing + Model
log_reg_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, solver='liblinear'))
])

# Train the model
log_reg_pipeline.fit(X_train, y_train)

# Make predictions
y_pred_lr = log_reg_pipeline.predict(X_test)

# Evaluate the model
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Model Accuracy: {accuracy_lr * 100:.2f}%")
print("Classification Report (Logistic Regression):")
print(classification_report(y_test, y_pred_lr, target_names=['Fail (0)', 'Pass (1)']))


print("\n--- 6. Model Implementation (Random Forest) ---")

# Create the full pipeline: Preprocessing + Model
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
rf_pipeline.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_pipeline.predict(X_test)

# Evaluate the model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Model Accuracy: {accuracy_rf * 100:.2f}%")
print("Classification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf, target_names=['Fail (0)', 'Pass (1)']))


print("\n--- 7. Impact/Skills (Feature Importance Analysis) ---")

# Extract feature importances from the Random Forest model
# This shows which features were the "strongest predictors"

# Get the feature names from the OneHotEncoder
ohe_feature_names = rf_pipeline.named_steps['preprocessor'] \
                               .named_transformers_['cat'] \
                               .named_steps['onehot'] \
                               .get_feature_names_out(categorical_features)

# Combine with numeric feature names
all_feature_names = numeric_features + list(ohe_feature_names)

# Get the importances
importances = rf_pipeline.named_steps['classifier'].feature_importances_

# Create a DataFrame for plotting
feature_importance_df = pd.DataFrame({
    'feature': all_feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

print("Top 5 Strongest Predictors:")
print(feature_importance_df.head())

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='rocket')
plt.title('Feature Importance (from Random Forest)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

print("\n--- Project Complete ---")
print(f"Achieved accuracy (LR/RF): {accuracy_lr*100:.2f}% / {accuracy_rf*100:.2f}%.")
print("This demonstrates proficiency in data cleaning (pipelined), EDA, and statistical modeling.")