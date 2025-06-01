import pandas as pd
import joblib
from ydata_profiling import ProfileReport
from sklearn.preprocessing import LabelEncoder
# Load dataset (adjust path accordingly)
df = pd.read_csv("Expresso_churn_dataset.csv")
# Display first 5 rows
print(df.head())
# Basic info
print(df.info())
# Statistics
print(df.describe())
# Check nulls
print(df.shape)
# Create a pandas profiling reports to gain insights into the dataset
#from ydata_profiling import ProfileReport
# Generate the profile report :
# "explorative=True" :Activate advanced or in-depth analysis (explorative data analysis).
#profile = ProfileReport(df, title="Expresso Churn Report", explorative=True)
# Display the report
#profile.to_notebook_iframe()
# Save the report to an HTML file
#profile.to_file("expresso_churn_report.html")
#print("✔ Rapport generated succefully ! Open file expresso_churn_report.html")
df_cleaned = df.copy() # Make a copy of data before  preprocessing
# Handle Missing and corrupted values
print(df_cleaned.isnull().sum())
print("\nMissing values of the data = ",df_cleaned.isnull().sum().sum())

# 'TENURE' is a categorical column (e.g., 'K > 24 month'), not numeric.
# We'll fill missing values with the most frequent category (mode).
most_frequent_tenure = df_cleaned['TENURE'].mode()[0]
df_cleaned['TENURE'] = df_cleaned['TENURE'].fillna(most_frequent_tenure)
# 'REGION' is a categorical variable representing customer location.
# We'll fill missing values using the mode as well.
most_frequent_region = df_cleaned['REGION'].mode()[0]
df_cleaned['REGION'] = df_cleaned['REGION'].fillna(most_frequent_region)
# These columns are numerical and may contain outliers.
# We'll use the median to fill missing values because it is robust to outliers.
numeric_cols = [
    'MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT',
    'FREQUENCE', 'DATA_VOLUME', 'ON_NET', 'ORANGE'
]

for col in numeric_cols:
    median_value = df_cleaned[col].median()
    df_cleaned[col] = df_cleaned[col].fillna(median_value)
# 'MRG' is likely a binary or categorical indicator (e.g., Yes/No).
# We'll use the mode to fill the single missing value.
df_cleaned['MRG'] = df_cleaned['MRG'].fillna(df_cleaned['MRG'].mode()[0])
# 'REGULARITY' is a numeric variable (activity frequency in 90 days).
# We'll use the median as it's numeric and could contain outliers.
df_cleaned['REGULARITY'] = df_cleaned['REGULARITY'].fillna(df_cleaned['REGULARITY'].median())
# 'TOP_PACK' is categorical; we use mode
df_cleaned['TOP_PACK'] = df_cleaned['TOP_PACK'].fillna(df_cleaned['TOP_PACK'].mode()[0])

# 'FREQ_TOP_PACK' is numeric; we use median
df_cleaned['FREQ_TOP_PACK'] = df_cleaned['FREQ_TOP_PACK'].fillna(df_cleaned['FREQ_TOP_PACK'].median())
# ZONE2 showed strong correlation with CHURN, so we keep it.
# We'll fill missing values with 0, assuming it means no calls to that zone.
# TIGO and ZONE1 likely represent call durations or counts.
# Missing values can reasonably be interpreted as 0 (i.e., no usage).

df_cleaned['TIGO'] = df_cleaned['TIGO'].fillna(0)
df_cleaned['ZONE1'] = df_cleaned['ZONE1'].fillna(0)

df_cleaned['ZONE2']= df_cleaned['ZONE2'].fillna(0)

# 'CHURN' is the target variable (usually 0 or 1).
# We drop the row with missing target value to avoid issues in model training.
df_cleaned = df_cleaned[df_cleaned['CHURN'].notna()]
print(df_cleaned.isnull().sum())
print(df_cleaned.duplicated().sum())

# Drop rows with missing CHURN (target)
df_cleaned = df_cleaned[df_cleaned['CHURN'].notna()]

# Remove duplicates
df_cleaned = df_cleaned.drop_duplicates()

# Function to detect outliers in all numerical columns using the IQR method
def detect_outliers_iqr(data):
    outlier_info = {}  # Dictionary to store outliers per column

    # Loop through each numerical column
    for column in data.select_dtypes(include=['number']).columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Boolean mask for outliers
        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]

        # Store results
        outlier_info[column] = {
            'count': len(outliers),
            'indices': outliers.index.tolist()
        }

        print(f"🔍 Column: {column} | Outliers: {len(outliers)}")

    return outlier_info

# Run the function on your cleaned DataFrame
outliers_dict = detect_outliers_iqr(df_cleaned)



# OUTLIER REMOVAL using IQR method (no log)
def remove_outliers_iqr(data, columns):
    for column in columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return data

columns_with_outliers = [
    'REVENUE', 'MONTANT', 'ARPU_SEGMENT', 'DATA_VOLUME',
    'ON_NET', 'ORANGE', 'TIGO', 'FREQUENCE_RECH', 'FREQUENCE',
    'ZONE1', 'ZONE2', 'FREQ_TOP_PACK'
]

df_cleaned = remove_outliers_iqr(df_cleaned, columns_with_outliers)

# Encode categorical variables
# Label encode binary/ordinal columns
label_enc_cols = ['TENURE', 'MRG']
le = LabelEncoder()
for col in label_enc_cols:
    df_cleaned[col] = le.fit_transform(df_cleaned[col].astype(str))

# One-Hot encode nominal columns
df_cleaned = pd.get_dummies(df_cleaned, columns=['TOP_PACK', 'REGION'], drop_first=True)

# Final check
print("✅ Final dataset shape:", df_cleaned.shape)
print("✅ Columns:", df_cleaned.columns)

# Train/test split and model training using RandomForestClassifier

from sklearn.model_selection import train_test_split
# Features and target
X = df_cleaned.drop(['user_id', 'CHURN'], axis=1)
y = df_cleaned['CHURN']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# GridSearch for best parameters
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
# Define model and param grid
param_grid = {
    'n_estimators': [100, 150],           # number of trees
    'max_depth': [None, 10, 20],          # depth of each tree
    'min_samples_split': [2, 5],          # min samples to split a node
    'min_samples_leaf': [1, 2],           # min samples at a leaf
    'max_features': ['sqrt', 'log2']      # number of features to consider when looking for the best split
}
model = RandomForestClassifier(random_state=42)

# GridSearchCV will test all parameter combinations using 3-fold cross-validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                           cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
grid_search.fit(X_train, y_train)
# Best parameters and model evaluation
print("✅ Best Parameters:", grid_search.best_params_)

best_model = grid_search.best_estimator_

# Predict on test set
y_pred = best_model.predict(X_test)

# Evaluate
from sklearn.metrics import accuracy_score, classification_report
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("✅ Classification Report:\n", classification_report(y_test, y_pred))

# Final Model Evaluation & Visualizations
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
# Predict again using the best model
y_pred = best_model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Streamlit App: Step-by-Step Structure
# Save only the best estimator
print("Saving model...")

joblib.dump(best_model, 'churn_model.pkl', compress=3)
# Save column names for future inference (Streamlit)
joblib.dump(X.columns.tolist(), 'model_features.pkl', compress=3)
print("✅ Model saved as churn_model.pkl")


