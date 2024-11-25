# Import modules

# data processing
import pandas as pd
import numpy as np

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# encoding data and scaling
from sklearn.preprocessing import LabelEncoder # encoding data
from sklearn.preprocessing import MinMaxScaler # scaling

# model training
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, make_scorer, accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# color dictionary
color_dict = {
    'Yes': '#CC8963',
    'No': '#5975A4',
    'Sometimes': '#5F9E6E',
    'Never': '#5975A4',
    'Rarely': '#CC8963',
    'Often': '#B55D60',
    "Don't know": '#5F9E6E',
    'Unknown': '#E3EDCD'
}

# Loading the dataset
def get_original_data():
    file = './input/survey.csv'
    df = pd.read_csv(file)
    return df

df = get_original_data()

# The num of rows and columns of dataset
print("Data Shape:", df.shape)

# The type of columns
print(df.info())

# Remove unused columns
df = df.drop(['Timestamp','Country','state','comments'], axis = 1)
df.head(5)

# Check for null values in each column
columnStr = 'column'
mcStr = 'missing count'
print(f"{columnStr:<25} {mcStr}")

for col in df:
    missing_count = df[col].isnull().sum()
    print(f"{col:<30} {missing_count}")

# Assign all empty values to 'NaN'
df['self_employed'] = df['self_employed'].fillna('NaN')
df['work_interfere'] = df['work_interfere'].fillna('NaN')
df.head(5)

# Replace 'NaN' in 'self_employed' column
print(df['self_employed'].value_counts(), "\n")
df['self_employed'] = df['self_employed'].replace(['NaN'], 'No')

# Replace 'NaN' in 'work_interfere' column
print(df['work_interfere'].value_counts())
df['work_interfere'] = df['work_interfere'].replace('NaN', 'Unknown')

# Clean gender
def clean_gender(gender):
    gender = str(gender).strip().lower()
    if gender in ['male', 'm', 'man', 'cis male', 'male (cis)', 'cis man', 'Guy']:
        return 'Male'
    elif gender in ['female', 'f', 'woman', 'cis female', 'female (cis)', 'cis woman']:
        return 'Female'
    elif 'trans' in gender and 'male' in gender:
        return 'transMale'
    elif 'trans' in gender and 'female' in gender:
        return 'transFemale'
    elif gender in ['non-binary', 'nonbinary', 'nb', 'genderqueer', 'gender fluid']:
        return 'Other'
    else:
        return 'Other'  # 将 unknown 和 Non-binary 归为 Other

df['Gender'] = df['Gender'].apply(clean_gender)
print(df['Gender'].value_counts())

# Clean age
print("min age", df['Age'].min())
print("max age", df['Age'].max())

age_series = pd.Series(df['Age'])
age_median = age_series.median()
age_series[age_series <= 18] = age_median
age_series[age_series >= 100] = age_median
df['Age'] = age_series

print("min age", df['Age'].min())
print("max age", df['Age'].max())

# Create a copy of cleaned dataset
train_df = df.copy()
# Store the mapping between column value and encoded data
encoded_value_mapping = {}

for col in train_df:
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df[col])
    # Assgin encoded data to df
    train_df[col] = label_encoder.transform(train_df[col])
    encoded_value_mapping[col] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

train_df.head(5)
for k, v in encoded_value_mapping.items():
    print(k, v)

    scaler = MinMaxScaler()

# scaling Age
train_df['Age'] = scaler.fit_transform(train_df[['Age']])
train_df.head(5)

# Correlation Coefficient Matrix
train_df.corr()

X = train_df.drop('treatment', axis=1)
y = train_df['treatment']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=88)

# Train random forest model
rf_model = RandomForestClassifier(random_state=88)
rf_model.fit(X_train, y_train)

# Evaluate feature importance
importances = rf_model.feature_importances_

# Create DataFrame
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Visualize feature importance
plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.show()

# Select features
X = train_df[['work_interfere', 'family_history', 'care_options', 'Age', 'no_employees', 'leave', 'benefits', 'Gender']]
y = train_df[['treatment']].values.ravel()
X

# Models need to be evaluated
models = {
    "Decision Tree" : DecisionTreeClassifier(min_samples_split=20, max_features=8, min_samples_leaf=20, max_depth=6, min_impurity_decrease=0.01),
    "KNN" : KNeighborsClassifier(n_neighbors=17),
    "Logistic Regression": LogisticRegression(),
    "SVC" : SVC(kernel='rbf', random_state=88)
}

# Store model name -> model accuracy 
model_score = {}

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=88)

# Feature standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def evaluate_models(X, y):
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        # Use cross validation to reduce the risk of overfitting
        f1 = cross_val_score(model, X_train, y_train, cv=5, scoring=make_scorer(f1_score)).mean()
        model_score[model_name] = f1
        print("Build", model_name, "model success!")


evaluate_models(X, y)
for model_name, score in model_score.items():
    print(model_name, score)