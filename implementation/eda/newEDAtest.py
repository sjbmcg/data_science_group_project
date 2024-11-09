import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = 'processed_survey_data.csv'
data = pd.read_csv(file_path, encoding='ISO-8859-1')


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

data['Gender'] = data['Gender'].apply(clean_gender)

print(data['Gender'].value_counts())

plt.figure(figsize=(8, 5))
sns.countplot(data=data, x='Gender', order=['Male', 'Female', 'transMale', 'transFemale', 'Other'])
plt.title("Gender Distribution After Cleaning")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

# Check for missing values in each column
print("\nMissing values in each column:")
print(data.isnull().sum())

# Display basic statistics of numeric data
print("\nDescriptive statistics of numeric data:")
print(data.describe())

# Check unique values in categorical columns
print("\nUnique values in categorical columns:")
print(data.select_dtypes(include=['object']).nunique())



# Set plot style
sns.set(style="whitegrid")

# Age distribution plot
plt.figure(figsize=(8, 5))
sns.histplot(data['Age'], kde=True, bins=30)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# # Gender distribution plot
# plt.figure(figsize=(8, 5))
# sns.countplot(data=data, x='Gender')
# plt.title("Gender Distribution")
# plt.xlabel("Gender")
# plt.ylabel("Count")
# plt.show()

# Relationship between company size and mental health benefits
plt.figure(figsize=(12, 6))
sns.countplot(data=data, x='no_employees', hue='benefits')
plt.title("Company Size and Mental Health Benefits")
plt.xlabel("Company Size")
plt.ylabel("Count")
plt.show()

# Analyzing the relationship between family history and mental health treatment
plt.subplot(1, 2, 1)
sns.countplot(data=data, x='family_history', hue='treatment')
plt.title("Family History vs. Mental Health Treatment")
plt.xlabel("Family History (0=No, 1=Yes)")
plt.ylabel("Count")

# Analyzing the relationship between tech company status and mental health treatment
plt.subplot(1, 2, 2)
sns.countplot(data=data, x='tech_company', hue='treatment')
plt.title("Tech Company vs. Mental Health Treatment")
plt.xlabel("Tech Company (0=No, 1=Yes)")
plt.ylabel("Count")

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='family_history', hue='seek_help')
plt.title("Family History vs. Awareness of Mental Health Resources")
plt.xlabel("Family History (0=No, 1=Yes)")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(14, 8))

# Relationship between company size and access to mental health benefits
plt.subplot(1, 2, 1)
sns.countplot(data=data, x='no_employees', hue='benefits')
plt.title("Company Size vs. Access to Mental Health Benefits")
plt.xlabel("Company Size")
plt.ylabel("Count")

# Relationship between tech company status and access to mental health benefits
plt.subplot(1, 2, 2)
sns.countplot(data=data, x='tech_company', hue='benefits')
plt.title("Tech Company vs. Access to Mental Health Benefits")
plt.xlabel("Tech Company (0=No, 1=Yes)")
plt.ylabel("Count")

plt.tight_layout()
plt.show()
