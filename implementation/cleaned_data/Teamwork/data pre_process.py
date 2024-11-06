import pandas as pd
from sklearn.preprocessing import LabelEncoder

file_path = './survey.csv'
survey_data = pd.read_csv(file_path)

# Filling missing values in 'self_employed' and 'state' with 'Unknown'
survey_data['self_employed'] = survey_data['self_employed'].fillna('Unknown')
survey_data['state'] = survey_data['state'].fillna('Unknown')

# Replacing various gender representations with standardized labels
survey_data['Gender'] = survey_data['Gender'].replace(
    ['M', 'Male', 'male', 'm', 'Cis Male', 'Man'], 'Male'
)
survey_data['Gender'] = survey_data['Gender'].replace(
    ['F', 'Female', 'female', 'f', 'Cis Female', 'Woman'], 'Female'
)
survey_data['Gender'] = survey_data['Gender'].replace(
    ['Trans-female', 'Trans woman', 'Trans Female'], 'Trans Female'
)
survey_data['Gender'] = survey_data['Gender'].replace(
    ['Trans-male', 'Trans man', 'Trans Male'], 'Trans Male'
)
survey_data['Gender'] = survey_data['Gender'].replace(
    ['Non-binary', 'non-binary', 'Genderqueer', 'Other'], 'Non-binary/Other'
)
print(survey_data)

#Convertinng Timestamp to data format.
survey_data['Timestamp'] = pd.to_datetime(survey_data['Timestamp'])

#Dropping unreasonable Age data
survey_data = survey_data[(survey_data['Age'] >= 18) & (survey_data['Age'] <= 100)]

#Encoding the columns have 'yes' 'no' 'unknow' to '1' '2' '3'  (example)


label_encoder = LabelEncoder()# labelencoder it's a kind of method to process 'yes or no' changed to number for modeling

categorical_columns = [
    'self_employed', 'family_history', 'treatment', 'work_interfere',
    'leave', 'mental_health_consequence', 'phys_health_consequence',
    'coworkers', 'supervisor', 'mental_health_interview',
    'phys_health_interview', 'mental_vs_physical', 'obs_consequence'
]


for column in categorical_columns:
    survey_data[column] = label_encoder.fit_transform(survey_data[column].astype(str))

survey_data.to_csv('./processed_survey_data.csv', index=False)
