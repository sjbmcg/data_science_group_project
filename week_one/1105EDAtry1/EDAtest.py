# 导入所需的库
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
data = pd.read_csv('survey.csv')

# 1. 性别归类函数
def standardize_gender(gender):
    if pd.isnull(gender):
        return 'Non-binary'  # 合并 Unknown 和 Non-binary
    gender = gender.strip().lower()
    if gender in ['male', 'm', 'man', 'cis male', 'cis-man', 'male-ish']:
        return 'Male'
    elif gender in ['female', 'f', 'woman', 'cis female', 'cis-woman']:
        return 'Female'
    elif 'trans' in gender and 'male' in gender:
        return 'Trans Male'
    elif 'trans' in gender and 'female' in gender:
        return 'Trans Female'
    elif 'non-binary' in gender or 'genderqueer' in gender:
        return 'Non-binary'
    else:
        return 'Non-binary'  # 合并其他类别为 Non-binary

# 应用函数对 Gender 列进行处理
data['Gender'] = data['Gender'].apply(standardize_gender)

# 显示性别分布
print("Gender distribution:\n", data['Gender'].value_counts())

# 2. 检查和处理年龄异常值
# 计算四分位数和 IQR
Q1 = data['Age'].quantile(0.25)
Q3 = data['Age'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 找出异常值
outliers = data[(data['Age'] < lower_bound) | (data['Age'] > upper_bound)]
print("Outliers:\n", outliers[['Age']])

# 移除异常值
cleaned_data = data[(data['Age'] >= lower_bound) & (data['Age'] <= upper_bound)]
print("Data size after removing outliers:", cleaned_data.shape)

# 3. EDA 分析

# 性别分布柱状图
plt.figure(figsize=(8, 5))
sns.countplot(x='Gender', data=cleaned_data, hue='Gender', palette='Set2', dodge=False, legend=False)
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()


# 年龄分布直方图
plt.figure(figsize=(10, 6))
sns.histplot(cleaned_data['Age'], bins=20, kde=True, color='blue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# # 相关性热力图
# # 只选择数值型列
# numeric_data = cleaned_data.select_dtypes(include=['number'])
# correlation_matrix = numeric_data.corr()
#
# plt.figure(figsize=(12, 10))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
# plt.title('Correlation Heatmap')
# plt.show()

# 工作干扰和治疗的交互分析
plt.figure(figsize=(10, 6))
sns.countplot(x='work_interfere', hue='treatment', data=cleaned_data, palette='pastel')
plt.title('Work Interference vs Treatment')
plt.xlabel('Work Interference')
plt.ylabel('Count')
plt.legend(title='Treatment')
plt.show()
