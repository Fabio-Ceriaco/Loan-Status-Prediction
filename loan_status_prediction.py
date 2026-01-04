# Model Used: Support Vector Machine (SVM)

# Work Flow:
# 1. Importing Libraries
# 2. Loading Dataset
# 3. Exploratory Data Analysis
# 4. Data Preprocessing
# 5. Splitting the Data
# 6. Splitting the Data into Training and Testing Sets
# 7. Model Training
# 8. Model Evaluation
# 9. Making Predictions


# DataSet Structure:
# The dataset contains the following columns:
# Loan_ID: Unique Loan ID
# Gender: Male/Female
# Married: Applicant married (Yes/No)
# Dependents: Number of dependents
# Education: Applicant Education (Graduate/Not Graduate)
# Self_Employed: Self employed (Yes/No)
# ApplicantIncome: Applicant income in thousands of dollars
# CoapplicantIncome: Coapplicant income in thousands of dollars
# LoanAmount: Loan amount in thousands of dollars
# Loan_Amount_Term: Term of loan in months
# Credit_History: credit history meets guidelines
# Property_Area: Urban/Semi-Urban/Rural
# Loan_Status: Loan approved (Y/N)

# =====================================================================================#
#                           1. Importing Libraries                                    #
# =====================================================================================#

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt


# =====================================================================================#
#                           2. Loading Dataset                                        #
# =====================================================================================#

data_df = pd.read_csv("personal_loan_data.csv")


# =====================================================================================#
#                     3. Exploratory Data Analysis                                    #
# =====================================================================================#

# Display first few rows of the dataset

print(data_df.head())

# Display Shape of the dataset

print(data_df.shape)

# Check for missing values

print(data_df.isnull().sum())

# In this case I have to drop the missing values because as the dataset have missing values in multiple columns
# with categorical data, it would be difficult to impute them accurately.
# The imputation method is used to numerical data. Instead of that I will drop the rows with missing values to ensure data integrity.


# Drop rows with missing values

data_df = data_df.dropna()

# After dropping missing values, check the shape again

print(data_df.shape)

# Check the missing values again to confirm

print(data_df.isnull().sum())

# Display information about the dataset

print(data_df.info())

# Display statistical summary of the dataset

print(data_df.describe())

# =====================================================================================#
#                           4. Data Preprocessing                                      #
# =====================================================================================#

# Label Encoding for Loan_Status and Dependents columns

data_df.replace({"Loan_Status": {"Y": 1, "N": 0}}, inplace=True)

print(data_df.head())

# Visualize the distribution of 'Dependents' column

print(data_df["Dependents"].value_counts())

# Replace '3+' with 4 in 'Dependents' column for numerical processing

data_df.replace({"Dependents": {"3+": 4}}, inplace=True)

# Data Visualization

# Education vs Loan Status

# x - axis: Education
# hue - Loan_Status, this will show the distribution of loan status for each education level hue means different colors for different categories
# data - data_df, the dataframe containing the data

sns.countplot(x="Education", hue="Loan_Status", data=data_df)
plt.show()

# Marital Status vs Loan Status

sns.countplot(x="Married", hue="Loan_Status", data=data_df)
plt.show()

# Gender vs Loan Status

sns.countplot(x="Gender", hue="Loan_Status", data=data_df)
plt.show()

# Self Employed vs Loan Status

sns.countplot(x="Self_Employed", hue="Loan_Status", data=data_df)
plt.show()

# Property Area vs Loan Status

sns.countplot(x="Property_Area", hue="Loan_Status", data=data_df)
plt.show()

# Encoding Categorical Variables
# We can do this encoding as before using replace method or we can use LabelEncoder from sklearn
# Here, I will use LabelEncoder for variety and simplicity

label_encoder = LabelEncoder()

categorical_columns = [
    "Gender",
    "Married",
    "Education",
    "Self_Employed",
    "Property_Area",
]

for column in categorical_columns:
    data_df[column] = label_encoder.fit_transform(data_df[column])

print(data_df.head())

# Check values of Property_Area after encoding

print(data_df["Property_Area"].value_counts())


# =====================================================================================#
#                               5. Splitting the Data                                  #
# =====================================================================================#

X = data_df.drop(columns=["Loan_ID", "Loan_Status"], axis=1)
y = data_df["Loan_Status"]

print(X)
print(y)


# ======================================================================================#
#                 6. Splitting the Data into Training and Testing Sets                  #
# ======================================================================================#

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=2
)

print(X.shape, X_train.shape, X_test.shape)
print(y.shape, y_train.shape, y_test.shape)


# =====================================================================================#
#                               7. Model Training                                      #
# =====================================================================================#

# Using linear kernel for SVM linear Kernel means the decision boundary will be a straight line (or hyperplane in higher dimensions)
model = SVC(kernel="linear")

model.fit(X_train, y_train)


# =====================================================================================#
#                               8. Model Evaluation                                    #
# =====================================================================================#

# Training Data Accuracy

X_train_prediction = model.predict(X_train)

training_data_accuracy = accuracy_score(y_train, X_train_prediction)

print("Accuracy on Training data : ", training_data_accuracy)

# Testing Data Accuracy

X_test_prediction = model.predict(X_test)

testing_data_accuracy = accuracy_score(y_test, X_test_prediction)

print("Accuracy on Testing data : ", testing_data_accuracy)

# Overfitting Check
# Overfitting occurs when a model learns the training data too well, including its noise and outliers, leading to poor generalization on new data.
# Underfitting occurs when a model is too simple to capture the underlying patterns in the data, resulting in poor performance on both training and testing data.
# To check for overfitting or underfitting, we can compare the training and testing accuracies.

if training_data_accuracy > testing_data_accuracy:
    print("The model is overfitting the training data.")
elif training_data_accuracy < testing_data_accuracy:
    print("The model is underfitting the training data.")
else:
    print("The model is performing well on both training and testing data.")


# =====================================================================================#
#                               9. Making Predictions                                  #
# =====================================================================================#

input_data = (1, 0, 5849, 0, 128, 360, 1, 1, 1, 0, 2)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)

if prediction[0] == 0:
    print("The person is not eligible for loan.")
else:
    print("The person is eligible for loan.")
