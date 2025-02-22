# Churn Prediction Project

## Description
This project focuses on predicting customer churn using a dataset from a credit card company. The goal is to build a machine learning model that can accurately predict whether a customer will leave the company (churn) based on various features such as credit score, age, tenure, balance, and more. The project involves data preprocessing, exploratory data analysis, feature engineering, and model training using a neural network implemented with TensorFlow and Keras.

## Dataset
The dataset used in this project is the `Churn_Modelling.csv` file, which contains information about 10,000 customers. The dataset includes the following features:

- **RowNumber**: Row number in the dataset  
- **CustomerId**: Unique identifier for the customer  
- **Surname**: Customer's surname  
- **CreditScore**: Customer's credit score  
- **Geography**: Country of the customer  
- **Gender**: Gender of the customer  
- **Age**: Age of the customer  
- **Tenure**: Number of years the customer has been with the company  
- **Balance**: Balance in the customer's account  
- **NumOfProducts**: Number of products the customer has purchased  
- **HasCrCard**: Whether the customer has a credit card (1 for yes, 0 for no)  
- **IsActiveMember**: Whether the customer is an active member (1 for yes, 0 for no)  
- **EstimatedSalary**: Estimated salary of the customer  
- **Exited**: Whether the customer has exited (1 for yes, 0 for no)

## Project Steps

### 1. Data Loading and Preprocessing
- Load the dataset using Pandas.
- Drop irrelevant columns (`RowNumber`, `CustomerId`, `Surname`).
- Convert categorical variables (`Geography`, `Gender`) into numerical format using one-hot encoding.
- Split the dataset into training and testing sets.

### 2. Exploratory Data Analysis (EDA)
- Check for missing values and duplicates.
- Analyze the distribution of key features.
- Visualize the correlation between features.

### 3. Feature Scaling
- Standardize the features using `StandardScaler` to normalize the data.

### 4. Model Building
- Build a neural network model using TensorFlow and Keras.
- The model consists of three dense layers with sigmoid activation functions.
- Compile the model using the Adam optimizer and binary cross-entropy loss.

### 5. Model Training
- Train the model on the training dataset.
- Use a validation split to monitor the model's performance during training.

### 6. Model Evaluation
- Evaluate the model on the test dataset.
- Calculate the accuracy of the model.
- Plot the training and validation loss and accuracy over epochs.

### 7. Prediction
- Use the trained model to make predictions on the test dataset.
- Convert the predicted probabilities into binary outcomes (0 or 1).

## Results
- The model achieved an accuracy of approximately **79.75%** on the test dataset.
- The training and validation loss and accuracy plots show that the model converges well, although there is some overfitting as the training accuracy is slightly higher than the validation accuracy.

## Requirements
- **Python 3.x**
- **Libraries**: NumPy, Pandas, Scikit-learn, TensorFlow, Keras, Matplotlib

## Installation
```bash
# Clone the repository
git clone https://github.com/irehmanar/ChurnPrediction.git
cd churn-prediction

# Install the required libraries
pip install numpy pandas scikit-learn tensorflow matplotlib
```

## Usage
1. Open the Jupyter notebook `churn-prediction.ipynb`.
2. Run each cell sequentially to load the data, preprocess it, build and train the model, and evaluate its performance.
3. Modify the model architecture or hyperparameters as needed to improve performance.
