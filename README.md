# PRODIGY_ML_01

Task 01 - Implement a linear regression model to predict the prices of houses based on their square footage and the number of bedrooms and bathrooms.
Given Dataset - https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

The python file named as - task01.py does all the task mentioned below.
To tackle the task of predicting house prices based on square footage, the number of bedrooms, and the number of bathrooms, I followed a systematic approach using a linear regression model. Here’s a step-by-step overview of how I completed this task:
1. Data Loading: I began by loading the training and test datasets from CSV files using pandas. The datasets contained information on house features and their sale prices.
2. Feature and Target Definition: I specified the features for the model as square footage (GrLivArea), number of bedrooms (BedroomAbvGr), and number of bathrooms (FullBath). The target variable I aimed to predict was the house sale price (SalePrice).
3. Data Preparation: I extracted the relevant features and target variable from the training data. For the test data, I selected the same features but did not include the target variable as it was the focus of prediction.
4. Data Splitting: To evaluate the model’s performance, I split the training data into training and validation sets using a 80-20 split. This allowed me to train the model on one subset of the data and test its performance on another.
5. Feature Standardization: I standardized the features using StandardScaler to ensure that they had zero mean and unit variance. This step is crucial for linear regression models as it helps in achieving better convergence and performance.
6. Model Training: I initialized a LinearRegression model and trained it on the scaled training data. The model learned the relationship between the features and the target variable.
7. Making Predictions: After training, I used the model to make predictions on the training, validation, and test datasets. This allowed me to assess how well the model performed and to generate predictions for the test set.
8. Model Evaluation: I evaluated the model’s performance by calculating the Mean Squared Error (MSE) and R² score for both the training and validation datasets. These metrics provided insights into how well the model predicted house prices.
9. Submission Preparation: For the final step, I prepared a submission file containing the predicted house prices for the test dataset. This file was saved in CSV format for easy submission.
10. Visualization: To visualize the model’s performance, I plotted scatter plots comparing the true vs. predicted house prices for both the training and validation datasets. This helped in understanding the accuracy of the predictions visually.

By following these steps, I successfully implemented a linear regression model to predict house prices based on the given features. The plotting of the graph shows the accuracy of the model trained which is near-about 85% that means the model is well trained.
