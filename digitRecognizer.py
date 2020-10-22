# Shahriyar Mammadli
# Import required libraries
import pandas as pd
import helperFunctions as hf
from sklearn.model_selection import train_test_split

# Read the train and test data
trainDf = pd.read_csv('../Datasets/KaggleDigitRecognizer/train.csv')
predDf = pd.read_csv('../Datasets/KaggleDigitRecognizer/test.csv')

# Split train, test
# For the sake of simplicity, in this version, we skip validation skip
X_train, X_test, y_train, y_test = train_test_split(trainDf.drop(trainDf.columns[0], 1), trainDf[trainDf.columns[0]], test_size=0.1, random_state=123)

# Build a CNN model
CNNSubmitDf = hf.buildCNNModel(X_train, X_test, y_train, y_test, X_train.shape[0], X_test.shape[0], predDf)
CNNSubmitDf.to_csv("submissionCNN.csv", header=True, index=False)
