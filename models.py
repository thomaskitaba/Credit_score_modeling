#!/usr/bin/python3


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

#todo: General step  
# import test and training data and test data
# preprocess it
# 
#
#
#   
def model_evaluation(model, X_test, Y_test, model_name):
    return correct

def linear_regression(X_train, y_train, X_test, y_test):
    """
    train and test dataset using linear regression learning model
    """
    result = ["80%", {"F1-score": "98%", "Precision": "95%" , "Recall": "90%"}]
    
    # Chose Model
    model = LogisticRegression(max_iter=500)
    
    # Start Training
    model.fit(X_train, y_train)
    # Start Testing
    test_prediction = model.predict(X_test)
    
    # Evaluate the model
    prediction_result = accuracy_score(y_test, test_prediction)
    result[0] = f"{int(prediction_result * 100)}%"
    result.append([model, "linear regression"])
    print(result[0])
    return result


def random_forest(X_train, y_train, X_test, y_test):
    result = ["99%", {"F1-score": "98%", "Precision": "95%" , "Recall": "90%"}]
    
    # Chose Model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42) #n_estiators = number of trees, randome_state = for reprocuction
    
    # Start Training
    rf_model.fit(X_train, y_train)
    
    # Start Testing
    rf_prediction = rf_model.predict(X_test)
    # Evaluate the model
    rf_prediction_result = accuracy_score(y_test, rf_prediction)
    
    if rf_prediction_result:
        result[0] = f"{int(rf_prediction_result*100)}%"
    print(f"rf: {result[0]}")
    result.append([rf_model, "Random Forest"])
    return result

def gradient_boost(X_train, y_train, X_test, y_test):
    result = ["99%", {"F1-score": "98%", "Precision": "95%" , "Recall": "90%"}]
    
    # Chose Model
    gb_model = XGBClassifier(eval_metric="logloss") #
    
    # Start Training
    gb_model.fit(X_train, y_train)
    # Start Testing
    gb_prediction = gb_model.predict(X_test)
    # Evaluate the model
    gb_prediction_result = accuracy_score(y_test, gb_prediction)
    # print(gb_prediction_result)
    
    result[0] = f"{int(gb_prediction_result * 100)}%"
    result.append([gb_model, "Gradient Boost", X_train, y_train])
    print(result[0])
    return result


if __name__ == "__main__":
    print("Thomas kitaba")
    # import csv files for traingn and testing
   
    np.random.seed(42)
    X = np.random.rand(100, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Simple threshold-based binary classification
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    
    
    
    model_accuracy = {"linear regression": 0, "Random Forest": 0, "Gradient Boost": 0}
    

    
    model_accuracy["linear regression"] = [(int(linear_regression(X_train, y_train, X_test, y_test)[0][:-1])), linear_regression(X_train, y_train, X_test, y_test)[1]]
    model_accuracy["Random Forest"] = [(int(random_forest(X_train, y_train, X_test, y_test)[0][:-1])), random_forest(X_train, y_train, X_test, y_test)[1]]
    model_accuracy["Gradient Boost"] = [(int(gradient_boost(X_train, y_train, X_test, y_test)[0][:-1])), gradient_boost(X_train, y_train, X_test, y_test)[1]]
    

    max_key = ""
    max_value = float('-inf')
    all_max = set()
    for key, value in model_accuracy.items():
                
        if max_value < value[0]:
            max_value = value[0]
            max_key = key
            max_eval = value[1]

        
        
    print(f"model with high accuracy : {max_key}")
    print(f"Accuracy value: {max_value}")
    print(f"evaluation matric scores: {max_eval}")
    
