#!/usr/bin/python3


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import lime.lime_tabular, shap, dice_ml
# from mpl_toolkits.mplot3d import Axes3D

from sklearn.datasets import make_classification

#todo: General step  
# import test and training data and test data
# preprocess it
# 
#
#
#   

def shap_xai(model_evaluation_list):
    model, X_test, y_test, model_name = model_evaluation_list
    explainer = None
    if model_name == "Linear Regression":
        explainer = shap.LinearExplainer(model, X_test)
    if model_name == "Random Forest":
        explainer = shap.TreeExplainer(model)
    if model_name == "Gradient Boost":
        explainer = shap.TreeExplainer(model)
        
    # Compute contriution of each feature to the models prediction
    shap_values = explainer.shap_values(X_test)
    feature_names = [f'Feature_{i}' for i in range(X_test.shape[1])]
    
    # X_train = pd.DataFrame(X_train, columns=feature_names)
    X_test = pd.DataFrame(X_test, columns=feature_names)
    # Visualize Feature importance
    shap.summary_plot(shap_values, X_test, feature_names=X_test.columns)
    
def lime_xai(model_evaluation_list):
    model, X_test, y_test, model_name = model_evaluation_list
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_test, mode="classification"
    )
    explanation = explainer.explain_instance(X_test[0], model.predict_proba)
    explanation.show_in_notebook()

def dice_xai(model_evaluation_list):
    model, X_test, y_test, model_name = model_evaluation_list   
    return "Thomas Kitaba"



def model_evaluation(model_evaluation_list):
    model, X_test, y_test, model_name = model_evaluation_list
    result = {}
    
    # Start testing and Evaluation
    model_predicted = model.predict(X_test)
    model_predicted_prob = model.predict_proba(X_test)[:, 1]  # Get predicted probabilities for ROC-AUC

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, model_predicted)
    F1_score = f1_score(y_test, model_predicted)
    Precision_score = precision_score(y_test, model_predicted)
    Recall_score = recall_score(y_test, model_predicted)
    
    Roc_auc_score = roc_auc_score(y_test, model_predicted_prob)
    
    result["model_name"] = f"{model_name}"
    result["accuracy"] = accuracy
    result["f1_score"] = F1_score
    result["precision_score"] = Precision_score
    result["recall_score"] = Recall_score
    result["roc_auc_score"] = Roc_auc_score
    return result
    
    
    return("Thomas Kitaba")
    

def linear_regression(X_train, y_train):
    """
    train and test dataset using linear regression learning model
    X_train: training data
    y_train: target data
    """
    result = []
    
    # Chose Model
    lgr_model = LogisticRegression(max_iter=500)

    # Start Training
    lgr_model.fit(X_train, y_train)
    
        
    return lgr_model


def random_forest(X_train, y_train):
    """
    Train and test dataset using Random Forest learning model
    X_train: training data
    y_train: target data
    """
    # Chose Model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42) #n_estiators = number of trees, randome_state = for reprocuction
    
    # Start Training
    rf_model.fit(X_train, y_train)
    return rf_model  # This is the trained model
    
def gradient_boost(X_train, y_train):
    """
    Train and test dataset using Gradient Boosting learning model
    X_train: training data
    y_train: target data
    """
    # Chose Model
    gb_model = XGBClassifier(eval_metric="logloss") #
    # Start Training
    gb_model.fit(X_train, y_train)
    # Start Testing
    return gb_model

if __name__ == "__main__":
    print("Thomas kitaba")
   
    # np.random.seed(42)
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                           n_redundant=5, n_clusters_per_class=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
     # import csv files for traingn and testing
    # training_dataframe = pd.read_csv('cs-training.csv', na_values='NA')
    # test_dataframe = pd.read_csv('cs-test.csv', na_values='NA')
    
    # X_train = training_dataframe.iloc[:, :-1] # This are the training Features
    # y_train = training_dataframe.iloc[:, -1]  # This are the test lables
    
    # X_test = test_dataframe.iloc[:, :-1] # This are the test Features
    # y_test = test_dataframe.iloc[:, -1] # This are the test lables
    #initialize an empty list to hold the three trained models
    model_evaluation_list = []
    
    # Train model using Linear Regression
    model_evaluation_list.append([linear_regression(X_train, y_train), X_test, y_test, "Linear Regression"]) # recive only model name

    # Train Model Using Randome Forest
    model_evaluation_list.append([random_forest(X_train, y_train), X_test, y_test, "Random Forest"]) # recive only model name
    
    # Train model using Gradient boost
    model_evaluation_list.append([gradient_boost(X_train, y_train), X_test, y_test, "Gradient Boost"])
    
    all_models = []
    for model in model_evaluation_list:
        model_evaluation_results = model_evaluation(model)
        all_models.append(model_evaluation_results)
        print(model_evaluation_results)
        # shap_xai(model)
        lime_xai(model)

    
