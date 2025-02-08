#!/usr/bin/python3

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from sklearn.preprocessing import StandardScaler

import lime.lime_tabular, shap, dice_ml 

import matplotlib.pyplot as plt

# from mpl_toolkits.mplot3d import Axes3D

from sklearn.datasets import make_classification


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
    # feature_names = [f'Feature_{i}' for i in range(X_test.shape[1])]
    feature_names = [
    "RevUtil",      # RevolvingUtilizationOfUnsecuredLines
    "Age",          # age
    "PastDue30_59", # NumberOfTime30-59DaysPastDueNotWorse
    "DebtRatio",    # DebtRatio
    "Income",       # MonthlyIncome
    "OpenCredits",  # NumberOfOpenCreditLinesAndLoans
    "Late90",       # NumberOfTimes90DaysLate
    "RELoans",      # NumberRealEstateLoansOrLines
    "PastDue60_89", # NumberOfTime60-89DaysPastDueNotWorse
    "Dependents"    # NumberOfDependents
]
    
    # X_train = pd.DataFrame(X_train, columns=feature_names)
    X_test = pd.DataFrame(X_test, columns=feature_names)
    # Visualize Feature importance
    shap.summary_plot(shap_values, X_test, feature_names=X_test.columns)
    
def lime_xai(model_evaluation_list):
    model, X_test, y_test, model_name = model_evaluation_list
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_test, mode="classification"
    )
    
    # convert
    # explanation = explainer.explain_instance(X_test[0], model.predict_proba)
    
    # feature_names = [f'Feature_{i}' for i in range(X_test.shape[1])]
    feature_names = [
    "RevUtil",      # RevolvingUtilizationOfUnsecuredLines
    "Age",          # age
    "PastDue30_59", # NumberOfTime30-59DaysPastDueNotWorse
    "DebtRatio",    # DebtRatio
    "Income",       # MonthlyIncome
    "OpenCredits",  # NumberOfOpenCreditLinesAndLoans
    "Late90",       # NumberOfTimes90DaysLate
    "RELoans",      # NumberRealEstateLoansOrLines
    "PastDue60_89", # NumberOfTime60-89DaysPastDueNotWorse
    "Dependents"    # NumberOfDependents
]

    # Example usage:
    # Explain a single prediction (e.g., the first instance)
    # todo: the LimeTabularExplainer expects the training data to be a numpy array, not a pandas DataFrame.
    instance =  np.array(X_test)[0]# Select the first instance in the dataset
    
    print("**********************************************")
    
    exp = explainer.explain_instance(
    instance,  # The instance to explain
    model.predict_proba,  # Use the model's probability predictions
    num_features=5  # Include all features in the explanation
)
    # explanation.show_in_notebook()
    predicted_class = np.argmax(exp.predict_proba)  # Get class with highest probability
    # Print Explanation as Text
    
    print("-------------------LIME----------------------------------")
    if model_name == "Linear Regression":
        print("\nLIME text explanation for Linear Regression")
    if model_name == "Random Forest":
        print("\nLIME text explanation for Random Forest")
    if model_name == "Gradient Boost":
        print("\nLIME text explanation forGradient Boost")
   
    print("Predicted class:", predicted_class)
    print("Predicted probabilities:", exp.predict_proba)
    
    # Print Feature Contributions
    print("\nFeature contributions (feature importance):")
    # for feature, weight in exp.as_list():
    #     print(f"{feature}: {weight:.4f}")
    for count, (feature, weight) in enumerate(exp.as_list()):
        print(f"{feature_names[count]}. : {weight:.2f}")
    exp.as_pyplot_figure()
    
    plt.show()

    # # To have more control over the visuslization use this Create a Matplotlib visualization
    # explanation_data = exp.as_list()  # Get explanation as a list of (feature, weight) tuples
    # plt.figure(figsize=(10, 6))  # Set figure size
    # features, weights = zip(*explanation_data)  # Unpack features and weights
    # colors = ['green' if w > 0 else 'red' for w in weights]  # Color code positive/negative contributions
    # # Plot the feature weights
    # plt.barh(features, weights, color=colors)
    # plt.xlabel('Weight (Contribution to Prediction)')  # Label for x-axis
    # plt.ylabel('Features')  # Label for y-axis
    # plt.title('LIME Explanation for Instance Prediction')  # Title of the plot
    # plt.grid(axis='x', linestyle='--', alpha=0.7)  # Add gridlines for better readability
    # # Show the plot
    # plt.tight_layout()
    # plt.show()


def dice_xai(model_evaluation_list):
    model, X_test, y_test, model_name = model_evaluation_list   
    feature_names = [
    "RevUtil",      # RevolvingUtilizationOfUnsecuredLines
    "Age",          # age
    "PastDue30_59", # NumberOfTime30-59DaysPastDueNotWorse
    "DebtRatio",    # DebtRatio
    "Income",       # MonthlyIncome
    "OpenCredits",  # NumberOfOpenCreditLinesAndLoans
    "Late90",       # NumberOfTimes90DaysLate
    "RELoans",      # NumberRealEstateLoansOrLines
    "PastDue60_89", # NumberOfTime60-89DaysPastDueNotWorse
    "Dependents"    # NumberOfDependents
]
    
    # prepare data
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    X_test_df['target'] = y_test

    # Create the DiCE Data object using the proper parameter name "dataframe"
    d = dice_ml.Data(
        dataframe=X_test_df,
        continuous_features=feature_names,
        outcome_name='target'
    )
    m = dice_ml.Model(model=model, backend='sklearn')
    
    # create explainer
    exp = dice_ml.Dice(d, m)
    
    # select the fires instance
    instance = X_test_df.iloc[[1]]
    
    # generate explanation
    generated_explanation = exp.generate_counterfactuals(instance, total_CFs=4, desired_class=1)
    
    generate_counterfactuals.visualize_as_dataframe()
    
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
   
    # Sample Data for test
    # np.random.seed(42)
    # X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
    #                        n_redundant=5, n_clusters_per_class=2, random_state=42)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Use CSV Data Only ---
    df = pd.read_csv('cs-training.csv')

    # Separate features and labels
    X = df.iloc[:, :-1]  # All columns except the last one are features
    y = df.iloc[:, -1]   # The last column is the target label

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the data (to avoid convergence issues)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #initialize an empty list to hold the three trained models
    model_evaluation_list = []
    
    # Train model using Linear Regression
    model_evaluation_list.append([linear_regression(X_train, y_train), X_test, y_test, "Linear Regression"]) # recive only model name

    # Train Model Using Randome Forest
    model_evaluation_list.append([random_forest(X_train, y_train), X_test, y_test, "Random Forest"]) # recive only model name
    
    # Train model using Gradient boost
    model_evaluation_list.append([gradient_boost(X_train, y_train), X_test, y_test, "Gradient Boost"])
    
    for model in model_evaluation_list:
        model_evaluation_results = model_evaluation(model)
        # print("-------------------Evaluation Matric----------------------------------")
        # print(model_evaluation_results)
        # shap_xai(model)
        # lime_xai(model)
        dice_xai(model)
        print("===========================================")
        
    
