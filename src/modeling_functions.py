import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, make_scorer, f1_score
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from prettytable import PrettyTable

def recall_score_class3(y_true, y_pred):
    return recall_score(y_true, y_pred, average=None)[2]

def scorer_recall3():
    scorer = make_scorer(recall_score_class3)
    return scorer

def f1_score_class3(y_true, y_pred):
    return f1_score(y_true, y_pred, average=None)[2]

def scorer_f13():
    scorer = make_scorer(f1_score_class3)
    return scorer

def plot_confusion_matrices(model, X_test, y_test, X_val, y_val, X_train, y_train):
    """Plots confusion matrix for test validation and train data
    based on the model
    order of inputs: model, test- X y, Val- X y, and Train- X y
    """
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 6), sharey=True)

    axes[0].set_title("Test Data",fontsize=14, pad=20)
    axes[1].set_title("Validation Data", fontsize=14, pad=20)
    axes[2].set_title("Train Data", fontsize=14, pad=20)

    plot_confusion_matrix(model, X_test, y_test, ax=axes[0], cmap='cividis',
                         normalize='true', display_labels=['Grade 1','Grade 2','Grade 3'])
    plot_confusion_matrix(model, X_val, y_val, ax=axes[1], cmap='cividis',
                         normalize='true', display_labels=['Grade 1','Grade 2','Grade 3'])
    plot_confusion_matrix(model, X_train, y_train, ax=axes[2], cmap='cividis',
                         normalize='true', display_labels=['Grade 1','Grade 2','Grade 3'])
    
    for i in range(3):
        axes[i].tick_params(labelcolor='olive', labelsize=12, pad=5)
        axes[i].set_xlabel('Predicted Damage Grade', fontsize=12, labelpad=14,fontstyle='italic', color='grey')
        axes[i].set_ylabel('True Damage Grade', fontsize=12, labelpad=14, fontstyle='italic', color='grey')
    fig.tight_layout()
    return plt.show();


def print_model_eval_table():    
    x = PrettyTable()

    x.field_names = ["Model", "Recall Score for Damage Grade 3"]

    x.add_row(["Random Predictor", '33%'])
    x.add_row(["Logistic Regression Model", '64%'])
    x.add_row(["Random Forest Classifier",'67%'])
    x.add_row(["XGBoost Model", '66%'])

    return print(x)


def plot_important_features(important_features):
    
    sorted_features = sorted(important_features, key=lambda x: x[1])

    importance = []
    feature_names = []

    for i in sorted_features:
        feature_names.append(i[0])
        importance.append(i[1])

    plt.figure(figsize=(8, 16))
    n_features = 76    
    plt.barh(range(n_features)[40:], importance[40:])
    plt.yticks(np.arange(n_features)[40:], feature_names[40:])
    plt.xlabel('Feature importance', fontsize= 14)
    plt.ylabel('Feature', fontsize= 14)
    
    return plt.show()