import pandas as pd
from utils import load_and_fillna
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from os.path import exists
from pathlib import Path
from xgboost import XGBClassifier
import joblib


# Logistic Regression
lr_model = Pipeline([("model", LogisticRegression(class_weight="balanced", solver="liblinear", random_state=42))])
# SVM
svm_model = Pipeline([("model", svm.SVC(kernel="linear", class_weight="balanced"))])
# Decision Tree
dt_model = Pipeline([("model", DecisionTreeClassifier(class_weight="balanced"))])
# Random Forest
rf_model = Pipeline([("model", RandomForestClassifier(class_weight="balanced", n_estimators=200, n_jobs=-1))])
# XGBoost
xgb_model = Pipeline([ # Add a scale_pos_weight to make it balanced : 1 - y.mean()
                        ("model", XGBClassifier(scale_pos_weight=(0.5), n_jobs=-1))])


Path("./models/").mkdir(parents=True, exist_ok=True)


def run_svm(Xtrain, Xtest, Ytrain, Ytest, name):
    print("Starting SVM")
    gs = GridSearchCV(svm_model, {"model__C": [0.001, 0.1, 1],
                                 "model__gamma": [0.001, 0.01, 0.1, 1]}, n_jobs=-1, cv=5, scoring="accuracy")
    gs.fit(Xtrain, Ytrain)

    print(gs.best_params_)
    print(gs.best_score_)

    svm_model.set_params(**gs.best_params_)

    svm_model.fit(Xtrain, Ytrain)

    joblib.dump(svm_model, f"models/{name}_SVM.mdl")

    print(f"Done training, model saved to model/{name}_SVM.mdl")



def run_logistic_regression(Xtrain, Xtest, Ytrain, Ytest, name):
    print("Starting Logistic Regression")
    gs = GridSearchCV(lr_model, {"model__C": [1, 1.3, 1.5]}, n_jobs=-1, cv=5, scoring="accuracy")
    gs.fit(Xtrain, Ytrain)

    print(gs.best_params_)
    print(gs.best_score_)

    lr_model.set_params(**gs.best_params_)

    lr_model.fit(Xtrain, Ytrain)

    joblib.dump(lr_model, f"models/{name}_LR.mdl")

    print(f"Done training, model saved to model/{name}_LR.mdl")



def run_random_forest(Xtrain, Xtest, Ytrain, Ytest, name):
    print("Starting Random Forest")
    gs = GridSearchCV(rf_model, {"model__max_depth": [10, 15, 20],
                                 "model__min_samples_split": [5, 10]},
                      n_jobs=-1, cv=5, scoring="accuracy")

    gs.fit(Xtrain, Ytrain)

    print(gs.best_params_)
    print(gs.best_score_)

    rf_model.set_params(**gs.best_params_)

    rf_model.fit(Xtrain, Ytrain)
    joblib.dump(rf_model, f"models/{name}_RF.mdl")

    print(f"Done training, model saved to model/{name}_RF.mdl")


def run_decision_trees(Xtrain, Xtest, Ytrain, Ytest, name):
    print("Starting Decision Trees")
    gs = GridSearchCV(dt_model, {"model__max_depth": [3, 5, 7],
                                 "model__min_samples_split": [2, 5]},
                      n_jobs=-1, cv=5, scoring="accuracy")

    gs.fit(Xtrain, Ytrain)

    print(gs.best_params_)
    print(gs.best_score_)

    dt_model.set_params(**gs.best_params_)

    dt_model.fit(Xtrain, Ytrain)
    joblib.dump(dt_model, f"models/{name}_DT.mdl")

    print(f"Done training, model saved to model/{name}_DT.mdl")


def run_xgboost(Xtrain, Xtest, Ytrain, Ytest, name):
    print("Starting XGBoost")

    gs = GridSearchCV(xgb_model, {"model__max_depth": [5, 10],
                                "model__min_child_weight": [5, 10],
                                "model__n_estimators": [25]},
                    n_jobs=-1, cv=5, scoring="accuracy")

    gs.fit(Xtrain, Ytrain)

    print(gs.best_params_)
    print(gs.best_score_)

    xgb_model.set_params(**gs.best_params_)
    xgb_model.fit(Xtrain, Ytrain)
    joblib.dump(xgb_model, f"models/{name}_XGB.mdl")
    print(f"Done training, model saved to model/{name}_XGB.mdl")


def train_all(Xtrain, Xtest, Ytrain, Ytest, name):
    run_svm(Xtrain, Xtest, Ytrain, Ytest, name)
    run_decision_trees(Xtrain, Xtest, Ytrain, Ytest, name)
    run_logistic_regression(Xtrain, Xtest, Ytrain, Ytest, name)
    run_random_forest(Xtrain, Xtest, Ytrain, Ytest, name)
    run_xgboost(Xtrain, Xtest, Ytrain, Ytest, name)
