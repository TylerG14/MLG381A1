from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from Evaluations import evaluate_classification_model

def train_and_evaluate_models(X_train, X_test, y_train, y_test):

    #Train and test 3 models: Logistic Regression, Random Forest, and XGBoost.

    print("\n--- Model Training and Evaluation ---")

    # Train and test Logistic Regression
    
