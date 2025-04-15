import pandas as pd
from sklearn.model_selection import train_test_split

def load_student_data(filepath):
    df = pd.read_csv(filepath)
    print(" Data loaded successfully.")
    return df

def explore_student_data(df):
    print("\nInfo about the dataset:")
    print(df.info())
    
    print("\nSummary statistics of the dataset:")
    print(df.describe())
    
    print("\nGradeClass distribution:")
    print(df['GradeClass'].value_counts())

def prepare_data_for_training(df, test_size=0.2, random_state=42):
    X = df.drop(columns=["GradeClass"])  # Features
    y = df["GradeClass"]                 # Target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"\nData split: {len(X_train)} training and {len(X_test)} testing samples.")
    return X_train, X_test, y_train, y_test


