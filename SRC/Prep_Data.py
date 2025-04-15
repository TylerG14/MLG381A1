import pandas as pd
from sklearn.model_selection import train_test_split
if __name__ == "__main__":
    def load_and_split_data(filepath, test_size=0.2, random_state=42):
        data = pd.read_csv(filepath)
        print(f" Data loaded successfully. Total records: {len(data)}")

        # Split features and target
        X = data.drop(columns=['GradeClass'])
        y = data['GradeClass']

        # Use exact test size for 479 records out of 2392
        test_ratio = 479 / 2392

        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_state, stratify=y
        )

        print(f"Data split: {len(X_train)} training, {len(X_test)} testing samples.")
        return X_train, X_test, y_train, y_test
