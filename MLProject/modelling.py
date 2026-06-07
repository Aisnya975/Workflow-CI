import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_estimators', type=int, default=100)
parser.add_argument('--max_depth', type=int, default=5)
args = parser.parse_args()

# Load data
df = pd.read_csv('namedataset_preprocessing/dataset_clean.csv')
X = df.drop('Heart_ stroke', axis=1)
y = df['Heart_ stroke']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

mlflow.set_experiment('workflow-ci')

with mlflow.start_run():
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    mlflow.log_param('n_estimators', args.n_estimators)
    mlflow.log_param('max_depth', args.max_depth)
    mlflow.log_metric('accuracy', acc)
    mlflow.log_metric('f1_score', f1)
    mlflow.sklearn.log_model(model, 'model')

    print(f'Accuracy: {acc:.4f}, F1: {f1:.4f}')