import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline


# Function to train and evaluate classifiers
def train_and_evaluate(classifiers, X_train, X_test, y_train, y_test):
    results = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        print(f"{name}: Accuracy = {accuracy:.2f}")
    return results


# Define test data
def test_train_and_evaluate():
    iris_df = pd.read_csv("iris.csv")
    X = iris_df.drop(columns=['species'])
    y = iris_df['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize classifiers
    classifiers = {
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(),
        'Naive Bayes': GaussianNB(),
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier()
    }

    # Train and evaluate classifiers
    results = train_and_evaluate(classifiers, X_train, X_test, y_train, y_test)

    # Check if all classifiers are present in results
    assert all(name in results for name in classifiers)

    # Check if all accuracy scores are between 0 and 1
    assert all(0 <= accuracy <= 1 for accuracy in results.values())

    # Check if results are sorted in descending order of accuracy
    sorted_results = sorted(results.values(), reverse=True)
    assert list(results.values()) == sorted_results


# Run the test
test_train_and_evaluate()
