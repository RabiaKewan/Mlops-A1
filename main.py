import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


iris_df_original = pd.read_csv("iris.csv")

X_original = iris_df_original.drop(columns=['species'])
y_original = iris_df_original['species']


X_train_original, X_test_original, y_train_original, y_test_original = train_test_split(X_original, y_original, test_size=0.2, random_state=42)

# Initialize classifiers
classifiers = {
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(),
    'Naive Bayes': GaussianNB(),
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier()
}

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


print("Results on original dataset:")
original_results = train_and_evaluate(classifiers, X_train_original, X_test_original, y_train_original, y_test_original)



scaler = StandardScaler()
X_preprocessed = scaler.fit_transform(X_original)


X_train_preprocessed, X_test_preprocessed, y_train_preprocessed, y_test_preprocessed = train_test_split(X_preprocessed, y_original, test_size=0.2, random_state=42)

print("\nResults on pre-processed dataset:")
preprocessed_results = train_and_evaluate(classifiers, X_train_preprocessed, X_test_preprocessed, y_train_preprocessed, y_test_preprocessed)
