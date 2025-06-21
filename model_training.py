import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_and_evaluate():
    # Load data
    X = pd.read_csv('features.csv')
    y = pd.read_csv('target.csv')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize models
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results = {}
    
    # Train and evaluate models
    for name, model in models.items():
        model.fit(X_train, y_train.values.ravel())
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)
        results[name] = {'accuracy': acc, 'confusion_matrix': cm, 'report': cr}
        
        print(f"\n{name} Performance:")
        print(f"Accuracy: {acc:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        print(f"Classification Report:\n{cr}")
    
    return results

if __name__ == "__main__":
    train_and_evaluate()
