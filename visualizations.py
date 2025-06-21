import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import os

def generate_visualizations():
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Load data
    df = pd.read_csv('full_data.csv')
    X = pd.read_csv('features.csv')
    y = pd.read_csv('target.csv')
    
    # Feature importance
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y.values.ravel())
    importances = rf.feature_importances_
    features = X.columns
    
    plt.figure(figsize=(12, 8))
    plt.barh(features, importances)
    plt.xlabel('Importance Score')
    plt.title('Feature Importance for Microplastics Risk Prediction')
    plt.tight_layout()
    plt.savefig('results/feature_importance.png', dpi=300)
    plt.show()
    
    # Risk distribution
    plt.figure(figsize=(10, 6))
    risk_counts = df['risk_level'].value_counts().sort_index()
    risk_counts.plot(kind='bar', color=['#4CAF50', '#FFC107', '#F44336'])
    plt.title('Distribution of Risk Levels in Population')
    plt.xlabel('Risk Level')
    plt.ylabel('Count')
    plt.xticks(ticks=[0, 1, 2], labels=['Low', 'Medium', 'High'], rotation=0)
    plt.savefig('results/risk_distribution.png', dpi=300)
    plt.show()
    
    # Exposure vs risk
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='risk_level', y='mp_score', data=df, palette=['#4CAF50', '#FFC107', '#F44336'])
    plt.title('Microplastic Exposure Score by Risk Level')
    plt.xlabel('Risk Level')
    plt.ylabel('Exposure Score')
    plt.xticks(ticks=[0, 1, 2], labels=['Low', 'Medium', 'High'])
    plt.savefig('results/exposure_vs_risk.png', dpi=300)
    plt.show()
    
    # Correlation matrix
    plt.figure(figsize=(12, 8))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.savefig('results/correlation_matrix.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    generate_visualizations()
