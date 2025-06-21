import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Simulate realistic microplastic exposure data
def gen_data(n_samples=500):
    np.random.seed(42)
    
    Water_Intake = np.random.normal(1500, 300, n_samples).clip(500, 2500)
    bottled_water = np.random.binomial(1, 0.4, n_samples)
    Seafood = np.abs(np.random.normal(100, 50, n_samples)).clip(0, 300)
    Plastic_packaging = np.random.randint(1, 11, n_samples)
    Residence = np.random.choice(['urban', 'rural'], n_samples, p=[0.7, 0.3])
    Awareness = np.random.choice(['low', 'medium', 'high'], n_samples, p=[0.3, 0.5, 0.2])
    Health_Symptoms = np.random.binomial(1, 0.25, n_samples)
    
    # Calculate microplastic exposure
    mp_water = bottled_water * (Water_Intake/1000 * 3.57) + (1-bottled_water) * (Water_Intake/1000 * 9.24)
    mp_seafood = Seafood * 0.1
    mp_packaging = Plastic_packaging * 2.0
    mp_urban = (Residence == 'urban') * 1.2
    mp_awareness = np.where(Awareness == 'low', 1.0, np.where(Awareness == 'medium', 0.8, 0.6))
    mp_score = (mp_water + mp_seafood + mp_packaging) * mp_urban * mp_awareness
    
    # Define risk levels
    bins = [0, 20, 40, 70]
    labels = [0, 1, 2]  # 0: Low, 1: Medium, 2: High
    risk_level = pd.cut(mp_score, bins=bins, labels=labels, include_lowest=True)
    
    # Create DataFrame
    df = pd.DataFrame({
        'water_intake_ml': Water_Intake,
        'bottled_water': bottled_water,
        'seafood_g_per_week': Seafood,
        'plastic_packaging': Plastic_packaging,
        'residence': Residence,
        'awareness': Awareness,
        'health_symptoms': Health_Symptoms,
        'mp_score': mp_score,
        'risk_level': risk_level
    })
    
    # Encode categorical variables
    le = LabelEncoder()
    df['residence'] = le.fit_transform(df['residence'])
    df['awareness'] = le.fit_transform(df['awareness'])
    
    return df

if __name__ == "__main__":
    df = gen_data()
    X = df.drop(['risk_level', 'mp_score'], axis=1)
    y = df['risk_level'].astype(int)
    
    # Export data
    X.to_csv('features.csv', index=False)
    y.to_csv('target.csv', index=False)
    df.to_csv('full_data.csv', index=False)
