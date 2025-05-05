import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer

def train_models():
    try:
        # Load and prepare data
        df = pd.read_csv('hcvdat0.csv')
        if 'Unnamed: 0' in df.columns:
            del df['Unnamed: 0']
        df = df[['Age','Sex','ALB','ALP','ALT','AST','BIL','CHE','CHOL','CREA','GGT','PROT','Category']]
        
        # Encode categorical variables
        df['Sex'] = df['Sex'].map({'m': 0, 'f': 1})
        df['Category'] = pd.Categorical(df['Category']).codes
        
        # Split features and target
        X = df.drop('Category', axis=1)
        y = df['Category']
        
        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train models
        rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
        ada_model = AdaBoostClassifier(random_state=42)
        nb_model = GaussianNB()
        
        rf_model.fit(X_scaled, y)
        ada_model.fit(X_scaled, y)
        nb_model.fit(X_scaled, y)
        
        return {
            'random_forest': rf_model,
            'adaboost': ada_model,
            'naive_bayes': nb_model,
            'scaler': scaler,
            'imputer': imputer
        }
    except Exception as e:
        print(f"Error in train_models: {str(e)}")
        raise

def doctor_diagnosis(prediction):
    diagnoses = {
        0: "Blood Donor - The patient is healthy and can donate blood.",
        1: "Suspect Blood Donor - The patient needs further medical testing.",
        2: "Hepatitis - The patient shows signs of liver inflammation and requires treatment.",
        3: "Fibrosis - The patient has liver scarring; ongoing medical care is needed.",
        4: "Cirrhosis - The patient has advanced liver disease; urgent treatment is required."
    }
    return diagnoses.get(prediction, "Diagnosis Unknown - Please verify the input.")

def predict_disease(features, models):
    try:
        # Handle missing values
        features_imputed = models['imputer'].transform(features)
        
        # Scale features
        features_scaled = models['scaler'].transform(features_imputed)
        
        predictions = {}
        for name, model in models.items():
            if name not in ['scaler', 'imputer']:
                pred = model.predict(features_scaled)[0]
                predictions[name] = {
                    'prediction': int(pred),
                    'diagnosis': doctor_diagnosis(pred)
                }
        return predictions
    except Exception as e:
        print(f"Error in predict_disease: {str(e)}")
        raise
