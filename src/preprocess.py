import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib

def preprocess_data(file_path='../data/ames_housing.csv'):
    # Load data
    data = pd.read_csv(file_path)
    
    # Separate features and target
    X = data.drop(columns=['SalePrice'])
    y = data['SalePrice']
    
    # Define feature types
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Transform data
    X_preprocessed = preprocessor.fit_transform(X)
    
    # Save preprocessor
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    preprocess_data()
