from sklearn.metrics import mean_squared_error
import joblib
import numpy as np
from preprocess import preprocess_data

def evaluate_model(model_name, X_test, y_test):
    model = joblib.load(f'models/{model_name}.pkl')
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return rmse

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = preprocess_data()
    
    # Evaluate models
    for model_name in ['linear_regression', 'random_forest', 'xgboost']:
        rmse = evaluate_model(model_name, X_test, y_test)
        print(f'{model_name} RMSE: {rmse}')
