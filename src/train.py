from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib
from preprocess import preprocess_data

def train_models():
    X_train, X_test, y_train, y_test = preprocess_data()
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    joblib.dump(lr, 'models/linear_regression.pkl')
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, 'models/random_forest.pkl')
    
    # XGBoost
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    xgb.fit(X_train, y_train)
    joblib.dump(xgb, 'models/xgboost.pkl')
    
    print("Models trained and saved successfully.")

if __name__ == '__main__':
    train_models()
