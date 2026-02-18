import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

def prepare_features(df, date_col):
    """Extracts seasonal features from the date column."""
    df[date_col] = pd.to_datetime(df[date_col])
    df["Month"] = df[date_col].dt.month
    df["DayOfWeek"] = df[date_col].dt.dayofweek
    return df

def train_model(df, price_col, demand_col, date_col):
    """Trains a Random Forest and returns the model plus the test accuracy score."""
    df = prepare_features(df, date_col)

    X = df[[price_col, "Month", "DayOfWeek"]]
    y = df[demand_col]

    # Split data to validate performance on unseen data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Calculate R2 score on the test set
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)

    return model, r2

def calculate_elasticity(df, price_col, demand_col):
    """Calculates price elasticity of demand."""
    df = df.sort_values(price_col)
    pct_change_demand = df[demand_col].pct_change().mean()
    pct_change_price = df[price_col].pct_change().mean()

    if pct_change_price != 0:
        elasticity = pct_change_demand / pct_change_price
    else:
        elasticity = 0

    return elasticity

def simulate_price(model, price, month, dayofweek, price_col):
    """Predicts units and revenue for a specific price point."""
    input_data = pd.DataFrame({
        price_col: [price],
        "Month": [month],
        "DayOfWeek": [dayofweek]
    })

    predicted_units = model.predict(input_data)[0]
    revenue = price * predicted_units

    return predicted_units, revenue

def find_optimal_price(model, price_range, month, dayofweek, price_col):
    """Finds the price that maximizes total revenue."""
    revenues = []

    for p in price_range:
        units, rev = simulate_price(model, p, month, dayofweek, price_col)
        revenues.append(rev)

    optimal_index = np.argmax(revenues)

    return price_range[optimal_index], revenues[optimal_index]