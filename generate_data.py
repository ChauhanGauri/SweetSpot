import pandas as pd
import numpy as np

np.random.seed(42)

dates = pd.date_range(start="2022-01-01", end="2023-12-31")

base_price = 100
seasonality = 20 * np.sin(2 * np.pi * dates.dayofyear / 365)
trend = np.linspace(0, 10, len(dates))

price = base_price + seasonality + trend + np.random.normal(0, 5, len(dates))

# Demand decreases when price increases (negative relationship)
units_sold = 500 - 3 * price + 0.5 * seasonality + np.random.normal(0, 20, len(dates))

df = pd.DataFrame({
    "Date": dates,
    "Price": price,
    "Units_Sold": units_sold
})

df.to_csv("pricing_data.csv", index=False)

print("pricing_data.csv generated successfully!")
