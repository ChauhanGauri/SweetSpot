# The SweetSpot: Dynamic Pricing Intelligence

## Overview
The SweetSpot is a Streamlit-powered platform for data-driven pricing optimization. It uses Random Forest Regression to model price elasticity, simulate market response, and recommend optimal pricing strategies for maximum revenue, profit, or sales volume.

## Features
- Step-based workflow for easy data upload, mapping, model training, and optimization
- Supports multiple products via Product_ID column
- Interactive dashboard for price simulation and optimization
- Business objectives: Revenue Maximization, Profit Maximization, Volume Growth
- Executive PDF report generation
- Data health dashboard (missing values, outliers, correlation)
- Customizable cost per unit and optimization settings

## Getting Started
1. **Clone the repository**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install streamlit reportlab kaleido plotly scikit-learn pandas numpy
   ```
3. **Run the app**:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Upload your pricing dataset (CSV)
2. Map columns (Date, Price, Demand, Product_ID)
3. Train the model and view metrics (Price Elasticity, Model Confidence)
4. Simulate pricing scenarios and optimize for your business goal
5. Download executive reports as PDF

## Data Format
- CSV file with columns: Date, Price, Units_Sold, (optional) Product_ID

## Project Structure
- `app.py` — Streamlit app
- `model.py` — Model logic (training, simulation, optimization)
- `generate_data.py` — Synthetic data generator
- `pricing_data.csv` — Sample dataset
- `requirements.txt` — Python dependencies

## License
MIT License

## Authors
- Your Name

---
*Finding the intersection of value and volume.*