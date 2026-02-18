import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import tempfile
import plotly.io as pio
from PIL import Image as PILImage
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle

# Import your custom logic from model.py
from model import train_model, calculate_elasticity, simulate_price, find_optimal_price

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="The SweetSpot | Pricing Intelligence",
    page_icon="🎯",
    layout="wide"
)

# --- 2. Custom UI Styling ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .stButton>button { width: 100%; border-radius: 5px; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. Helper Functions ---
def generate_report_pdf(model_r2, elasticity, opt_price, opt_value, y_label, fig, business_objective):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("<b>Executive Pricing Optimization Report</b>", styles['Title']))
    elements.append(Spacer(1, 18))

    data = [
        ["Model Confidence (R²)", f"{model_r2:.3f}"],
        ["Price Elasticity", f"{elasticity:.2f}"],
        ["Recommended Price", f"₹{opt_price:.2f}"],
        [f"Max {y_label.split('(')[0]}", f"₹{opt_value:.2f}" if '₹' in y_label else f"{opt_value:.0f} units"]
    ]
    table = Table(data, hAlign='LEFT', colWidths=[200, 200])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10)
    ]))
    elements.append(table)
    elements.append(Spacer(1, 24))

    # Export Plotly chart to PNG for PDF
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        try:
            pio.write_image(fig, tmpfile.name, format="png", width=600, height=350, scale=2)
            elements.append(Image(tmpfile.name, width=450, height=260))
        except Exception as e:
            elements.append(Paragraph(f"Chart error: {e}", styles['Normal']))

    summary = f"<br/><br/>This report confirms the optimal price point for <b>{business_objective}</b>. Based on market simulations, a price of ₹{opt_price:.2f} is expected to yield the highest return."
    elements.append(Paragraph(summary, styles['BodyText']))

    doc.build(elements)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf

def profit_for_price(model, price, month, dayofweek, price_col, cost):
    units, _ = simulate_price(model, price, month, dayofweek, price_col)
    return units, (price - cost) * units

def volume_for_price(model, price, month, dayofweek, price_col):
    units, _ = simulate_price(model, price, month, dayofweek, price_col)
    return units

def find_optimal(model, price_range, month, dayofweek, price_col, cost, objective):
    results = []
    for p in price_range:
        if objective == "Revenue Maximization":
            _, val = simulate_price(model, p, month, dayofweek, price_col)
        elif objective == "Profit Maximization":
            _, val = profit_for_price(model, p, month, dayofweek, price_col, cost)
        else:
            val = volume_for_price(model, p, month, dayofweek, price_col)
        results.append(val)
    idx = np.argmax(results)
    return price_range[idx], results[idx]

# --- 4. Initialization ---
if 'step' not in st.session_state: st.session_state.step = 1
if 'df' not in st.session_state: st.session_state.df = None

# --- 5. Sidebar Navigation ---
with st.sidebar:
    st.title("🎯 The SweetSpot")
    if st.button("🔄 Start New Analysis"):
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.rerun()
    st.divider()
    if st.session_state.df is not None:
        st.info(f"Dataset: {len(st.session_state.df)} records loaded.")

# --- 6. Main Workflow ---

# STEP 1: UPLOAD
if st.session_state.step == 1:
    st.header("1️⃣ Data Intake")
    uploaded_file = st.file_uploader("Upload Sales CSV", type="csv")
    if uploaded_file:
        st.session_state.df = pd.read_csv(uploaded_file)
        if st.button("Continue to Mapping"):
            st.session_state.step = 2
            st.rerun()

# STEP 2: MAPPING
elif st.session_state.step == 2:
    st.header("2️⃣ Feature Selection")
    df = st.session_state.df
    c1, c2, c3 = st.columns(3)
    with c1: d_col = st.selectbox("Date Column", df.columns)
    with c2: p_col = st.selectbox("Price Column", df.columns)
    with c3: q_col = st.selectbox("Demand/Quantity Column", df.columns)
    
    if st.button("Train AI Model"):
        st.session_state.update({"date_col": d_col, "price_col": p_col, "demand_col": q_col, "step": 3})
        st.session_state.df[d_col] = pd.to_datetime(st.session_state.df[d_col])
        st.rerun()

# STEP 3: TRAINING
elif st.session_state.step == 3:
    st.header("3️⃣ Intelligence Training")
    with st.spinner("Analyzing demand patterns..."):
        try:
            model, r2 = train_model(st.session_state.df, st.session_state.price_col, 
                                   st.session_state.demand_col, st.session_state.date_col)
            elasticity = calculate_elasticity(st.session_state.df, st.session_state.price_col, st.session_state.demand_col)
            st.session_state.update({"model": model, "test_r2": r2, "elasticity": elasticity, "step": 4})
            st.rerun()
        except Exception as e:
            st.error(f"Training Error: {e}")
            if st.button("Back"): st.session_state.step = 2

# STEP 4: DASHBOARD
elif st.session_state.step == 4:
    st.header("📈 Optimization Dashboard")
    
    # Sidebar Dashboard Controls
    cost = st.sidebar.number_input("Cost per Unit (₹)", value=0.0)
    objective = st.sidebar.radio("Optimization Goal", ["Revenue Maximization", "Profit Maximization", "Volume Growth"])
    
    # Data & Model Setup
    df = st.session_state.df
    model = st.session_state.model
    p_col = st.session_state.price_col
    
    # Logic for calculations
    last_date = df[st.session_state.date_col].iloc[-1]
    p_min, p_max = float(df[p_col].min()), float(df[p_col].max())
    p_range = np.linspace(p_min, p_max, 100)
    
    opt_price, opt_val = find_optimal(model, p_range, last_date.month, last_date.dayofweek, p_col, cost, objective)

    # Metrics Display
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
        <div style='color:white;font-size:1.2em;font-weight:bold;'>Recommended Price</div>
        <div style='color:white;font-size:2em;font-weight:bold;'>₹{opt_price:.2f}</div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div style='color:white;font-size:1.2em;font-weight:bold;'>Price Elasticity</div>
        <div style='color:white;font-size:2em;font-weight:bold;'>{st.session_state.elasticity:.2f}</div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div style='color:white;font-size:1.2em;font-weight:bold;'>Model Confidence</div>
        <div style='color:white;font-size:2em;font-weight:bold;'>{st.session_state.test_r2:.1%}</div>
        """, unsafe_allow_html=True)

    # Chart Generation
    if objective == "Revenue Maximization":
        y_vals = [simulate_price(model, p, last_date.month, last_date.dayofweek, p_col)[1] for p in p_range]
        y_lab = "Projected Revenue (₹)"
    elif objective == "Profit Maximization":
        y_vals = [profit_for_price(model, p, last_date.month, last_date.dayofweek, p_col, cost)[1] for p in p_range]
        y_lab = "Projected Profit (₹)"
    else:
        y_vals = [volume_for_price(model, p, last_date.month, last_date.dayofweek, p_col) for p in p_range]
        y_lab = "Projected Volume (Units)"

    fig = px.line(x=p_range, y=y_vals, labels={'x': 'Price (₹)', 'y': y_lab}, title=f"{objective} Curve")
    fig.add_vline(x=opt_price, line_dash="dash", line_color="green")
    st.plotly_chart(fig, use_container_width=True)

    # PDF Download
    if st.button("Generate Executive PDF"):
        pdf = generate_report_pdf(st.session_state.test_r2, st.session_state.elasticity, opt_price, opt_val, y_lab, fig, objective)
        st.download_button("Download Report", pdf, "SweetSpot_Report.pdf", "application/pdf")

st.caption("Built with Python • Streamlit • Scikit-Learn")