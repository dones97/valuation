"""
Streamlit App for P/E Ratio Prediction and Stock Valuation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pe_prediction_model import PEPredictionModel

# Page configuration
st.set_page_config(
    page_title="Indian Stock P/E Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .positive {
        color: #28a745;
        font-weight: bold;
    }
    .negative {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained P/E prediction model"""
    try:
        model = PEPredictionModel('indian_stocks_tickers.csv')
        model.load_model('pe_prediction_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model not found! Please train the model first by running 'pe_prediction_model.py'")
        return None


@st.cache_data
def load_tickers():
    """Load available stock tickers (NSE only)"""
    try:
        df = pd.read_csv('indian_stocks_tickers.csv')
        # Filter to only NSE stocks
        df = df[df['Exchange'] == 'NSE']
        return df
    except FileNotFoundError:
        st.error("Ticker file not found! Please ensure 'indian_stocks_tickers.csv' exists.")
        return pd.DataFrame()


def display_prediction_results(prediction):
    """Display prediction results in a formatted way"""
    if not prediction:
        st.error("Unable to make prediction for this stock. Data might be unavailable.")
        return

    st.markdown(f"### 📊 Valuation Analysis: {prediction['company_name']}")

    # Metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Current Price",
            value=f"₹{prediction['current_price']:.2f}" if not pd.isna(prediction['current_price']) else "N/A"
        )

    with col2:
        st.metric(
            label="Current P/E",
            value=f"{prediction['current_pe']:.2f}" if not pd.isna(prediction['current_pe']) else "N/A"
        )

    with col3:
        st.metric(
            label="Predicted Fair P/E",
            value=f"{prediction['predicted_pe']:.2f}" if not pd.isna(prediction['predicted_pe']) else "N/A",
            delta=f"{prediction['predicted_pe'] - prediction['current_pe']:.2f}" if not pd.isna(prediction['predicted_pe']) and not pd.isna(prediction['current_pe']) else None
        )

    with col4:
        st.metric(
            label="Fair Price",
            value=f"₹{prediction['fair_price']:.2f}" if not pd.isna(prediction['fair_price']) else "N/A"
        )

    # Upside/Downside
    st.markdown("---")
    upside_col1, upside_col2, upside_col3 = st.columns([1, 2, 1])

    with upside_col2:
        if not pd.isna(prediction['upside_downside_pct']):
            upside = prediction['upside_downside_pct']

            if upside > 0:
                st.markdown(f"**Potential Upside: +{upside:.2f}%**")
                st.markdown("The stock appears **undervalued** based on predicted fair P/E ratio.")
            else:
                st.markdown(f"**Potential Downside: {upside:.2f}%**")
                st.markdown("The stock appears **overvalued** based on predicted fair P/E ratio.")
        else:
            st.warning("Unable to calculate upside/downside. EPS data unavailable.")

    # Additional details
    st.markdown("---")
    st.markdown("### 📌 Stock Details")

    detail_col1, detail_col2 = st.columns(2)

    with detail_col1:
        st.markdown(f"**Ticker:** {prediction['ticker']}")
        st.markdown(f"**Sector:** {prediction['sector']}")
        st.markdown(f"**Industry:** {prediction['industry']}")

    with detail_col2:
        st.markdown(f"**EPS (Trailing 12M):** ₹{prediction['eps']:.2f}" if not pd.isna(prediction['eps']) else "**EPS:** N/A")
        st.markdown(f"**P/E Deviation:** {((prediction['current_pe'] - prediction['predicted_pe']) / prediction['predicted_pe'] * 100):.2f}%" if not pd.isna(prediction['current_pe']) and not pd.isna(prediction['predicted_pe']) else "**P/E Deviation:** N/A")

    # Gauge chart for valuation
    st.markdown("---")
    st.markdown("### 🎯 Valuation Meter")

    if not pd.isna(prediction['current_pe']) and not pd.isna(prediction['predicted_pe']):
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prediction['current_pe'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Current P/E vs Fair P/E", 'font': {'size': 24}},
            delta={'reference': prediction['predicted_pe'], 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge={
                'axis': {'range': [None, max(prediction['current_pe'], prediction['predicted_pe']) * 1.5]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, prediction['predicted_pe'] * 0.8], 'color': "lightgreen"},
                    {'range': [prediction['predicted_pe'] * 0.8, prediction['predicted_pe'] * 1.2], 'color': "lightyellow"},
                    {'range': [prediction['predicted_pe'] * 1.2, max(prediction['current_pe'], prediction['predicted_pe']) * 1.5], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': prediction['predicted_pe']
                }
            }
        ))

        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)


def main():
    # Header
    st.markdown("<h1 class='main-header'>Indian Stock P/E Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>AI-Powered Fair Valuation Analysis using Random Forest</p>", unsafe_allow_html=True)

    # Load model and data
    model = load_model()
    tickers_df = load_tickers()

    if model is None or tickers_df.empty:
        st.stop()

    # Sidebar
    st.sidebar.title("Stock Selection")

    # Stock ticker input
    st.sidebar.info("💡 Enter ticker with .NS suffix (e.g., RELIANCE.NS)")

    selected_ticker = st.sidebar.text_input(
        "Enter Stock Ticker:",
        placeholder="e.g., RELIANCE.NS",
        help="Enter the stock ticker symbol with .NS suffix for NSE stocks"
    )

    # Optional: Show dropdown of available stocks for reference
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Or select from list:**")

    ticker_options = tickers_df['Ticker'].tolist()
    company_names = tickers_df['Company Name'].tolist()
    display_options = [f"{ticker} - {name}" for ticker, name in zip(ticker_options, company_names)]

    selected_from_list = st.sidebar.selectbox(
        "Available NSE stocks:",
        options=[""] + display_options,
        index=0,
        help="Select from pre-loaded NSE stocks"
    )

    # If user selected from dropdown, override the text input
    if selected_from_list:
        selected_ticker = selected_from_list.split(' - ')[0]

    # Predict button
    st.sidebar.markdown("---")
    predict_button = st.sidebar.button("🔮 Predict Fair P/E", type="primary", use_container_width=True)

    # Main content
    if predict_button:
        if not selected_ticker:
            st.warning("Please enter a stock ticker or select from the list.")
        else:
            with st.spinner(f"Analyzing {selected_ticker}..."):
                # Get prediction
                prediction = model.predict_pe(selected_ticker)

                if prediction:
                    # Display results
                    display_prediction_results(prediction)

                else:
                    st.error(f"Unable to analyze {selected_ticker}. The stock might not have sufficient data.")

    else:
        # Welcome message
        st.info("👈 Enter a stock ticker or select from the list, then click 'Predict Fair P/E' to get started!")

        # Show some statistics about the model
        st.markdown("### About This Tool")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🎯 Features Used**")
            st.markdown("""
            - Financial Ratios (ROE, ROCE, ROA)
            - Profitability Metrics
            - Growth Rates
            - Debt Metrics
            - Market Cap & Sector
            """)

        with col2:
            st.markdown("**🤖 Model Details**")
            st.markdown("""
            - Algorithm: Random Forest
            - Training Split: 80/20
            - Features: 27 fundamental metrics
            - Auto-updates predictions
            """)

        with col3:
            st.markdown("**📊 How It Works**")
            st.markdown("""
            1. Fetches fundamental data
            2. Predicts fair P/E ratio
            3. Calculates fair price (P/E × EPS)
            4. Shows upside/downside %
            """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; padding: 1rem;'>
        <p>This tool is for educational purposes only. Always do your own research before investing.</p>
        <p>Data powered by Yahoo Finance</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
