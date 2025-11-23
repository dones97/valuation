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


def get_model_statistics(model):
    """Get Random Forest model statistics"""
    if model is None or model.model is None:
        return None

    # Get tree depths
    depths = [tree.tree_.max_depth for tree in model.model.estimators_]
    avg_depth = np.mean(depths)
    max_depth_configured = model.model.max_depth

    # Get average number of samples in leaf nodes
    leaf_samples = []
    for tree in model.model.estimators_:
        tree_structure = tree.tree_
        # Leaf nodes have -1 for both children
        is_leaf = (tree_structure.children_left == -1) & (tree_structure.children_right == -1)
        leaf_samples.extend(tree_structure.n_node_samples[is_leaf])

    avg_leaf_samples = np.mean(leaf_samples)

    return {
        'max_depth': max_depth_configured,
        'avg_depth': avg_depth,
        'avg_leaf_samples': avg_leaf_samples,
        'n_trees': model.model.n_estimators
    }


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


def get_key_contributors(prediction):
    """Identify key factors contributing to the P/E prediction"""
    contributors = []

    # Check ROE
    if 'roe' in prediction and not pd.isna(prediction['roe']):
        if prediction['roe'] > 0.20:  # 20%+
            contributors.append("high ROE")
        elif prediction['roe'] < 0.10:  # <10%
            contributors.append("low ROE")

    # Check profit margins
    if 'profit_margins' in prediction and not pd.isna(prediction['profit_margins']):
        if prediction['profit_margins'] > 0.15:  # 15%+
            contributors.append("strong profit margins")
        elif prediction['profit_margins'] < 0.05:  # <5%
            contributors.append("weak profit margins")

    # Check revenue growth
    if 'revenue_growth' in prediction and not pd.isna(prediction['revenue_growth']):
        if prediction['revenue_growth'] > 0.20:  # 20%+
            contributors.append("high revenue growth")
        elif prediction['revenue_growth'] < 0:
            contributors.append("declining revenues")

    # Check ROCE
    if 'roce' in prediction and not pd.isna(prediction['roce']):
        if prediction['roce'] > 0.20:  # 20%+
            contributors.append("excellent capital efficiency")

    # Check debt levels
    if 'debt_to_equity' in prediction and not pd.isna(prediction['debt_to_equity']):
        if prediction['debt_to_equity'] < 0.5:
            contributors.append("low debt levels")
        elif prediction['debt_to_equity'] > 2.0:
            contributors.append("high debt burden")

    # Return top 3 contributors
    if len(contributors) >= 2:
        if len(contributors) == 2:
            return f"{contributors[0]} and {contributors[1]}"
        else:
            return f"{contributors[0]}, {contributors[1]}, and {contributors[2]}"
    elif len(contributors) == 1:
        return contributors[0]
    else:
        return "the company's overall fundamental profile"


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
            value=f"{prediction['predicted_pe']:.2f}" if not pd.isna(prediction['predicted_pe']) else "N/A"
        )

    with col4:
        st.metric(
            label="Fair Price",
            value=f"₹{prediction['fair_price']:.2f}" if not pd.isna(prediction['fair_price']) else "N/A"
        )

    # Main Takeaway Section
    st.markdown("---")
    st.markdown("### 🎯 Key Takeaway")

    if not pd.isna(prediction['upside_downside_pct']):
        # Create two columns: left for text, right for gauge
        takeaway_col1, takeaway_col2 = st.columns([1.2, 1])

        with takeaway_col1:
            upside = prediction['upside_downside_pct']

            # Large prominent text for upside/downside
            if upside > 0:
                st.markdown(f"""
                <div style='padding: 2rem; background-color: #d4edda; border-radius: 10px; border-left: 5px solid #28a745;'>
                    <h1 style='color: #155724; margin: 0; font-size: 3rem;'>+{upside:.1f}%</h1>
                    <h3 style='color: #155724; margin-top: 0.5rem;'>Potential Upside</h3>
                    <p style='color: #155724; font-size: 1.1rem; margin-top: 1rem;'>
                        The stock appears <strong>undervalued</strong> based on the predicted fair P/E ratio.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='padding: 2rem; background-color: #f8d7da; border-radius: 10px; border-left: 5px solid #dc3545;'>
                    <h1 style='color: #721c24; margin: 0; font-size: 3rem;'>{upside:.1f}%</h1>
                    <h3 style='color: #721c24; margin-top: 0.5rem;'>Potential Downside</h3>
                    <p style='color: #721c24; font-size: 1.1rem; margin-top: 1rem;'>
                        The stock appears <strong>overvalued</strong> based on the predicted fair P/E ratio.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Add key contributors
            st.markdown("")
            key_factors = get_key_contributors(prediction)
            st.markdown(f"**Key Contributors:** The prediction is primarily driven by {key_factors}.")

        with takeaway_col2:
            # Valuation gauge (without delta indicator)
            if not pd.isna(prediction['current_pe']) and not pd.isna(prediction['predicted_pe']):
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prediction['current_pe'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Current P/E vs Fair P/E", 'font': {'size': 18}},
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

                fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)
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

        # Model architecture details
        st.markdown("---")
        st.markdown("### 🌳 Random Forest Architecture")

        model_stats = get_model_statistics(model)

        if model_stats:
            arch_col1, arch_col2, arch_col3, arch_col4 = st.columns(4)

            with arch_col1:
                st.metric(
                    label="Number of Trees",
                    value=f"{model_stats['n_trees']}"
                )

            with arch_col2:
                st.metric(
                    label="Max Tree Depth",
                    value=f"{model_stats['max_depth']}"
                )

            with arch_col3:
                st.metric(
                    label="Avg Actual Depth",
                    value=f"{model_stats['avg_depth']:.1f}"
                )

            with arch_col4:
                st.metric(
                    label="Avg Stocks per Leaf",
                    value=f"{model_stats['avg_leaf_samples']:.1f}"
                )

            st.markdown("""
            The Random Forest uses **100 decision trees**, each with a maximum depth of **15 levels**.
            On average, the final leaf nodes (decision endpoints) contain approximately **{:.1f} stocks**,
            which prevents overfitting by ensuring predictions are based on multiple similar stocks rather
            than individual outliers.
            """.format(model_stats['avg_leaf_samples']))

        # Feature importance visualization
        st.markdown("---")
        st.markdown("### 📊 Top 10 Features Explaining P/E Ratio")

        try:
            from PIL import Image
            feature_img = Image.open('model_visualizations/feature_importance.png')
            st.image(feature_img, use_container_width=True)
            st.caption("Feature importance shows which fundamental metrics have the strongest influence on P/E ratio predictions. Higher values indicate greater predictive power.")
        except FileNotFoundError:
            st.warning("Feature importance visualization not found. Please train the model first.")

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
