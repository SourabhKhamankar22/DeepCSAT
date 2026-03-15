import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Configuration ---
st.set_page_config(page_title="DeepCSAT Predictor", page_icon="🛍️", layout="wide")

# --- 1. Load Models, Preprocessors & Data ---
@st.cache_resource 
def load_assets():
    model = load_model('best_deepcsat_ann_model.keras')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    return model, tfidf

@st.cache_data
def load_data():
    # Replace 'your_dataset_name.csv' with the actual name of your original CSV file!
    try:
        df = pd.read_csv('eCommerce_Customer_support_data.csv')
        return df
    except FileNotFoundError:
        return None

model, tfidf = load_assets()
df = load_data()

# --- App Header ---
st.title("🛍️ DeepCSAT: E-Commerce Sentiment Predictor & Dashboard")
st.markdown("Predict Customer Satisfaction (CSAT) scores in real-time and analyze historical trends.")

# --- Create Tabs ---
tab1, tab2 = st.tabs(["🚀 CSAT Prediction", "📊 Data Insights (EDA)"])

# ==========================================
# TAB 1: PREDICTION LOGIC
# ==========================================
with tab1:
    st.header("Enter Customer Feedback")
    
    # Text box for remarks
    customer_remarks = st.text_area("Customer Remarks", "The agent was very helpful, but the delivery was incredibly delayed and I am frustrated.")
    
    if st.button("Predict CSAT Score", use_container_width=True):
        if customer_remarks.strip() == "":
            st.warning("Please enter customer remarks.")
        else:
            with st.spinner('Analyzing sentiment...'):
                # 1. Vectorize the text
                text_vector = tfidf.transform([customer_remarks]).toarray()
                
                # 2. Dynamic Shape Matcher
                expected_shape = model.input_shape[1] 
                
                if text_vector.shape[1] > expected_shape:
                    final_input = text_vector[:, :expected_shape] 
                elif text_vector.shape[1] < expected_shape:
                    padding = np.zeros((1, expected_shape - text_vector.shape[1]))
                    final_input = np.hstack((text_vector, padding))
                else:
                    final_input = text_vector
                    
                # 3. Predict
                prediction_prob = model.predict(final_input)
                predicted_class = np.argmax(prediction_prob, axis=1)[0]
                
                # 4. Display Result
                csat_score = predicted_class + 1 
                
                st.subheader(f"Predicted CSAT Score: {csat_score} ⭐")
                if csat_score <= 3:
                    st.error("⚠️ High Risk of Churn! Recommend triggering retention protocol.")
                else:
                    st.success("✅ Positive Interaction. Customer is satisfied.")

# ==========================================
# TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
with tab2:
    st.header("Historical Customer Satisfaction Insights")
    
    if df is not None:
        col1, col2 = st.columns(2)
        
        # --- Chart 1: Distribution of CSAT Scores ---
        with col1:
            st.subheader("1. Distribution of CSAT Scores")
            st.markdown("Visualizing the class imbalance in our historical data.")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            sns.countplot(x='CSAT Score', data=df, palette='viridis', ax=ax1)
            ax1.set_xlabel("CSAT Score (1-5)")
            ax1.set_ylabel("Number of Tickets")
            st.pyplot(fig1)

        # --- Chart 2: CSAT by Channel ---
        with col2:
            if 'channel_name' in df.columns:
                st.subheader("2. Average CSAT by Channel")
                st.markdown("Which communication channels result in the happiest customers?")
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                sns.barplot(x='channel_name', y='CSAT Score', data=df, palette='mako', ax=ax2, ci=None)
                ax2.set_xlabel("Communication Channel")
                ax2.set_ylabel("Average CSAT Score")
                st.pyplot(fig2)
            else:
                st.info("Channel column not found in dataset.")

        st.divider()

        col3, col4 = st.columns(2)
        
        # --- Chart 3: Handling Time vs CSAT ---
        with col3:
            if 'connected_handling_time' in df.columns:
                st.subheader("3. Handling Time vs. CSAT")
                st.markdown("Do longer resolution times lead to lower satisfaction?")
                fig3, ax3 = plt.subplots(figsize=(6, 4))
                # Limiting to 95th percentile to avoid massive outliers ruining the plot
                cap = df['connected_handling_time'].quantile(0.95)
                filtered_df = df[df['connected_handling_time'] < cap]
                sns.boxplot(x='CSAT Score', y='connected_handling_time', data=filtered_df, palette='coolwarm', ax=ax3)
                ax3.set_xlabel("CSAT Score")
                ax3.set_ylabel("Handling Time (seconds)")
                st.pyplot(fig3)
            else:
                st.info("Handling time column not found in dataset.")

        # --- Chart 4: Agent Shift vs CSAT ---
        with col4:
            if 'Agent Shift' in df.columns:
                st.subheader("4. CSAT by Agent Shift")
                st.markdown("Identifying if service quality drops during specific times of day.")
                fig4, ax4 = plt.subplots(figsize=(6, 4))
                sns.barplot(x='Agent Shift', y='CSAT Score', data=df, palette='rocket', ax=ax4, ci=None)
                ax4.set_xlabel("Agent Shift")
                ax4.set_ylabel("Average CSAT Score")
                plt.xticks(rotation=45)
                st.pyplot(fig4)
            else:
                st.info("Agent Shift column not found in dataset.")
                
    else:
        st.error("Dataset not found! Please ensure your CSV file is in the same directory as this app and update the 'load_data()' function with the correct file name.")