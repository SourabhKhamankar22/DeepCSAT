# 🛍️ DeepCSAT: E-Commerce Customer Satisfaction Prediction
**🔗 [View the Live Interactive Dashboard Here](https://your-streamlit-app-link-here.streamlit.app/)**

## 📌 Overview
Customer satisfaction is the lifeblood of e-commerce. Traditionally, platforms rely on post-interaction surveys to measure Customer Satisfaction (CSAT), which are highly reactive and suffer from low response rates. DeepCSAT engineers an end-to-end Machine Learning and Deep Learning pipeline to transition from a reactive survey model to a proactive, real-time predictive model.

Through an interactive Streamlit web dashboard, the system ingests operational tabular data (handling time, shift, channel) and unstructured natural language customer remarks to predict CSAT scores instantly. This allows the business to automatically flag frustrated customers and deploy targeted "save" protocols to prevent revenue churn.

## 🚀 Features
- **Predictive Analytics (Deep Learning):** Utilizes a deeply engineered Artificial Neural Network (ANN) to predict a customer's CSAT score (1-5) based on their interaction metadata and feedback.

- **Advanced NLP Pipeline:** Cleans and transforms thousands of unstructured text reviews into sparse mathematical matrices using TF-IDF Vectorization.

- **Smart Feature Assembly:** Deliberately bypasses traditional PCA (Dimensionality Reduction) and synthetic oversampling (SMOTE) to preserve the inherent sparsity of the text data, preventing information loss.

- **Interactive Streamlit Dashboard:** * Real-Time Prediction Engine: Allows agents to paste customer remarks and get instant CSAT predictions with dynamic churn-risk warnings.

    - **Historical EDA Dashboard:** Provides management with interactive visualizations of historical ticket distributions, handling time correlations, and channel performance.


## 🛠️ Tech Stack
- **Language:** Python
- **Deep Learning:** TensorFlow / Keras
- **Machine Learning:** XGBoost, scikit-learn, RandomForest
- **Natural Language Processing (NLP):** NLTK, TF-IDF
- **Data Manipulation & Stats:** pandas, NumPy, SciPy (ANOVA, Chi-Square)
- **Web Framework:** Streamlit
- **Data Visualization:** Matplotlib, Seaborn

## 📊 Key Results & Insights
- **The "Sparsity" Breakthrough:** Traditional tree-based models (Random Forest, XGBoost) and strict class weights hit a performance ceiling around 40%. By utilizing a wide-input ANN (512 -> 256 -> 128 neurons) and feeding it uncompressed TF-IDF vectors alongside scaled tabular data, the model shattered the ceiling to achieve 72.25% Overall Accuracy.
- **High Precision on Churn Risk:** The model achieved 67% Precision on Class 0 (extreme dissatisfaction). This high precision ensures that the business does not waste expensive retention budgets (discounts, refunds) on customers who are actually satisfied.
- **Feature Importance:** Explainability analysis proved the model utilizes a logical hybrid of data: tabular constraints (like `Item_price` and `handling_time`) dictate the baseline expectation, while specific TF-IDF text tokens (like "bad", "worst", "good") act as the final triggers for the prediction.

## 🌐 Deployment
This project is deployed using Streamlit Community Cloud for seamless, real-time web access.
- The Deep Learning ANN (`.keras` format) and NLP preprocessors (`.pkl` formats) are cached in the app's memory to ensure instant inference without reloading the heavy neural network on every user click.
- The application handles dynamic shape-matching to ensure raw user text input perfectly aligns with the 1,000-feature TF-IDF matrix expected by the neural network.

## 📂 Project Structure
- `ML_Submission_Template.ipynb` - The core Jupyter Notebook containing EDA, hypothesis testing, NLP preprocessing, model training, evaluation, and business insights.
- `app.py` - The Streamlit application script containing the prediction UI and interactive EDA dashboard.
- `best_deepcsat_ann_model.keras` - The serialized, trained Artificial Neural Network.
- `tfidf_vectorizer.pkl` & `tabular_scaler.pkl` - Serialized preprocessors needed for real-time inference.
- `eCommerce_Customer_support_data.csv` - The historical dataset used for the EDA dashboard.
- `requirements.txt` - Python dependencies required to run the application.

## ⚙️ Installation and Setup
1. Clone the repository:
   ```
   git clone https://github.com/SourabhKhamankar22/DeepCSAT
   ```
2. Set up a virtual environment:
    ```
    conda create --name deepcsat_env python=3.10
    conda activate deepcsat_env
    ```
3. Install dependencies:
    ```
    pip install -r requirements.txt
    ```
4. Run the Streamlit application:
    ```
    streamlit run app.py
    ```