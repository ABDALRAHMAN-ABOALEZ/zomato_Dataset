import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tensorflow.keras.models import load_model

# ------------------- Page Setup -------------------
st.set_page_config(page_title="Zomato Restaurant Popularity", layout="wide")
st.title("🍽️ Zomato Restaurant Popularity Predictor")

# ------------------- Load Sampled Dataset -------------------
@st.cache_data
def load_data():
    df = pd.read_csv("zomato_sample.csv")
    return df

df = load_data()

# ------------------- Load Models -------------------
@st.cache_resource
def load_models():
    rf_model = joblib.load("random_forest_pipeline.pkl")
    nn_model = load_model("nn_model.keras")
    return rf_model, nn_model

rf_model, nn_model = load_models()

# ------------------- Sidebar Navigation -------------------
page = st.sidebar.selectbox("Choose Page", ["Analysis", "Prediction"])

# ------------------- Analysis Page -------------------
if page == "Analysis":
    st.header("📊 EDA & Key Findings")

    st.subheader("Distribution of Ratings")
    plt.figure(figsize=(10,5))
    sns.histplot(df['rate'], bins=20, kde=True)
    st.pyplot(plt)
    plt.clf()

    st.subheader("Top Restaurant Types")
    rest_counts = df['rest_type'].value_counts().head(20)
    st.bar_chart(rest_counts)

    st.subheader("Key Insights")
    st.markdown("""
    - Casual Dining and Quick Bites dominate the dataset.  
    - Restaurants accepting online orders tend to have higher ratings.  
    - Popular cuisines: North Indian, Chinese, Italian.  
    """)

# ------------------- Prediction Page -------------------
if page == "Prediction":
    st.header("🤖 Predict Restaurant Popularity")
    st.subheader("Enter Restaurant Details")

    # Input form
    online_order = st.selectbox("Online Order", ["Yes", "No"])
    book_table = st.selectbox("Book Table", ["Yes", "No"])
    votes = st.number_input("Number of Votes", min_value=0, value=50)
    location = st.text_input("Location", "Bangalore")
    rest_type = st.text_input("Restaurant Type", "Casual Dining")
    cuisines = st.text_input("Cuisines", "Italian")
    cost = st.number_input("Cost for Two", min_value=0, value=500)

    if st.button("Predict"):
        # Create dataframe
        new_restaurant = pd.DataFrame([{
            'online_order': online_order,
            'book_table': book_table,
            'votes': votes,
            'location': location,
            'rest_type': rest_type,
            'cuisines': cuisines,
            'cost': cost
        }])

        # Random Forest prediction
        rf_pred = rf_model.predict(new_restaurant)
        rf_result = "Yes" if rf_pred[0] == 1 else "No"

        # Neural Network prediction
        preprocessor = rf_model.named_steps['preprocessor']
        new_nn = preprocessor.transform(new_restaurant)
        nn_pred = nn_model.predict(new_nn)
        nn_result = "Yes" if int(nn_pred[0][0] > 0.5) else "No"

        # Display results
        st.subheader("Predictions")
        st.write(f"✅ Random Forest Prediction: **{rf_result}**")
        st.write(f"✅ Neural Network Prediction: **{nn_result}**")
