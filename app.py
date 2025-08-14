import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf

st.set_page_config(page_title="Zomato Analysis & Prediction", layout="wide")


# -------------------
# Load Data & Models
# -------------------
@st.cache_data
def load_data():
    return pd.read_csv("zomato_sample.csv")


@st.cache_resource
def load_models():
    model_ml = joblib.load("random_forest_pipeline.pkl")
    model_nn = tf.keras.models.load_model("nn_model.keras")
    return model_ml, model_nn


df = load_data()
model_ml, model_nn = load_models()

# -------------------
# Sidebar Navigation
# -------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Analysis", "Prediction"])

# -------------------
# Analysis Page
# -------------------
if page == "Analysis":
    st.title("📊 Exploratory Data Analysis")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Distribution of Ratings
    fig1, ax1 = plt.subplots()
    sns.histplot(df["rate"], bins=20, kde=True, ax=ax1)
    ax1.set_title("Distribution of Ratings")
    st.pyplot(fig1)

    # Votes vs Rating
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=df, x="votes", y="rate", ax=ax2)
    ax2.set_title("Votes vs Rating")
    st.pyplot(fig2)

    # --- Clean column names ---
    df.columns = df.columns.str.strip()

    st.write("Available columns:", df.columns.tolist())

    # --- Average Rating by City ---
    if "listed_in(city)" in df.columns:
        avg_rating = (
            df.groupby("listed_in(city)")["rate"].mean().sort_values(ascending=False)
        )
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        avg_rating.plot(kind="bar", ax=ax3)
        ax3.set_title("Average Rating by City")
        ax3.set_ylabel("Average Rate")
        ax3.set_xlabel("City")
        st.pyplot(fig3)
    else:
        st.error("'listed_in(city)' column not found in dataset.")

    # Count Plot of Locations
    fig4, ax4 = plt.subplots(figsize=(16, 10))
    sns.countplot(
        y="location", data=df, order=df["location"].value_counts().index, ax=ax4
    )
    ax4.set_title("Restaurant Count by Location")
    st.pyplot(fig4)

    # Online Order
    fig5, ax5 = plt.subplots()
    sns.countplot(x="online_order", data=df, palette="inferno", ax=ax5)
    ax5.set_title("Online Order Availability")
    st.pyplot(fig5)

    # Book Table
    fig6, ax6 = plt.subplots()
    sns.countplot(x="book_table", data=df, palette="rainbow", ax=ax6)
    ax6.set_title("Book Table Availability")
    st.pyplot(fig6)

    # Online Order vs Rate
    fig7, ax7 = plt.subplots()
    sns.boxplot(x="online_order", y="rate", data=df, ax=ax7)
    ax7.set_title("Online Order vs Rating")
    st.pyplot(fig7)

    # Book Table vs Rate
    fig8, ax8 = plt.subplots()
    sns.boxplot(x="book_table", y="rate", data=df, ax=ax8)
    ax8.set_title("Book Table vs Rating")
    st.pyplot(fig8)

    # Online Order by Location
    df_online = (
        df.groupby(["location", "online_order"])["name"].count().unstack(fill_value=0)
    )
    fig9, ax9 = plt.subplots(figsize=(15, 8))
    df_online.plot(kind="bar", ax=ax9)
    ax9.set_title("Online Order by Location")
    st.pyplot(fig9)

    # Book Table by Location
    df_book = (
        df.groupby(["location", "book_table"])["name"].count().unstack(fill_value=0)
    )
    fig10, ax10 = plt.subplots(figsize=(15, 8))
    df_book.plot(kind="bar", ax=ax10)
    ax10.set_title("Book Table by Location")
    st.pyplot(fig10)

    # Type vs Rate
    fig11, ax11 = plt.subplots(figsize=(14, 8))
    sns.boxplot(x="Type", y="rate", data=df, palette="inferno", ax=ax11)
    ax11.set_title("Restaurant Type vs Rating")
    st.pyplot(fig11)

    # Types of Restaurants by Location
    df_type = df.groupby(["location", "Type"])["name"].count().unstack(fill_value=0)
    fig12, ax12 = plt.subplots(figsize=(36, 8))
    df_type.plot(kind="bar", ax=ax12)
    ax12.set_title("Types of Restaurants by Location")
    st.pyplot(fig12)

    # Votes by Location
    votes_loc = df.groupby("location")["votes"].sum().sort_values(ascending=False)
    fig13, ax13 = plt.subplots(figsize=(15, 8))
    sns.barplot(x=votes_loc.index, y=votes_loc.values, ax=ax13)
    ax13.set_title("Total Votes by Location")
    ax13.set_xticklabels(ax13.get_xticklabels(), rotation=90)
    st.pyplot(fig13)

    # Top Cuisines
    cuisines_votes = (
        df.groupby("cuisines")["votes"].sum().sort_values(ascending=False).iloc[1:]
    )
    fig14, ax14 = plt.subplots(figsize=(15, 8))
    sns.barplot(x=cuisines_votes.index, y=cuisines_votes.values, ax=ax14)
    ax14.set_title("Top Cuisines by Total Votes")
    ax14.set_xticklabels(ax14.get_xticklabels(), rotation=90)
    st.pyplot(fig14)

# -------------------
# Prediction Page
# -------------------
elif page == "Prediction":
    st.title("🔮 Restaurant Rating Prediction")

    with st.form("prediction_form"):
        votes = st.number_input("Number of Votes", min_value=0)
        rate = st.number_input("Current Rating", min_value=0.0, max_value=5.0, step=0.1)
        city = st.text_input("City")
        submit = st.form_submit_button("Predict")

    if submit:
        input_df = pd.DataFrame(
            [[votes, rate, city]], columns=["votes", "rate", "listed_in(city)"]
        )

        ml_pred = model_ml.predict(input_df)[0]
        nn_pred = model_nn.predict(input_df)[0]

        st.success(f"ML Model Prediction: {ml_pred}")
        st.success(f"Neural Network Prediction: {nn_pred}")
