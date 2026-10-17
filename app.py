
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from prediction import predict_churn

# PATHS AND CONFIGURATION

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Data" / "netflix_churn.csv"

st.set_page_config(
    page_title="Netflix Churn Prediction",
    page_icon="🎬",
    layout="wide"
)



# NETFLIX COLORS

BLACK = "#141414"
DARK_GREY = "#242424"
GREY = "#777777"
RED = "#E50914"
DARK_RED = "#B20710"
WHITE = "#FFFFFF"


# MATPLOTLIB DARK THEME

plt.rcParams.update({
    "figure.facecolor": BLACK,
    "axes.facecolor": BLACK,
    "axes.edgecolor": "#444444",
    "axes.labelcolor": WHITE,
    "axes.titlecolor": WHITE,
    "xtick.color": WHITE,
    "ytick.color": WHITE,
    "text.color": WHITE,
    "grid.color": "#444444",
    "font.size": 11,
    "legend.facecolor": DARK_GREY,
    "legend.edgecolor": "#444444"
})


# =========================
# STREAMLIT CUSTOM THEME
# =========================

st.markdown("""
<style>
.stApp {
    background-color: #141414;
    color: #ffffff;
}

[data-testid="stHeader"] {
    background-color: #141414;
}

h1, h2, h3 {
    color: #E50914 !important;
}

p, label, .stMarkdown {
    color: #FFFFFF;
}

div[data-testid="stMetric"] {
    background: #242424;
    padding: 18px;
    border-radius: 10px;
    border-bottom: 3px solid #E50914;
}

div[data-testid="stMetricLabel"] {
    color: #CCCCCC;
}

div[data-testid="stMetricValue"] {
    color: #FFFFFF;
}

div.stButton > button,
div[data-testid="stFormSubmitButton"] > button {
    background-color: #E50914;
    color: white;
    border-radius: 7px;
    border: none;
    font-weight: bold;
    padding: 0.55rem 1rem;
}

div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    background-color: #B20710;
    color: white;
    border: none;
}

div[data-testid="stForm"] {
    background-color: #1C1C1C;
    padding: 20px;
    border: 1px solid #333333;
    border-radius: 12px;
}

hr {
    border-color: #333333;
}

div[data-testid="stAlert"] {
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)



# LOAD DATA

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data.columns = data.columns.str.strip()
    return data



st.markdown("""
<div style="text-align: center; padding: 25px 0 10px 0;">
    <h1 style="
        color: #E50914;
        font-size: 64px;
        font-weight: 900;
        letter-spacing: 5px;
        margin: 0;
        text-align: center;
        font-family: Arial, Helvetica, sans-serif;
    ">
        NETFLIX
    </h1>
    <h3 style="
        color: white;
        font-size: 25px;
        font-weight: 500;
        margin-top: 12px;
        text-align: center;
    ">
        Customer Churn Prediction Dashboard
    </h3>
    <p style="
        color: #AAAAAA;
        font-size: 14px;
        text-align: center;
        margin-top: 8px;
    ">
    </p>
</div>
""", unsafe_allow_html=True)



try:
    df = load_data()

except FileNotFoundError:
    st.error(
        "Data/netflix_churn.csv nahi mili. "
        "File ka naam aur location check karo."
    )
    st.stop()

required_columns = {
    "subscriber_id",
    "age",
    "gender",
    "subscription_type",
    "watch_hours",
    "last_login_days",
    "region",
    "device",
    "monthly_fee",
    "payment_method",
    "number_of_profiles",
    "avg_watch_time_per_day",
    "favorite_genre",
    "tenure_months",
    "churned"
}

if not required_columns.issubset(df.columns):
    st.error("Netflix CSV ke columns expected format se match nahi karte.")
    st.write("CSV ke available columns:", df.columns.tolist())
    st.stop()

churn_labels = (
    df["churned"]
    .astype(str)
    .str.strip()
    .str.lower()
)

if not churn_labels.isin(["yes", "no"]).all():
    st.error("CSV ke churned column mein sirf Yes/No values honi chahiye.")
    st.stop()


# HISTORICAL METRICS

churned_count = int((churn_labels == "yes").sum())
stayed_count = int((churn_labels == "no").sum())
total = len(df)

historical_rate = (
    churned_count / total * 100
    if total > 0
    else 0
)

st.markdown("### 📊 Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Subscribers", total)
c2.metric("Historical Churned", churned_count)
c3.metric("Stayed", stayed_count)
c4.metric("Historical Churn Rate", f"{historical_rate:.1f}%")

st.caption(
    "Historical metrics sample dataset ke recorded labels par based hain. "
    "Ye future churn predictions nahi hain."
)

st.markdown("---")


# CHART 1: CHURN PIE CHART

st.markdown("### 📈 Subscriber Insights")

left, right = st.columns(2)

with left:
    st.markdown("#### Historical Churn Distribution")

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.pie(
        [churned_count, stayed_count],
        labels=["Churned", "Stayed"],
        autopct="%1.1f%%",
        startangle=90,
        colors=[RED, "#333333"],
        textprops={
            "color": WHITE,
            "fontsize": 12
        },
        wedgeprops={
            "edgecolor": BLACK,
            "linewidth": 3
        },
        pctdistance=0.68
    )

    ax.set_title("Recorded Churn Status", color=WHITE, pad=15)

    st.pyplot(fig)
    plt.close(fig)


# CHART 2: SUBSCRIPTION TYPE

with right:
    st.markdown("#### Churn by Subscription Type")

    chart_data = (
        df.assign(
            Churned=(churn_labels == "yes").astype(int)
        )
        .groupby("subscription_type")["Churned"]
        .mean()
        .mul(100)
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    chart_data.plot(
        kind="bar",
        ax=ax,
        color=RED,
        edgecolor=BLACK,
        width=0.6
    )

    ax.set_ylabel("Historical churn (%)")
    ax.set_xlabel("Subscription")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)

    for bar in ax.patches:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            (bar.get_x() + bar.get_width() / 2, height),
            ha="center",
            va="bottom",
            color=WHITE,
            fontsize=9,
            xytext=(0, 4),
            textcoords="offset points"
        )

    st.pyplot(fig)
    plt.close(fig)



# CHART 3: WATCH HOURS

left, right = st.columns(2)

with left:
    st.markdown("#### Watch Hours vs Last Login")

    fig, ax = plt.subplots(figsize=(7, 5))

    for label, group in df.assign(
        Churned=churn_labels
    ).groupby("Churned"):

        is_churned = label == "yes"

        ax.scatter(
            group["watch_hours"],
            group["last_login_days"],
            label="Churned" if is_churned else "Stayed",
            color=RED if is_churned else GREY,
            edgecolors=BLACK,
            alpha=0.9,
            s=65
        )

    ax.set_xlabel("Watch Hours (monthly)")
    ax.set_ylabel("Days Since Last Login")

    ax.legend(
        title="Recorded Status",
        facecolor=DARK_GREY,
        labelcolor=WHITE,
        edgecolor="#444444"
    )

    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)

    st.pyplot(fig)
    plt.close(fig)


# CHART 4: FAVOURITE GENRES

with right:
    st.markdown("#### Favourite Genres")

    genre_counts = df["favorite_genre"].value_counts()

    fig, ax = plt.subplots(figsize=(7, 5))

    genre_counts.plot(
        kind="bar",
        ax=ax,
        color=RED,
        edgecolor=BLACK,
        width=0.65
    )

    ax.set_ylabel("Subscribers")
    ax.set_xlabel("Favourite Genre")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    st.pyplot(fig)
    plt.close(fig)



# INDIVIDUAL CHURN PREDICTION

st.markdown("---")
st.markdown("### 🔮 Predict Individual Subscriber Churn")

st.write(
    "Subscriber ki details enter karo, phir Predict Churn button dabao."
)

with st.form("prediction_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30
        )

        gender = st.selectbox(
            "Gender",
            ["Female", "Male", "Other"]
        )

        subscription = st.selectbox(
            "Subscription Type",
            ["Basic", "Standard", "Premium"]
        )

        watch_hours = st.number_input(
            "Watch Hours (monthly)",
            min_value=0.0,
            max_value=200.0,
            value=20.0,
            step=1.0
        )

        last_login = st.number_input(
            "Days Since Last Login",
            min_value=0,
            max_value=365,
            value=5
        )

    with col2:
        region = st.selectbox(
            "Region",
            [
                "North America",
                "Europe",
                "Asia",
                "South America",
                "Africa"
            ]
        )

        device = st.selectbox(
            "Main Device",
            [
                "Smart TV",
                "Mobile",
                "Laptop",
                "Tablet",
                "Streaming Device"
            ]
        )

        payment = st.selectbox(
            "Payment Method",
            [
                "Credit Card",
                "Debit Card",
                "PayPal",
                "Gift Card",
                "Digital Wallet"
            ]
        )

        profiles = st.number_input(
            "Number of Profiles",
            min_value=1,
            max_value=10,
            value=2
        )

        tenure = st.number_input(
            "Subscription Tenure (months)",
            min_value=1,
            max_value=240,
            value=12
        )

    with col3:
        avg_watch = st.number_input(
            "Average Watch Time per Day (hours)",
            min_value=0.0,
            max_value=24.0,
            value=1.0,
            step=0.25
        )

        genre = st.selectbox(
            "Favourite Genre",
            [
                "Drama",
                "Comedy",
                "Action",
                "Documentary",
                "Thriller",
                "Romance",
                "Sci-Fi"
            ]
        )

    monthly_fee = {
        "Basic": 7.99,
        "Standard": 15.49,
        "Premium": 22.99
    }[subscription]

    submitted = st.form_submit_button(
        "🎬 Predict Churn",
        use_container_width=True
    )



# SHOW PREDICTION

if submitted:

    customer = {
        "age": age,
        "gender": gender,
        "subscription_type": subscription,
        "watch_hours": watch_hours,
        "last_login_days": last_login,
        "region": region,
        "device": device,
        "monthly_fee": monthly_fee,
        "payment_method": payment,
        "number_of_profiles": profiles,
        "avg_watch_time_per_day": avg_watch,
        "favorite_genre": genre,
        "tenure_months": tenure
    }

    try:
        result, churn_probability, stay_probability = predict_churn(
            customer
        )

        st.markdown("### Prediction Result")

        if churn_probability >= 0.5:
            st.error(f"🔴 {result}")
        else:
            st.success(f"🟢 {result}")

        m1, m2 = st.columns(2)

        m1.metric(
            "Churn Probability",
            f"{churn_probability * 100:.1f}%"
        )

        m2.metric(
            "Stay Probability",
            f"{stay_probability * 100:.1f}%"
        )

        fig, ax = plt.subplots(figsize=(7, 4))

        bars = ax.bar(
            ["Churn", "Stay"],
            [
                churn_probability * 100,
                stay_probability * 100
            ],
            color=[RED, "#333333"],
            edgecolor=BLACK,
            width=0.55
        )

        ax.set_ylabel("Probability (%)")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}%",
                (bar.get_x() + bar.get_width() / 2, height),
                ha="center",
                va="bottom",
                color=WHITE,
                xytext=(0, 4),
                textcoords="offset points"
            )

        st.pyplot(fig)
        plt.close(fig)

    except Exception as error:
        st.error("Error in prediction")
        st.exception(error)


# FOOTER

st.markdown("---")

st.markdown("""
<div style="
    text-align: center;
    width: 100%;
    color: #AAAAAA;
    font-size: 16px;
    padding: 15px 0;
">
    Netflix Customer Churn Prediction | Educational ML Project
</div>
""", unsafe_allow_html=True)
