import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Smart Campus Energy Analytics",
    page_icon="⚡",
    layout="wide"
)

# ---------- Data Generation ----------
@st.cache_data
def generate_data(n=360, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n, freq="D")

    buildings = rng.choice(
        ["Academic Block", "Library", "Hostel A", "Hostel B", "Laboratory"],
        size=n,
        p=[0.25, 0.15, 0.20, 0.20, 0.20]
    )

    base_temp = 27 + 5*np.sin(np.arange(n) * 2*np.pi/365)
    temperature = np.clip(base_temp + rng.normal(0, 2.2, n), 17, 40)
    humidity = np.clip(72 - 0.7*(temperature-25) + rng.normal(0, 7, n), 35, 95)
    occupancy = rng.integers(80, 850, n)
    working_hours = rng.choice([5, 6, 7, 8, 9, 10], size=n)
    weekend = pd.Series(dates).dt.dayofweek >= 5
    occupancy = np.where(weekend, occupancy * 0.55, occupancy)
    occupancy = occupancy.astype(int)

    building_factor = pd.Series(buildings).map({
        "Academic Block": 1.00,
        "Library": 0.80,
        "Hostel A": 1.15,
        "Hostel B": 1.10,
        "Laboratory": 1.30
    }).to_numpy()

    energy = (
        85
        + 0.085 * occupancy
        + 2.8 * temperature
        + 4.5 * working_hours
        + 0.35 * humidity
        + 55 * building_factor
        + rng.normal(0, 18, n)
    )

    # Previous-day energy feature is created from the generated target.
    previous_energy = np.roll(energy, 1)
    previous_energy[0] = energy[0] - 5

    df = pd.DataFrame({
        "Date": dates,
        "Building": buildings,
        "Temperature_C": np.round(temperature, 2),
        "Humidity_pct": np.round(humidity, 2),
        "Occupancy": occupancy,
        "Working_Hours": working_hours,
        "Previous_Day_Energy_kWh": np.round(previous_energy, 2),
        "Energy_Consumption_kWh": np.round(energy, 2)
    })

    return df


@st.cache_resource
def train_model(df):
    features = [
        "Temperature_C",
        "Humidity_pct",
        "Occupancy",
        "Working_Hours",
        "Previous_Day_Energy_kWh"
    ]
    X = df[features]
    y = df["Energy_Consumption_kWh"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "R2": r2_score(y_test, pred)
    }

    return model, metrics, y_test, pred


df = generate_data()
model, metrics, y_test, predictions = train_model(df)

# ---------- Header ----------
st.title("⚡ Smart Campus Energy Analytics")
st.write(
    "An interactive data analytics and machine learning dashboard "
    "for understanding and predicting campus electricity consumption."
)

# ---------- Sidebar ----------
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Select Module",
    ["Dashboard", "Data Analysis", "Visual Analytics", "Machine Learning", "Prediction"]
)

st.sidebar.markdown("---")
st.sidebar.info("Project by Md Mobashwer Hossain")

# ---------- Dashboard ----------
if page == "Dashboard":
    st.subheader("Campus Energy Dashboard")

    total = df["Energy_Consumption_kWh"].sum()
    avg = df["Energy_Consumption_kWh"].mean()
    peak = df["Energy_Consumption_kWh"].max()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Energy", f"{total:,.0f} kWh")
    c2.metric("Average / Day", f"{avg:,.1f} kWh")
    c3.metric("Peak Consumption", f"{peak:,.1f} kWh")
    c4.metric("Model R²", f"{metrics['R2']:.3f}")

    st.markdown("### Recent Consumption")
    recent = df.tail(30)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(recent["Date"], recent["Energy_Consumption_kWh"], marker="o")
    ax.set_title("Last 30 Days Energy Consumption")
    ax.set_xlabel("Date")
    ax.set_ylabel("Energy (kWh)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    st.pyplot(fig)

# ---------- Data Analysis ----------
elif page == "Data Analysis":
    st.subheader("Data Analysis")

    st.write("### Dataset Preview")
    st.dataframe(df, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.write("### Descriptive Statistics")
        st.dataframe(df.describe(numeric_only=True), use_container_width=True)

    with c2:
        st.write("### Missing Values")
        missing = df.isna().sum().rename("Missing Values").to_frame()
        st.dataframe(missing, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Dataset CSV",
        csv,
        "smart_campus_energy.csv",
        "text/csv"
    )

# ---------- Visual Analytics ----------
elif page == "Visual Analytics":
    st.subheader("Visual Analytics")

    tab1, tab2, tab3 = st.tabs(
        ["Building Usage", "Temperature Relationship", "Correlation"]
    )

    with tab1:
        building_avg = (
            df.groupby("Building")["Energy_Consumption_kWh"]
            .mean()
            .sort_values(ascending=False)
        )
        fig, ax = plt.subplots(figsize=(9, 4))
        building_avg.plot(kind="bar", ax=ax)
        ax.set_title("Average Energy Consumption by Building")
        ax.set_ylabel("Average kWh")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        st.pyplot(fig)

    with tab2:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.scatterplot(
            data=df,
            x="Temperature_C",
            y="Energy_Consumption_kWh",
            hue="Building",
            ax=ax
        )
        ax.set_title("Temperature vs Energy Consumption")
        fig.tight_layout()
        st.pyplot(fig)

    with tab3:
        numeric = df.select_dtypes(include=np.number)
        corr = numeric.corr()
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", ax=ax)
        ax.set_title("Feature Correlation Heatmap")
        fig.tight_layout()
        st.pyplot(fig)

# ---------- Machine Learning ----------
elif page == "Machine Learning":
    st.subheader("Machine Learning Model")

    st.write("### Algorithm")
    st.write(
        "Linear Regression is used to estimate energy consumption from "
        "environmental and campus-activity features."
    )

    st.write("### Evaluation Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{metrics['MAE']:.2f}")
    m2.metric("RMSE", f"{metrics['RMSE']:.2f}")
    m3.metric("R² Score", f"{metrics['R2']:.3f}")

    comparison = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": predictions
    }).reset_index(drop=True)

    st.write("### Actual vs Predicted")
    st.dataframe(comparison.head(20), use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_test, predictions)
    line_min = min(y_test.min(), predictions.min())
    line_max = max(y_test.max(), predictions.max())
    ax.plot([line_min, line_max], [line_min, line_max])
    ax.set_xlabel("Actual Energy (kWh)")
    ax.set_ylabel("Predicted Energy (kWh)")
    ax.set_title("Actual vs Predicted Energy")
    fig.tight_layout()
    st.pyplot(fig)

# ---------- Prediction ----------
else:
    st.subheader("Predict Energy Consumption")

    st.write(
        "Enter the expected conditions for a day. The trained model "
        "will estimate electricity consumption."
    )

    building = st.selectbox(
        "Building",
        ["Academic Block", "Library", "Hostel A", "Hostel B", "Laboratory"]
    )

    c1, c2 = st.columns(2)
    with c1:
        temperature = st.number_input("Temperature (°C)", 10.0, 45.0, 28.0)
        humidity = st.number_input("Humidity (%)", 20.0, 100.0, 65.0)
        occupancy = st.number_input("Occupancy", 0, 1500, 450)
    with c2:
        hours = st.number_input("Working Hours", 0, 24, 8)
        previous = st.number_input(
            "Previous Day Energy (kWh)", 50.0, 2500.0, 700.0
        )

    if st.button("Predict Energy"):
        sample = pd.DataFrame([{
            "Temperature_C": temperature,
            "Humidity_pct": humidity,
            "Occupancy": occupancy,
            "Working_Hours": hours,
            "Previous_Day_Energy_kWh": previous
        }])

        result = model.predict(sample)[0]

        st.success(
            f"Estimated energy consumption for {building}: "
            f"{result:,.2f} kWh"
        )

        if result > df["Energy_Consumption_kWh"].quantile(0.75):
            st.warning("The predicted usage is relatively high. Consider checking HVAC, lighting and occupancy schedules.")
        else:
            st.info("The predicted usage is within the normal range of the generated campus dataset.")

st.markdown("---")
st.caption("Smart Campus Energy Analytics | Md Mobashwer Hossain | CSE (AI & ML)")
