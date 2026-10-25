import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Smart Campus Energy Analytics", page_icon="⚡", layout="wide")

@st.cache_data
def generate_data(n=360, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    buildings = rng.choice(["Academic Block","Library","Hostel A","Hostel B","Laboratory"], n,
                           p=[.25,.15,.20,.20,.20])
    temp = np.clip(27+5*np.sin(np.arange(n)*2*np.pi/365)+rng.normal(0,2.2,n),17,40)
    humidity = np.clip(72-.7*(temp-25)+rng.normal(0,7,n),35,95)
    occupancy = rng.integers(80,850,n)
    hours = rng.choice([5,6,7,8,9,10],n)
    weekend = pd.Series(dates).dt.dayofweek >= 5
    occupancy = np.where(weekend,occupancy*.55,occupancy).astype(int)
    factor = pd.Series(buildings).map({"Academic Block":1.0,"Library":.8,"Hostel A":1.15,"Hostel B":1.1,"Laboratory":1.3}).to_numpy()
    energy = 85+.085*occupancy+2.8*temp+4.5*hours+.35*humidity+55*factor+rng.normal(0,18,n)
    prev = np.roll(energy,1); prev[0] = energy[0]-5
    return pd.DataFrame({"Date":dates,"Building":buildings,"Temperature_C":temp.round(2),
        "Humidity_pct":humidity.round(2),"Occupancy":occupancy,"Working_Hours":hours,
        "Previous_Day_Energy_kWh":prev.round(2),"Energy_Consumption_kWh":energy.round(2)})

df = generate_data()
features = ["Temperature_C","Humidity_pct","Occupancy","Working_Hours","Previous_Day_Energy_kWh"]
X_train, X_test, y_train, y_test = train_test_split(df[features],df["Energy_Consumption_kWh"],test_size=.2,random_state=42)
model = LinearRegression().fit(X_train,y_train)
pred = model.predict(X_test)
df["Model_Estimate_kWh"] = model.predict(df[features])
df["Excess_Energy_kWh"] = df["Energy_Consumption_kWh"]-df["Model_Estimate_kWh"]
threshold = max(float(df["Excess_Energy_kWh"].quantile(.90)),1.5*float(df["Excess_Energy_kWh"].std()))
df["Possible_Waste"] = df["Excess_Energy_kWh"] > threshold
metrics = {"MAE":mean_absolute_error(y_test,pred),"RMSE":np.sqrt(mean_squared_error(y_test,pred)),"R2":r2_score(y_test,pred)}

st.title("⚡ Smart Campus Energy Analytics")
st.write("Analyse energy use, forecast demand, flag unusual usage and estimate electricity-related carbon emissions.")
st.warning("DEMO MODE: the dataset is simulated, not connected to real campus meters. Forecasts and flags are illustrative.")
pages = ["Dashboard","Data Analysis","Visual Analytics","Machine Learning","Prediction","AI Energy Forecast","Energy Waste Detection","Carbon Footprint"]
page = st.sidebar.radio("Select Module",pages)
st.sidebar.info("Project by Md Mobashwer Hossain | CSE (AI & ML)")

if page == "Dashboard":
    st.subheader("Campus Energy Dashboard")
    a,b,c,d = st.columns(4)
    a.metric("Total Energy",f"{df.Energy_Consumption_kWh.sum():,.0f} kWh")
    b.metric("Average / Day",f"{df.Energy_Consumption_kWh.mean():,.1f} kWh")
    c.metric("Peak Consumption",f"{df.Energy_Consumption_kWh.max():,.1f} kWh")
    d.metric("Model R²",f"{metrics['R2']:.3f}")
    fig,ax=plt.subplots(figsize=(11,4)); ax.plot(df.Date.tail(30),df.Energy_Consumption_kWh.tail(30),marker="o")
    ax.set_title("Last 30 Days Consumption"); ax.set_ylabel("kWh"); ax.tick_params(axis="x",rotation=40); fig.tight_layout(); st.pyplot(fig)
elif page == "Data Analysis":
    st.subheader("Data Analysis")
    st.dataframe(df.drop(columns=["Model_Estimate_kWh","Excess_Energy_kWh","Possible_Waste"]),use_container_width=True)
    a,b=st.columns(2)
    with a:
        st.write("Descriptive Statistics")
        st.dataframe(df.select_dtypes(include="number").describe().round(2),use_container_width=True)
    with b:
        st.write("Missing Values")
        st.dataframe(df.isna().sum().rename("Missing Values").to_frame(),use_container_width=True)
    st.download_button("Download Demo CSV",df.to_csv(index=False).encode(), "smart_campus_energy_demo.csv","text/csv")
elif page == "Visual Analytics":
    st.subheader("Visual Analytics")
    tab1,tab2,tab3=st.tabs(["Building Usage","Temperature","Correlation"])
    with tab1:
        fig,ax=plt.subplots(figsize=(9,4)); df.groupby("Building").Energy_Consumption_kWh.mean().sort_values().plot(kind="bar",ax=ax); ax.set_ylabel("Average kWh"); fig.tight_layout(); st.pyplot(fig)
    with tab2:
        fig,ax=plt.subplots(figsize=(9,4)); sns.scatterplot(data=df,x="Temperature_C",y="Energy_Consumption_kWh",hue="Building",ax=ax); fig.tight_layout(); st.pyplot(fig)
    with tab3:
        fig,ax=plt.subplots(figsize=(9,5)); sns.heatmap(df.select_dtypes(include="number").corr(),annot=True,fmt=".2f",ax=ax); fig.tight_layout(); st.pyplot(fig)
elif page == "Machine Learning":
    st.subheader("Linear Regression Evaluation")
    a,b,c=st.columns(3); a.metric("MAE",f"{metrics['MAE']:.2f} kWh"); b.metric("RMSE",f"{metrics['RMSE']:.2f} kWh"); c.metric("R²",f"{metrics['R2']:.3f}")
    st.dataframe(pd.DataFrame({"Actual kWh":y_test.to_numpy(),"Predicted kWh":pred}).head(20),use_container_width=True)
    fig,ax=plt.subplots(figsize=(7,4)); ax.scatter(y_test,pred); lo=min(y_test.min(),pred.min()); hi=max(y_test.max(),pred.max()); ax.plot([lo,hi],[lo,hi],"--"); ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); fig.tight_layout(); st.pyplot(fig)
elif page == "Prediction":
    st.subheader("Predict Energy Consumption")
    building=st.selectbox("Building",["Academic Block","Library","Hostel A","Hostel B","Laboratory"])
    a,b=st.columns(2)
    with a:
        temp=st.number_input("Temperature °C",10.,45.,28.); humidity=st.number_input("Humidity %",20.,100.,65.); occupancy=st.number_input("Occupancy",0,1500,450)
    with b:
        hours=st.number_input("Working Hours",0,24,8); previous=st.number_input("Previous day energy kWh",0.,2500.,float(df.Energy_Consumption_kWh.iloc[-1]))
    if st.button("Predict Energy"):
        result=float(model.predict(pd.DataFrame([{"Temperature_C":temp,"Humidity_pct":humidity,"Occupancy":occupancy,"Working_Hours":hours,"Previous_Day_Energy_kWh":previous}]))[0])
        st.success(f"Estimated energy for {building}: {result:,.2f} kWh")
        if result>df.Energy_Consumption_kWh.quantile(.75): st.warning("High estimate: review AC, lighting, equipment and schedules.")
        else: st.info("Estimate is within the demo dataset's usual range.")
elif page == "AI Energy Forecast":
    st.subheader("🔮 AI Energy Forecast")
    days=st.slider("Forecast days",1,14,7)
    recent=df.tail(14); temp=float(recent.Temperature_C.mean()); humidity=float(recent.Humidity_pct.mean()); occupancy=float(recent.Occupancy.mean()); hours=float(recent.Working_Hours.mean()); previous=float(df.Energy_Consumption_kWh.iloc[-1])
    dates=pd.date_range(df.Date.iloc[-1]+pd.Timedelta(days=1),periods=days,freq="D"); rows=[]
    for date in dates:
        occ=occupancy*(.6 if date.dayofweek>=5 else 1)
        sample=pd.DataFrame([{"Temperature_C":temp,"Humidity_pct":humidity,"Occupancy":occ,"Working_Hours":hours,"Previous_Day_Energy_kWh":previous}])
        previous=max(0,float(model.predict(sample)[0])); rows.append({"Forecast Date":date.date(),"Predicted kWh":round(previous,2)})
    forecast=pd.DataFrame(rows); st.dataframe(forecast,use_container_width=True)
    fig,ax=plt.subplots(figsize=(10,4)); ax.plot(forecast["Forecast Date"],forecast["Predicted kWh"],marker="o"); ax.set_ylabel("kWh"); ax.tick_params(axis="x",rotation=35); fig.tight_layout(); st.pyplot(fig)
    st.metric(f"Predicted total for {days} days",f"{forecast['Predicted kWh'].sum():,.1f} kWh")
    st.caption("Illustrative forecast from simulated data; recent average weather and occupancy are assumed.")
elif page == "Energy Waste Detection":
    st.subheader("🚨 Energy Waste / Anomaly Detection")
    st.write("Days with actual simulated consumption well above the model estimate are flagged for investigation—not confirmed waste.")
    st.metric("Potential high-use days",int(df.Possible_Waste.sum()))
    st.metric("Flag threshold above model estimate",f"{threshold:.1f} kWh")
    flagged=df.loc[df.Possible_Waste,["Date","Building","Energy_Consumption_kWh","Model_Estimate_kWh","Excess_Energy_kWh"]].sort_values("Excess_Energy_kWh",ascending=False)
    st.dataframe(flagged.rename(columns={"Energy_Consumption_kWh":"Actual kWh","Model_Estimate_kWh":"Expected kWh","Excess_Energy_kWh":"Above estimate kWh"}),use_container_width=True)
    st.markdown("**Suggested checks:** AC and lighting schedules; equipment left on; occupancy versus use; confirm meter readings before action.")
    fig,ax=plt.subplots(figsize=(10,4)); ax.plot(df.Date,df.Energy_Consumption_kWh,label="Actual"); ax.plot(df.Date,df.Model_Estimate_kWh,label="Model estimate"); ax.scatter(df.loc[df.Possible_Waste,"Date"],df.loc[df.Possible_Waste,"Energy_Consumption_kWh"],label="Flagged"); ax.legend(); ax.set_ylabel("kWh"); fig.tight_layout(); st.pyplot(fig)
elif page == "Carbon Footprint":
    st.subheader("🌱 Carbon Footprint Calculator")
    st.write("Estimate emissions associated with electricity use. Set the factor to the latest official grid value for your location when available.")
    factor=st.number_input("Emission factor (kg CO₂ per kWh)",0.,2.,.716,.001,format="%.3f")
    period=st.selectbox("Period",["Last 7 days","Last 30 days","All demo data"])
    selected=df.tail(7) if period=="Last 7 days" else df.tail(30) if period=="Last 30 days" else df
    energy=float(selected.Energy_Consumption_kWh.sum()); emissions=energy*factor
    a,b,c=st.columns(3); a.metric("Energy",f"{energy:,.1f} kWh"); b.metric("Estimated CO₂",f"{emissions:,.1f} kg"); c.metric("Estimated CO₂",f"{emissions/1000:,.3f} tonnes")
    st.caption("0.716 is an illustrative default, not a verified current official factor. Results use simulated energy data.")
    st.markdown("**Ways to reduce emissions:** optimise AC settings; switch off unused equipment; maintain HVAC; compare like-for-like monthly kWh; consider renewable electricity.")
    by=selected.groupby("Building").Energy_Consumption_kWh.sum()*factor
    fig,ax=plt.subplots(figsize=(9,4)); by.sort_values().plot(kind="bar",ax=ax); ax.set_ylabel("Estimated kg CO₂"); fig.tight_layout(); st.pyplot(fig)

st.markdown("---")
st.caption("Smart Campus Energy Analytics | Md Mobashwer Hossain | CSE (AI & ML) | Simulated demo data")
