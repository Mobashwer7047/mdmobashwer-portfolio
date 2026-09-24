# Smart Campus Energy Analytics & Prediction

**Student:** Md Mobashwer Hossain  
**Course:** B.Tech CSE (AI & ML)  
**Project Type:** Data Analytics + Machine Learning + Interactive Dashboard

## 1. Project Overview
Smart Campus Energy Analytics & Prediction is a Python-based project that analyzes electricity consumption patterns in campus buildings and predicts daily energy usage.

The application includes:
- Synthetic campus energy data generation
- Data cleaning and validation
- Exploratory data analysis
- Interactive charts
- Correlation analysis
- Linear Regression model
- Model evaluation using MAE, RMSE and R²
- A prediction form for estimating energy consumption
- Downloadable processed data

The project is designed to run without requiring a separate dataset. A realistic sample dataset is generated automatically when the application starts.

## 2. Technology Stack
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit

## 3. Installation

```bash
python -m venv venv
```

### Windows
```bash
venv\Scripts\activate
```

### Linux / Ubuntu / macOS
```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 4. Run the Project

```bash
streamlit run app.py
```

The application will open in a browser.

## 5. Main Features
### Dashboard
Shows total energy usage, average daily consumption, peak consumption and model R² score.

### Data Analysis
Displays the generated dataset, descriptive statistics and missing-value checks.

### Visual Analytics
Provides:
- Daily energy consumption trend
- Energy consumption by building
- Temperature vs energy consumption
- Feature correlation heatmap

### Machine Learning
A Linear Regression model uses:
- Temperature
- Humidity
- Occupancy
- Working Hours
- Previous Day Energy

to predict daily energy consumption.

### Prediction
Users can enter building conditions and receive an estimated energy consumption value.

## 6. Project Structure

```text
Md_Mobashwer_Hossain_Smart_Campus_Project/
├── app.py
├── requirements.txt
├── README.md
└── Smart_Campus_Project_Report.docx
```

## 7. Educational Purpose
This project demonstrates how data analytics and machine learning can be combined with a simple web dashboard to solve a practical campus-management problem.

## 8. Author
**Md Mobashwer Hossain**
