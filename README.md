# hospital_data

### Problem Statement-
Small hospitals generate large amounts of patient data but rarely analyse it to predict resource needs like beds, oxygen, or critical care. How could data-driven forecasting improve healthcare preparedness?

---

## 🎯 Objective
To build a simple machine learning system that predicts:
- 🛏 Number of beds required  
- 🫧 Oxygen demand  

based on patient admissions.

---

## ⚙️ Approach (Data Science Lifecycle)
This project follows the standard data science workflow:

- **Question:** Can we predict hospital resource demand?  
- **Data:** Historical admissions, bed usage, and oxygen usage  
- **Model:** Linear Regression  
- **Insight:** Future resource requirements based on input  

---

## 🧠 Features
- 📊 Predicts hospital bed requirements  
- 🫧 Predicts oxygen demand  
- 📈 Dynamic visualizations (line, scatter, bar charts)  
- 🎚 Interactive dashboard using Streamlit  
- ⚠️ Alerts for high resource demand  

---

## 🛠️ Tech Stack
- Python  
- Pandas (data handling)  
- Scikit-learn (machine learning)  
- Streamlit (dashboard UI)  
- Matplotlib (initial visualization)  

---

## 📂 Project Structure

hospital-forecast/
├── data/
│ └── data.csv
├── src/
│ ├── model.py
│ └── app.py
├── outputs/
│ ├── beds_trend.png
│ ├── oxygen_trend.png
│ └── relationship.png
├── README.md
├── requirements.txt
└── LICENSE

## ▶️ How to Run

### 1. Install dependencies

pip install -r requirements.txt


### 2. Run Streamlit app

python -m streamlit run app.py


---

## 📊 Output
The dashboard allows users to:
- Input expected patient admissions  
- View predicted bed and oxygen requirements  
- Analyze trends and relationships through graphs  

---

## 💡 Insights
- There is a strong positive relationship between patient admissions and resource usage  
- As admissions increase, both bed and oxygen demand rise  

---

## ⚠️ Limitations
- Uses a small synthetic dataset  
- Does not include ICU or emergency scenarios  
- Assumes linear relationship between variables  

---

## 🚀 Future Improvements
- Add ICU and ventilator prediction  
- Use real-world hospital datasets  
- Implement advanced ML models  
- Deploy as a web application  

---

## 👨‍💻 Author
Developed as part of Sprint #3: Applied Data Science Foundations