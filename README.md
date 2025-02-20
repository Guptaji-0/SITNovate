# SITNovate

No worries! If you didn't include a `requirements.txt` file, we can modify the **README** accordingly.  

Here’s the updated **README.md** without `pip install -r requirements.txt` and with a manual installation approach instead.

---

# 🌾 AI-Powered Crop Recommendation System /  AgroTechHub 
 
🚀 **Empowering Farmers with Data-Driven Insights**  

## 📌 Project Overview  
This project leverages **machine learning and AI** to assist farmers by providing:  
✅ **Price Prediction** – Forecasts crop prices based on historical data.  
✅ **Final Price Prediction** – Enhances initial predictions with additional market factors.  
✅ **Yield Prediction** – Estimates crop yield based on soil and environmental conditions.  
✅ **Crop Recommendation** – Suggests the best crop based on soil nutrients and climate.  

🔹 **Tech Stack:** Python, Scikit-Learn, Pandas, NumPy, Matplotlib, Pyttsx3 (Text-to-Speech), Streamlit (Frontend).  

---

## 📊 1. Data Collection & Preprocessing  
### **1.1 Web Scraping & Data Gathering**  
We collected **agricultural datasets** from various sources:  
- **Crop Pricing Data:** Scraped from agricultural market databases.  
- **Soil & Climate Data:** Sourced from government databases & weather APIs.  
- **Historical Yield Data:** Obtained from research papers and agricultural studies.  

### **1.2 Data Preprocessing & Cleaning**  
- **Concatenation** – Merging multiple datasets into a unified DataFrame (`df`).  
- **Handling Missing Values** – Replacing missing entries with mean/median values.  
- **Feature Engineering** – Extracting useful parameters for prediction.  
- **Scaling & Normalization** – Standardizing numerical features for ML models.  

---

## 🧠 2. Model Development  
### **2.1 Price Prediction Model**  
- Built an **initial crop price prediction model** using **RandomForestRegressor**.  
- Features: **Historical prices, region, demand, seasonality.**  
- Output: **Estimated future price of a crop.**  

### **2.2 Final Price Prediction Model**  
- Enhanced the initial price prediction model by integrating **real-time market trends**.  
- Added additional features like **inflation rate, transportation costs, and weather impact.**  

### **2.3 Crop Yield Prediction**  
- Trained a **RandomForestClassifier** to predict yield based on soil composition and rainfall.  
- Features: **Nitrogen (N), Phosphorus (P), Potassium (K), Temperature, Humidity, pH, Rainfall.**  

### **2.4 Crop Recommendation Model**  
- Suggests the best crop to grow based on soil and environmental conditions.  
- Uses **classification models (DecisionTree, RandomForest) and AI-based NLP**.  

### **2.5 Model Serialization (pkl files)**  
- Each trained model is saved as a `.pkl` file for integration with the AI system.  
- Example: `price_prediction.pkl`, `yield_prediction.pkl`, `crop_recommendation.pkl`.  

---

## 🛠️ 3. AI Integration & System Development  
### **3.1 Backend (Python, Flask, FastAPI)**  
- Developed REST API endpoints to interact with the models.  
- AI-powered **voice assistant** integrated using **Pyttsx3 (Text-to-Speech)**.  
- API serves real-time predictions and crop recommendations.  

### **3.2 Frontend (Streamlit)**  
- Built a **user-friendly UI** for farmers to enter soil parameters and receive predictions.  
- Features:  
  ✅ Interactive input fields for soil, climate, and crop details.  
  ✅ Dynamic crop recommendations based on ML model predictions.  
  ✅ Voice output for recommendations.  

---

## 🏗️ 4. Installation & Setup  
### **Step 1: Clone Repository**  
```bash
git clone https://github.com/yourusername/ai-crop-recommendation.git
cd ai-crop-recommendation
```

### **Step 2: Create Virtual Environment**  
```bash
python -m venv env
source env/bin/activate  # On macOS/Linux
env\Scripts\activate     # On Windows
```

### **Step 3: Install Dependencies (Manually)**  
Since `requirements.txt` is not included, install dependencies manually:  
```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit pyttsx3 flask fastapi joblib
```

### **Step 4: Import Data Files**  
Ensure all CSV data files are inside the `DATA/` folder.  
```bash
mkdir DATA
mv your_data_file.csv DATA/
```

### **Step 5: Run the Application**  
#### **Backend API (Flask/FastAPI)**  
```bash
python app.py
```
#### **Frontend (Streamlit UI)**  
```bash
streamlit run app.py
```

---

## 📂 5. Project Structure  
```
📦 ai-crop-recommendation
│── 📁 DATA/                   # Dataset folder
│── 📁 models/                 # Saved ML models (.pkl files)
│── 📁 backend/                # Flask/FastAPI backend code
│── 📁 frontend/               # Streamlit UI code
│── app.py                     # Main script
│── README.md                   # Project documentation
│── .env                        # API keys and environment variables
```

---

## 🚀 Future Improvements  
- ✅ **Live market price updates** using web scraping APIs.  
- ✅ **Mobile App Integration** (Android/iOS).  
- ✅ **More AI-powered insights** using deep learning models.  

---

## 🤝 Contributors  
👤 **Your Name** - Lead Developer  
👤 **Team Members** - Data Scientists, AI Engineers  

🔗 **GitHub Repository:** [Your Repo Link]  
📧 **Contact:** your.email@example.com  

---
### **🌟 If you like this project, give it a ⭐ on GitHub!**  
🚀 Happy Coding! 🚀  

---

This version **removes the `pip install -r requirements.txt` step** and instead provides **manual installation instructions** for all required packages. Let me know if you need further edits! 🚀
