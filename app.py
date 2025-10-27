import streamlit as st
import numpy as np
import joblib

# === Load saved model and scaler ===
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

# === App Config ===
st.set_page_config(page_title="Smart Diabetes Prediction", page_icon="🩺", layout="centered")

# === Custom CSS for modern look ===
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #f0f9ff, #cbebff, #a6e3e9);
}
h1, h2, h3, h4 {
    color: #004d73;
}
div.stButton>button {
    background-color: #0077b6;
    color: white;
    border-radius: 12px;
    padding: 0.6em 1.2em;
    font-size: 1.05em;
    border: none;
}
div.stButton>button:hover {
    background-color: #023e8a;
    color: #f0f0f0;
}
.result-box {
    border-radius: 15px;
    padding: 20px;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# === Sidebar Navigation ===
menu = ["🏠 Home", "🩺 Precautions & Advice", "ℹ️ About"]
choice = st.sidebar.radio("Navigate", menu)

# === 1️⃣ HOME PAGE ===
if choice == "🏠 Home":
    st.title("🩺 Smart Health Risk Prediction")
    st.markdown("Use this app to predict whether a person is likely to have diabetes based on key health indicators.")
    st.markdown("<hr>", unsafe_allow_html=True)

    st.header("📋 Enter Patient Information")
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("🤰 Pregnancies", min_value=0, value=1)
        glucose = st.number_input("🍬 Glucose Level", min_value=0, value=120)
        blood_pressure = st.number_input("🩸 Blood Pressure (mm Hg)", min_value=0, value=70)
        skin_thickness = st.number_input("🧪 Skin Thickness (mm)", min_value=0, value=20)

    with col2:
        insulin = st.number_input("💉 Insulin Level (mu U/ml)", min_value=0, value=80)
        bmi = st.number_input("⚖️ BMI", min_value=0.0, value=25.0)
        dpf = st.number_input("🧬 Diabetes Pedigree Function", min_value=0.0, value=0.5)
        age = st.number_input("🎂 Age", min_value=0, value=30)

    # === Prediction Section ===
    if st.button("🔍 Predict"):
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                insulin, bmi, dpf, age]])
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)

        st.markdown("---")
        if prediction[0] == 1:
            st.error("⚠️ **Result: At Risk** — The model predicts that the patient is likely to have **diabetes**.")
            st.session_state["result"] = "at_risk"
        else:
            st.success("✅ **Result: Healthy** — The model predicts that the patient is **not likely** to have diabetes.")
            st.session_state["result"] = "healthy"

# === 2️⃣ PRECAUTIONS PAGE ===
elif choice == "🩺 Precautions & Advice":
    st.title("🩺 Health Precautions & Doctor Advice")
    st.markdown("<hr>", unsafe_allow_html=True)

    if "result" not in st.session_state:
        st.warning("⚠️ Please go to the Home page and make a prediction first.")
    else:
        if st.session_state["result"] == "at_risk":
            st.error("⚠️ You are **at risk** for diabetes. Please follow these precautions:")
            st.markdown("""
            **🩸 Lifestyle & Diet Tips:**
            - Eat more **fiber-rich** foods like vegetables and whole grains  
            - Avoid sugary drinks and junk food  
            - Exercise at least **30 minutes daily**  
            - Monitor blood glucose regularly  
            - Maintain a healthy weight  
            - Manage stress through yoga or meditation  

            **👨‍⚕️ Recommended Doctors:**
            - **Endocrinologist** (for diabetes management)  
            - **Dietitian/Nutritionist** (for meal planning)  
            - **Ophthalmologist** (for eye checkups)
            """)
        else:
            st.success("🎉 You are **healthy!** Keep maintaining a good lifestyle.")
            st.markdown("""
            **💚 Maintenance Tips:**
            - Eat balanced meals  
            - Stay physically active  
            - Avoid smoking and alcohol  
            - Get regular checkups  
            - Drink plenty of water  
            - Sleep 7–8 hours per day  
            """)

# === 3️⃣ ABOUT PAGE ===
elif choice == "ℹ️ About":
    st.title("ℹ️ About This App")
    st.markdown("""
    This smart app uses a **Machine Learning model** trained on the *PIMA Indian Diabetes Dataset*  
    to predict the likelihood of diabetes using key medical parameters.
    
    **⚙️ Tech Stack:**
    - Streamlit (Frontend)
    - Scikit-learn (Model Training)
    - NumPy, Pandas (Data Processing)
    - Joblib (Model Serialization)

    **🧠 Model Type:** Logistic Regression / Random Forest  
    **📊 Accuracy:** ~78–82% (depending on training parameters)

    **💡 Developer:** Bharti Saini  
    """)
