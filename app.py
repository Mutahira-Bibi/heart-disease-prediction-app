import streamlit as st
import joblib
import pandas as pd

# Load model files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")


# Page settings
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)
`

# Sidebar
st.sidebar.title("❤️ Heart Disease")
st.sidebar.write("ML Based Health Prediction")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "❤️ Prediction",
        "📊 Model Performance",
        "🤖 About Model",
        "📈 Visualizations",
        "📁 Dataset Info",
        "✉️ Contact"
    ]
)


# =========================================================
# HOME PAGE
# =========================================================

if page == "🏠 Home":

    st.title("❤️ Heart Disease Prediction App")

    st.subheader("Machine Learning Based Health Prediction")

    st.write(
        "Welcome to the Heart Disease Prediction App. "
        "This project uses Machine Learning to predict "
        "heart disease from patient information."
    )

    st.write("---")

    st.subheader("📊 Project Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Model", "Logistic Regression")
    col2.metric("Accuracy", "86.96%")
    col3.metric("Dataset", "918 Patients")

    st.write("---")

    st.subheader("🚀 How This App Works")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("### 1️⃣ Enter Information")
        st.write(
            "Enter the required patient information "
            "on the Prediction page."
        )

    with col2:
        st.write("### 2️⃣ Machine Learning")
        st.write(
            "The trained Logistic Regression model "
            "processes the information."
        )

    with col3:
        st.write("### 3️⃣ Get Prediction")
        st.write(
            "The model gives a prediction based on "
            "the entered information."
        )

    st.write("---")

     st.success("👩‍💻 Developed by Mutahira Bibi | Final Year Project 2026")
    st.caption("Heart Disease Prediction System - For Educational Purpose")


# =========================================================
# PREDICTION PAGE
# =========================================================

if page == "❤️ Prediction":

    st.title("❤️ Heart Disease Prediction")

    st.write(
        "Enter the patient information below and click Predict."
    )

    st.info(
        "⚠️ This is an educational machine learning prediction "
        "and is not a medical diagnosis."
    )

    st.write("---")

    st.subheader("👤 Patient Information")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=100,
            value=50
        )

        sex = st.selectbox(
            "Sex",
            ["Male", "Female"]
        )

        chest_pain = st.selectbox(
            "Chest Pain Type",
            ["ASY", "ATA", "NAP", "TA"]
        )

        resting_bp = st.number_input(
            "Resting Blood Pressure",
            min_value=1,
            max_value=250,
            value=120
        )

        cholesterol = st.number_input(
            "Cholesterol",
            min_value=1,
            max_value=600,
            value=200
        )

    with col2:

        fasting_bs = st.selectbox(
            "Fasting Blood Sugar",
            [0, 1]
        )

        resting_ecg = st.selectbox(
            "Resting ECG",
            ["Normal", "ST", "LVH"]
        )

        max_hr = st.number_input(
            "Maximum Heart Rate",
            min_value=1,
            max_value=250,
            value=150
        )

        exercise_angina = st.selectbox(
            "Exercise Angina",
            ["N", "Y"]
        )

        oldpeak = st.number_input(
            "Oldpeak",
            min_value=0.0,
            max_value=10.0,
            value=1.0
        )

        st_slope = st.selectbox(
            "ST Slope",
            ["Up", "Flat", "Down"]
        )

    st.write("---")

    if st.button("🔍 Predict Heart Disease"):

        input_data = pd.DataFrame(
            0,
            index=[0],
            columns=columns
        )

        input_data["Age"] = age
        input_data["RestingBP"] = resting_bp
        input_data["Cholesterol"] = cholesterol
        input_data["FastingBS"] = fasting_bs
        input_data["MaxHR"] = max_hr
        input_data["Oldpeak"] = oldpeak

        if sex == "Male":
            input_data["Sex_M"] = 1

        if chest_pain == "ATA":
            input_data["ChestPainType_ATA"] = 1

        elif chest_pain == "NAP":
            input_data["ChestPainType_NAP"] = 1

        elif chest_pain == "TA":
            input_data["ChestPainType_TA"] = 1

        if resting_ecg == "Normal":
            input_data["RestingECG_Normal"] = 1

        elif resting_ecg == "ST":
            input_data["RestingECG_ST"] = 1

        if exercise_angina == "Y":
            input_data["ExerciseAngina_Y"] = 1

        if st_slope == "Flat":
            input_data["ST_Slope_Flat"] = 1

        elif st_slope == "Up":
            input_data["ST_Slope_Up"] = 1

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]

        st.write("---")

        st.subheader("🔍 Prediction Result")

        if prediction == 1:

            st.error("❤️ Heart Disease Detected by Model")

        else:

            st.success("✅ No Heart Disease Detected by Model")


# =========================================================
# MODEL PERFORMANCE PAGE
# =========================================================

if page == "📊 Model Performance":

    st.title("📊 Model Performance")

    st.write(
        "This section shows the performance of the Logistic Regression model."
    )

    st.subheader("Model Evaluation")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", "86.96%")
    col2.metric("Precision", "91.09%")
    col3.metric("Recall", "85.98%")
    col4.metric("F1 Score", "88.46%")

    st.write("---")

    st.subheader("Confusion Matrix")

    st.write("Model correctly predicted 160 out of 184 patients")
    st.write("✅ True Positives (92): Disease correctly predicted")
    st.write("✅ True Negatives (68): No disease correctly predicted")
    st.write(
        "❌ False Positives (9) & False Negatives (15): "
        "Incorrect predictions"
    )

    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = [
        [68, 9],
        [15, 92]
    ]

    fig, ax = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix Heatmap")

    st.pyplot(fig)


# =========================================================
# ABOUT MODEL PAGE
# =========================================================

if page == "🤖 About Model":

    st.title("🤖 About the Model")

    st.subheader("❤️ Logistic Regression")

    st.write(
        "This application uses a Logistic Regression machine learning "
        "model to make a prediction based on patient information."
    )

    st.write(
        "The model was trained using patient health-related features "
        "from the Heart Disease dataset."
    )

    st.write("---")

    st.subheader("📌 Features Used")

    col1, col2 = st.columns(2)

    with col1:

        st.write("• Age")
        st.write("• Sex")
        st.write("• Chest Pain Type")
        st.write("• Resting Blood Pressure")
        st.write("• Cholesterol")

    with col2:

        st.write("• Fasting Blood Sugar")
        st.write("• Resting ECG")
        st.write("• Maximum Heart Rate")
        st.write("• Exercise Angina")
        st.write("• Oldpeak")
        st.write("• ST Slope")

    st.write("---")

    st.subheader("📊 Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", "86.96%")
    col2.metric("Precision", "91.09%")
    col3.metric("Recall", "85.98%")
    col4.metric("F1 Score", "88.46%")

    st.write("---")

    st.subheader("🔍 Model Output")

    st.success("0 → No Heart Disease")
    st.error("1 → Heart Disease")

    st.info(
        "⚠️ This application is an educational machine learning project "
        "and should not be used as a medical diagnosis."
    )


# =========================================================
# VISUALIZATIONS PAGE
# =========================================================

if page == "📈 Visualizations":

    st.title("📈 Data Visualizations")

    st.write(
        "Explore different patterns in the Heart Disease dataset."
    )

    df = pd.read_csv("heart.csv")

    st.subheader("❤️ Heart Disease Distribution")

    disease_count = df["HeartDisease"].value_counts()

    st.bar_chart(disease_count)

    st.write("---")

    st.subheader("👥 Heart Disease by Sex")

    sex_disease = pd.crosstab(
        df["Sex"],
        df["HeartDisease"]
    )

    st.bar_chart(sex_disease)

    st.write("---")

    st.subheader("🫀 Chest Pain Type")

    chest_pain_disease = pd.crosstab(
        df["ChestPainType"],
        df["HeartDisease"]
    )

    st.bar_chart(chest_pain_disease)

    st.write("---")

    st.subheader("📊 Age Distribution")

    st.bar_chart(
        df["Age"].value_counts().sort_index()
    )


# =========================================================
# DATASET INFO PAGE
# =========================================================

if page == "📁 Dataset Info":

    st.title("📁 Dataset Information")

    df = pd.read_csv("heart.csv")

    st.subheader("📊 Dataset Size")

    col1, col2 = st.columns(2)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])

    st.write("---")

    st.subheader("👀 First 5 Rows")

    st.dataframe(df.head())

    st.write("---")

    st.subheader("📌 Dataset Columns")

    st.write(df.columns.tolist())

    st.write("---")

    st.subheader("❓ Missing Values")

    missing_values = df.isnull().sum()

    st.dataframe(missing_values)


# =========================================================
# CONTACT PAGE
# =========================================================

if page == "✉️ Contact":

    st.title("✉️ Contact")

    st.subheader("About This Project")

    st.write(
        "This Heart Disease Prediction App is a machine learning "
        "project developed for educational and portfolio purposes."
    )

    st.write("---")

    st.subheader("👩‍💻 Project Information")

    st.write("• Project: Heart Disease Prediction")
    st.write("• Machine Learning Model: Logistic Regression")
    st.write("• Dataset: Heart Disease Dataset")
    st.write("• App Framework: Streamlit")
    st.write("• Programming Language: Python")

    st.write("---")

    st.subheader("📚 Purpose")

    st.write(
        "The purpose of this project is to demonstrate how "
        "machine learning can be used to make predictions "
        "from patient information."
    )

    st.info(
        "⚠️ This application is an educational project "
        "and should not be used as a medical diagnosis."
    )
