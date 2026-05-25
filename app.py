import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)

# =========================================
# CUSTOM CSS
# =========================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

h1, h2, h3 {
    color: white;
}

.stButton>button {
    background: linear-gradient(to right, #ff416c, #ff4b2b);
    color: white;
    border-radius: 12px;
    height: 50px;
    width: 100%;
    font-size: 20px;
    font-weight: bold;
    border: none;
}

.stButton>button:hover {
    background: linear-gradient(to right, #00b09b, #96c93d);
    color: white;
}

.metric-card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
    box-shadow: 0px 0px 15px rgba(255,255,255,0.1);
}

.desc-box {
    background-color: #1c1f26;
    padding: 25px;
    border-radius: 15px;
    color: white;
    box-shadow: 0px 0px 10px rgba(255,255,255,0.1);
}

</style>
""", unsafe_allow_html=True)

# =========================================
# LOAD MODEL
# =========================================

model = pickle.load(open("titanic_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

W1 = model['W1']
b1 = model['b1']
W2 = model['W2']
b2 = model['b2']

# =========================================
# ACTIVATION FUNCTION
# =========================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# =========================================
# PREPROCESS FUNCTION
# =========================================

def preprocess(pclass, age, fare):

    values = np.array([pclass, age, fare])

    normalized = (
        (values - scaler['min']) /
        (scaler['max'] - scaler['min'])
    )

    return normalized.reshape(1, -1)

# =========================================
# PREDICTION FUNCTION
# =========================================

def predict(data):

    hidden_input = np.dot(data, W1) + b1
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W2) + b2
    prediction = sigmoid(final_input)

    return prediction[0][0]

# =========================================
# HEADER
# =========================================

st.markdown("""
<h1 style='text-align:center; color:#ff4b2b;'>
🚢 Titanic Survival Prediction System
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<h3 style='text-align:center; color:white;'>
Deep Learning Based Passenger Survival Prediction
</h3>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================================
# DESCRIPTION
# =========================================

st.markdown("""
<div class="desc-box">

<h3>📌 Project Description</h3>

<p>
This AI-powered application predicts whether a passenger
would survive during an emergency situation using an
Artificial Neural Network developed completely using NumPy.
</p>

<ul>
<li>✔ Passenger Class Analysis</li>
<li>✔ Age-Based Prediction</li>
<li>✔ Fare-Based Risk Analysis</li>
<li>✔ ANN with Forward & Backpropagation</li>
<li>✔ NumPy Based Deep Learning</li>
</ul>

</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================================
# INPUT SECTION
# =========================================

st.markdown("## 🎯 Passenger Details")

col1, col2, col3 = st.columns(3)

with col1:
    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

with col2:
    age = st.slider(
        "Age",
        1,
        80,
        24
    )

with col3:
    fare = st.number_input(
        "Fare Amount",
        0.0,
        600.0,
        120.0
    )

st.markdown("<br>", unsafe_allow_html=True)

# =========================================
# PREDICT BUTTON
# =========================================

if st.button("🚀 Predict Survival"):

    processed_data = preprocess(
        pclass,
        age,
        fare
    )

    prediction = predict(processed_data)

    survived_prob = float(prediction)
    not_survived_prob = 1 - survived_prob

    st.markdown("---")

    # =====================================
    # RESULT
    # =====================================

    st.markdown("## 🧠 Prediction Result")

    if survived_prob > 0.5:
        st.success("✅ Passenger is likely to SURVIVE")
    else:
        st.error("❌ Passenger is likely NOT to survive")

    # =====================================
    # METRICS
    # =====================================

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric(
            "Survival Probability",
            f"{survived_prob*100:.2f}%"
        )

    with m2:
        st.metric(
            "Non-Survival Probability",
            f"{not_survived_prob*100:.2f}%"
        )

    with m3:
        st.metric(
            "Confidence Score",
            f"{max(survived_prob, not_survived_prob)*100:.2f}%"
        )

    st.markdown("---")

    # =====================================
    # CHART
    # =====================================

    st.markdown("## 📊 Probability Visualization")

    labels = ['Survived', 'Not Survived']
    values = [survived_prob, not_survived_prob]

    fig, ax = plt.subplots(figsize=(6,4))

    bars = ax.bar(labels, values)

    ax.set_ylim([0,1])

    ax.set_ylabel("Probability")

    ax.set_title("Survival Prediction Analysis")

    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            yval + 0.02,
            f"{yval:.2f}",
            ha='center'
        )

    st.pyplot(fig)

# =========================================
# FOOTER
# =========================================

st.markdown("""
<hr>

<center>




</center>
""", unsafe_allow_html=True)