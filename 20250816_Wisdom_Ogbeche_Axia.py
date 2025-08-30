import streamlit as st
import pandas as pd
import pickle 
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.linear_model import LogisticRegression
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="Wisdom Loan Predictor",
    page_icon="💳",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ------------------- CUSTOM CSS -------------------
st.markdown("""
    <style>
        /* App background with lilac gradient */
        .stApp {
            background: linear-gradient(135deg, #E6E6FA 0%, #D8BFD8 100%);
            background-attachment: fixed;
            background-size: cover;
            background-repeat: no-repeat;
        }

        /* Main title */
        .title {
            font-size: 36px;
            font-weight: 800;
            color: #4B0082;
            text-align: center;
            margin-bottom: 20px;
        }

        /* Section headers */
        .section-header {
            font-size: 20px;
            font-weight: 600;
            color: #6A5ACD;
            margin-top: 30px;
            margin-bottom: 10px;
            border-left: 6px solid #6A5ACD;
            padding-left: 10px;
        }

        /* Input boxes styling */
        .stSelectbox, .stNumberInput {
            border-radius: 8px !important;
        }

        /* Cards for results */
        .result-card {
            padding: 25px;
            border-radius: 15px;
            background: linear-gradient(135deg, #F8F0FF, #E6E6FA);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            text-align: center;
            margin-top: 25px;
            border: 1px solid #D1C4E9;
        }

        .result-card h3 {
            font-size: 24px;
            font-weight: 700;
            color: #4A235A;
        }

        .result-card p {
            font-size: 18px;
            margin-top: 10px;
            color: #512E5F;
        }

        /* Button styling */
        div.stButton > button:first-child {
            background: linear-gradient(to right, #9370DB, #8A2BE2);
            color: white;
            border-radius: 10px;
            padding: 0.6em 1.2em;
            font-size: 16px;
            font-weight: 600;
            border: none;
            box-shadow: 0px 3px 8px rgba(0,0,0,0.2);
            transition: 0.3s;
        }

        div.stButton > button:first-child:hover {
            background: linear-gradient(to right, #8A2BE2, #9370DB);
            transform: scale(1.02);
        }
    </style>
""", unsafe_allow_html=True)

# ------------------- LOAD MODEL -------------------
with open("wisdom_loan_predictor.pkl", 'rb') as f:
    model = pickle.load(f) 

# ------------------- APP TITLE -------------------
st.markdown('<p class="title">💳 Wisdom Loan Predictor App</p>', unsafe_allow_html=True)
st.write("Please provide your details below to check your loan eligibility.")

# ------------------- USER INPUTS -------------------
st.markdown('<p class="section-header">📋 Loan Details</p>', unsafe_allow_html=True)

valid_loannumber_x = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 
                     19, 20, 22, 23, 27]
valid_loanamount_x = [10000.0, 15000.0, 20000.0, 25000.0, 30000.0, 
                      35000.0, 40000.0, 50000.0, 60000.0]
valid_termdays_x = [15, 30, 60, 90]
valid_termdays_y = [15, 30, 60, 90]
valid_loannumber_y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 
                     19, 20, 22, 23, 27]
valid_loanamount_y = [3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 15000,
 20000, 25000, 30000, 35000, 40000, 50000, 60000]
valid_Banktypes = ["Current", "Savings", "Other"]
valid_Bankname = ['Access Bank', 'Diamond Bank', 'EcoBank', 'FCMB', 'Fidelity Bank',
 'First Bank', 'GT Bank', 'Heritage Bank', 'Keystone Bank',
 'Skye Bank', 'Stanbic IBTC', 'Standard Chartered', 'Sterling Bank',
 'UBA', 'Union Bank', 'Unity Bank', 'Wema Bank', 'Zenith Bank']
valid_employment = ['Contract', 'Permanent', 'Retired', 'Self-Employed', 'Student', 'Unemployed']
valid_Educvation = ['Graduate', 'Post-Graduate', 'Primary', 'Secondary']

loan_x_to_totaldue_x = {
    10000: [11500, 12250, 12500, 13000, 10750, 11000, 11125, 10250, 10500],
    15000: [17250, 18375, 15750, 16500],
    20000: [22250, 23000, 22000, 21500, 20500, 21000],
    25000: [28750, 27500],
    30000: [34500, 33000, 31500],
    35000: [39000, 38500],
    40000: [44000, 42000, 43500],
    50000: [52500, 55000, 57500],
    60000: [68100, 65400, 62700, 65000, 66500]
}

loan_y_to_totaldue_y = {
    3000: [3900, 5200],
    4000: [4600],
    5000: [5750, 6125],
    6000: [6900, 7800],
    7000: [8050],
    8000: [9200],
    9000: [10350, 10625],
    10000: [11500, 13000, 11125, 12250, 11000, 10750, 11400, 11450, 11700, 11750],
    15000: [17250, 18375, 15750, 16500, 16600, 16675, 16125],
    20000: [22250, 23000, 22000, 21500, 20500, 21000, 21750, 21800, 21700],
    25000: [28750, 27500, 26250, 26875],
    30000: [34500, 33000, 31500, 33900, 34100],
    35000: [39000, 38500, 39450, 36200],
    40000: [44000, 42000, 43500, 44800, 47600, 47100, 41900],
    50000: [52500, 55000, 57500, 47500],
    60000: [68100, 62700]
}

# Grouped inputs for better layout
col1, col2 = st.columns(2)
with col1:
    loannumber_x = st.selectbox("🔢 Loan Number", valid_loannumber_x)
    loanamount_x = st.selectbox("💰 Loan Amount ($)", valid_loanamount_x)
    termdays_x = st.selectbox("📅 Term (Days)", valid_termdays_x)
    totaldue_x = st.selectbox("💵 Total Due ($)", loan_x_to_totaldue_x.get(loanamount_x, [6000.0]))
with col2:
    loannumber_y = st.selectbox("🔢 Previous Loan Number", valid_loannumber_y)
    loanamount_y = st.selectbox("💰 Previous Loan Amount ($)", valid_loanamount_y)
    totaldue_y = st.selectbox("💵 Previous Total Due ($)", loan_y_to_totaldue_y.get(loanamount_y, [0.0]))
    termdays_y = st.selectbox("📅 Previous Term (Days)", valid_termdays_y)

st.markdown('<p class="section-header">👤 Client Details</p>', unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    age = st.number_input('🎂 Age',29,63,30)
    loan_paid_days_y = st.number_input("📆 Previous Loan Paid (Days)",1,400,1)
with col4:
    bank_account_type = st.selectbox("🏦 Bank Account Type", valid_Banktypes)  
    bank_name_clients = st.selectbox("🏦 Bank Name", valid_Bankname) 
employment_status_clients = st.selectbox("💼 Employment Status", valid_employment)
level_of_education_clients = st.selectbox("🎓 Education Level", valid_Educvation)

# ------------------- COMPUTATIONS -------------------
difference_days = termdays_y - loan_paid_days_y
st.write(f"📊 Difference in Days: **{difference_days}**")
    
if difference_days >= 1:
    payment_score = "Low"
elif difference_days == 0:
    payment_score = "Medium"
elif difference_days <= -1:
    payment_score = "High"
else:
    payment_score = "Error"
st.write(f"📊 Payment Score: **{payment_score}**")
    
if loanamount_x <= 10000:
    risk_band = "Low"
elif loanamount_x <= 30000:
    risk_band = "Medium"
elif loanamount_x > 30000:
    risk_band = "High"
else:
    risk_band = "Error"
st.write(f"📊 Risk Band: **{risk_band}**")

# ------------------- PREDICTION -------------------
if st.button('🚀 Can I get a loan?'):
    data = pd.DataFrame({
            "loannumber_x": [loannumber_x],
            "loanamount_x": [loanamount_x],
            "termdays_x": [termdays_x],
            "totaldue_x": [totaldue_x],
            "loanamount_y": [loanamount_y],
            "totaldue_y": [totaldue_y],
            "loannumber_y": [loannumber_y],
            "termdays_y": [termdays_y],
            "age": [age],
            "loan_paid_days_y": [loan_paid_days_y],
            "difference_days": [difference_days],
            "payment_score": [payment_score],
            "bank_account_type": [bank_account_type],
            "bank_name_clients": [bank_name_clients],
            "employment_status_clients": [employment_status_clients],
            "level_of_education_clients": [level_of_education_clients],
            "risk_band": [risk_band]
        })
    
    input_df = pd.DataFrame(data)
    
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0, 1]
    result = "✅ Yes, your loan is approved!" if prediction == 1 else "❌ No, your loan is not approved."

    st.markdown(f'<div class="result-card"><h3>{result}</h3><p>Approval Probability: {probability:.2%}</p></div>', unsafe_allow_html=True)
