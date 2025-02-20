import pandas as pd
import streamlit as st
import joblib
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('Loan_Data.csv')

st.markdown("<h1 style = 'color: #DD5746; text-align: center; font-size: 60px; font-family: Monospace'>LOAN DATA PREDICTOR</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #FFC470; text-align: center; font-family: Serif '>Built by DAVID</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html=True)

st.image('pngwing.com (3).png', caption = 'BUILT BY DAVID')

st.markdown("<h2 style = 'color: #F7C566; text-align: center; font-family: montserrat '>Background Of Study</h2>", unsafe_allow_html = True)
st.markdown('In the financial industry, loan approval is a critical process that requires thorough risk assessment. Traditional methods of evaluating loan applications rely on manual review and fixed rule-based systems, which can be time-consuming and prone to human bias. With advancements in data analytics and machine learning, loan prediction apps have emerged as efficient tools to automate and enhance the accuracy of loan approval decisions.')


st.sidebar.image('pngwing.com (4).png')

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.divider()
st.header('Project Data')
st.dataframe(data, use_container_width= True)


app_income = st.sidebar.number_input('Applicant Income', data['ApplicantIncome'].min(), data['ApplicantIncome'].max())
loan_amt = st.sidebar.number_input('Loan Amount', data['LoanAmount'].min(), data['LoanAmount'].max())
coapp_income = st.sidebar.number_input('CoApplicant Income', data['CoapplicantIncome'].min(), data['CoapplicantIncome'].max())
dep = st.sidebar.selectbox('Dependents', data['Dependents'].unique())
prop_area = st.sidebar.selectbox('Property Area', data['Property_Area'].unique())
cred_hist = st.sidebar.number_input('Credit History', data['Credit_History'].min(), data['Credit_History'].max())
loan_amt_term = st.sidebar.number_input('Loan Amount Term', data['Loan_Amount_Term'].min(), data['Loan_Amount_Term'].max())

#users input

input_var = pd.DataFrame()
input_var['ApplicantIncome'] = [app_income]
input_var['LoanAmount'] = [loan_amt]
input_var['CoapplicantIncome'] = [coapp_income]
input_var['Dependents'] = [dep]
input_var['Property_Area'] = [prop_area]
input_var['Credit_History'] = [cred_hist]
input_var['Loan_Amount_Term'] = [loan_amt_term]

st.markdown("<br>", unsafe_allow_html= True)
st.divider()
st.subheader('Users Inputs')
st.dataframe(input_var, use_container_width = True)

app_income = joblib.load('ApplicantIncome_scaler.pkl')
coapp_income = joblib.load('CoapplicantIncome_scaler.pkl')
prop_area = joblib.load('Property_Area_encoder.pkl')

# transform the users input with the imported scalers

input_var['ApplicantIncome'] = app_income.transform(input_var[['ApplicantIncome']])
input_var['CoapplicantIncome'] = coapp_income.transform(input_var[['CoapplicantIncome']])
input_var['Property_Area'] = prop_area.transform(input_var[['Property_Area']])


#bringing in the model for prediction
model = joblib.load('loanmodel.pkl')
predict = model.predict(input_var)

if st.button('Check Your Loan Approval Status'):
    if predict[0] == 0:
        st.error(f"Unfortunately...Your Loan request has been denied")
        st.image('Denied Icon.jpg', width = 300)
    else:
        st.success(f"Congratulations...Your loan request has been approved. Pls come to the office to process your loan")
        st.image('pngwing.com (5).png', width = 300)
        st.balloons()



