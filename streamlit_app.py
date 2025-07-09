import streamlit as st
import pickle
import numpy as np


st.title("🚢 Titanic Survival Predictor")
st.write("Welcome! This app will predict if a passenger survived.")

model_list={
        'Logistic Regression':'Titanic_Classifier_LogReg.pkl',
        'Decision Tree Classifier':'Titanic_Classifier_DecTree.pkl',
        'Random Forest Classifier':'Titanic_Classifier_RanForest.pkl'
    }
select_model = st.selectbox('Choose a model', list(model_list.keys()))
model_file = model_list[select_model]
with open(model_file,'rb') as f:
    model = pickle.load(f)
    
pclass = st.selectbox('Passenger class',[1,2,3])
sex = st.radio('Sex',['male','female'])
age = st.slider('Age',1, 80)
sibsp = st.number_input('Number of Siblings',0, 10)
parch = st.number_input('Parents/Children', 0, 7)
fare = st.number_input('Fare',min_value=0.0, max_value= 513.0, step= 0.01)
embarked = st.selectbox('Embarked',['C','Q','S'])

if st.button('predict'):
    with open('age_scalar','rb') as f:
        age_scalar = pickle.load(f)
    with open('fare_scalar','rb') as f:
        fare_scalar = pickle.load(f)
    with open('embarked_label','rb') as f:    
        embarked_label = pickle.load(f)

    sex = 0 if sex == 'male' else 1
    age = age_scalar.transform([[age]])[0][0]
    fare = fare_scalar.transform([[fare]])[0][0]
    embarked = embarked_label.transform([embarked])[0]

    data = [pclass, sex, age, sibsp, parch, fare, embarked]
    data = np.array(data)
    data = data.reshape(1,-1)

    prediction = model.predict(data)
    if prediction == 1:
        st.success('The passenger survived')
    else:
        st.error('The passenger died')
    probability_predict = model.predict_proba(data)[0][1]
    st.info(f'Model confidence: {probability_predict * 100:.2f}% chance of survival.')
    
    