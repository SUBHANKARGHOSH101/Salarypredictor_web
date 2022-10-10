import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

data=pd.read_csv("salary.csv")
X = data.iloc[:,0:1].values
y = data.iloc[:,1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model=LinearRegression()
model.fit(X_train, y_train)
y_pred =model.predict(X_test)
accuracy = int(r2_score(y_test,y_pred)*100)
st.title("EXPECTED SALARY PREDICTION")
experience=st.number_input("Enter Years of experience :")

if st.button("SUBMIT"):

    st.markdown("<h1 style='color: green;'><b>Expected Salary :</b></h1>",unsafe_allow_html=True)
    output=int(model.predict(np.array(experience).reshape(1,-1))[0])
    st.markdown(f'<h4 style="color: green;">{output}</h4>',unsafe_allow_html=True)
    st.markdown("<h1 style='color: green;'><b>Accuracy :</b></h1>",unsafe_allow_html=True)
    st.markdown(f'<h4 style="color: green;">{accuracy}%</h4>',unsafe_allow_html=True)
