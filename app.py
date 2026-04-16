import pandas as pd  
import streamlit as st  
import joblib  

model=joblib.load('attrition.pkl')
scale=joblib.load('scale.pkl')
le=joblib.load('encoder.pkl')
trained=joblib.load('columns.pkl')

df=pd.read_csv("Attrition\WA_Fn-UseC_-HR-Employee-Attrition.csv")
x=df.drop(columns='Attrition',errors='ignore')
user_input={}
for col in x.columns:
    if x[col].dtype in ['int64','float64']:

       min_val= float(x[col].min())
       max_val=float(x[col].max())
       mean_val=float(x[col].mean())
       if min_val==max_val:
            user_input[col] = min_val
       else:
        user_input[col]=st.sidebar.slider(col,min_val,max_val,mean_val)
    else:    
        user_input[col]=st.sidebar.selectbox(col, x[col].unique())
user_df=pd.DataFrame([user_input])
st.write("### User input:")
st.dataframe(user_df)
if st.button("predict"):
    user_df_encoded = pd.get_dummies(user_df).reindex(columns=trained, fill_value=0)
    scaled=scale.transform(user_df_encoded)    
    scaled_model=model.predict(scaled)[0]
    encoded_model=le.inverse_transform([scaled_model])[0]    
    st.success(f"predicted Attrition:{encoded_model}")
st.write("### Output Visualization")
st.bar_chart(user_df)    
if st.button("Restart"):
    st.rerun()    
