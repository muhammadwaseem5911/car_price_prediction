import streamlit as st
st.header(":red[**Car**]  :orange[Price]  :blue[**Prediction**]")
st.write('This is a project related to car price prediction in which you can predict price by giving some input.')
st.write('The dataset that is used to train a model is also available in :blue[kaggle].')
import pandas as pd
df=pd.read_csv('car_price.csv')
st.dataframe(df.head())
st.write('If you want to download this :green[dataset] :blue-background[press] :arrow_down_small:')
@st.cache_data
def convert_for_download(df):
    return df.to_csv().encode("utf-8")

# df = get_data()
csv = convert_for_download(df)
st.download_button(
    label="Download CSV",
    data=csv,
    file_name="data.csv",
    mime="text/csv",
    icon=":material/download:",
)
col1, col2 = st.columns(2)

fuel_type = col1.selectbox("Fueltype?",["Diesel","Petrol","CNG", "LPG" , "Electric"])
transmission_type = col1.selectbox("transmission_type?",["Manual","Automatic"])

engine = col2.slider("engine" ,500, 8000, 100)
seats = col2.selectbox("seats?",[4,5,6,7,8,9,10])
encode_dict = {
    "fuel_type" : {"Diesel":1,"Petrol":2, "CNG":3, "LPG":4, "Electric":5 },
    "transmission_type" : {"Manual":1, "Automatic":2}
}
import pickle as pkl
def model_pred(fuel_type,transmission_type,engine,seats):
    with open ("car_model.pkl",'rb') as file :
        reg_model = pkl.load(file)
        input_features=[[2018,27000, fuel_type, transmission_type,20,engine, 15, seats]]
        return reg_model.predict(input_features)

if st.button("Predict price") :    
    st.spinner("the program is loading")
    fuel_type = encode_dict["fuel_type"][fuel_type]
    transmission_type = encode_dict["transmission_type"][transmission_type]
    price = model_pred(fuel_type,transmission_type,engine,seats)
    st.success(f"the price predicted for the car is {price[0].round(2)} lakh rupees")