import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def loadData():
    flights = pd.read_csv('/home/vaishali/projects/python_proj/Flight/flights.csv')
    airport = pd.read_csv('/home/vaishali/projects/python_proj/Flight/airports.csv')

    variables_to_remove=["YEAR","FLIGHT_NUMBER","TAIL_NUMBER","DEPARTURE_TIME","TAXI_OUT","WHEELS_OFF","ELAPSED_TIME","AIR_TIME","WHEELS_ON","TAXI_IN","ARRIVAL_TIME","DIVERTED","CANCELLED","CANCELLATION_REASON","AIR_SYSTEM_DELAY", "SECURITY_DELAY","AIRLINE_DELAY","LATE_AIRCRAFT_DELAY","WEATHER_DELAY","SCHEDULED_TIME","SCHEDULED_ARRIVAL"]
    flights.drop(variables_to_remove,axis=1,inplace= True)

    flights.loc[~flights.ORIGIN_AIRPORT.isin(airport.IATA_CODE.values),'ORIGIN_AIRPORT']='OTHER'
    flights.loc[~flights.DESTINATION_AIRPORT.isin(airport.IATA_CODE.values),'DESTINATION_AIRPORT']='OTHER'

    flights=flights.dropna()

    df=pd.DataFrame(flights)
    df['DAY_OF_WEEK']= df['DAY_OF_WEEK'].apply(str)
    df["DAY_OF_WEEK"].replace({"1":"SUNDAY", "2": "MONDAY", "3": "TUESDAY", "4":"WEDNESDAY", "5":"THURSDAY", "6":"FRIDAY", "7":"SATURDAY"},inplace=True)

    dums = ['AIRLINE','ORIGIN_AIRPORT','DESTINATION_AIRPORT','DAY_OF_WEEK']
    df_cat=pd.get_dummies(df[dums],drop_first=True)

    var_to_remove=["DAY_OF_WEEK","AIRLINE","ORIGIN_AIRPORT","DESTINATION_AIRPORT"]
    df.drop(var_to_remove,axis=1,inplace=True)

    data=pd.concat([df,df_cat],axis=1)
    final_data = data.sample(n=60000)
    return final_data

def preprocessing(final_data):
    X=final_data.drop("DEPARTURE_DELAY",axis=1)
    Y=final_data.DEPARTURE_DELAY
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    return X_train,y_train,X

def rfg(X_train,y_train):
    reg_rf = RandomForestRegressor()
    reg_rf.fit(X_train,y_train)
    return reg_rf

def accept_data():
    month = st.number_input("Enter month ",min_value=1,max_value=12)
    day = st.number_input("Enter day",min_value=1,max_value=31)
    sch_dept = st.number_input("Enter scheduled departure")
    distance = st.number_input("Enter distance in miles")
    arrival_delay = st.number_input("Enter arrival delay (Enter negative value if arrival is delayed else enter positive value)")
    airline = st.text_area("Enter airline code in place of XX","AIRLINE_XX")
    origin = st.text_area("Enter origin airport code in place of XXX","ORIGIN_AIRPORT_XX")
    destination = st.text_area("Enter destination airport code in place of XXX","DESTINATION_AIRPORT_XX")
    day_of_week = st.text_area("Enter day of week in place of XX","XXDAY")
    return month,day,sch_dept,distance,arrival_delay,airline,origin,destination,day_of_week

def prediction(X,month, day,sch_dept,distance,arrival_delay,airline,origin,destination,day_of_week,reg_rf):
    AIRLINE_index = np.where(X.columns==airline)
    ORIGIN_index = np.where(X.columns==origin)
    DESTINATION_index = np.where(X.columns==destination)
    DAY_OF_WEEK_index = np.where(X.columns==day_of_week)
    x= np.zeros(len(X.columns))
    x[0] = month
    x[1] = day
    x[2] = sch_dept
    x[3] = distance
    x[4] = arrival_delay
    x[AIRLINE_index] = 1
    x[ORIGIN_index] = 1
    x[DESTINATION_index] = 1
    x[DAY_OF_WEEK_index] = 1
    return reg_rf.predict([x])[0]

def main():
    st.title("Flight Delay Prediction")
    st.subheader("Prediction using Machine Learning Algorithm")

    choice= st.selectbox("Choose Machine Learning Model",["None","Random Forest Regressor"])
    if choice=="Random Forest Regressor":
        final_data = loadData()
        X_train,y_train,X= preprocessing(final_data)
        reg_rf = rfg(X_train,y_train)

        month,day,sch_dept,distance,arrival_delay,airline,origin,destination,day_of_week = accept_data()

        if st.button("Predict using Random Forest Regressor"):
            res= prediction(X,month,day,sch_dept,distance,arrival_delay,airline,origin,destination,day_of_week,reg_rf)
            if(res>=0):
                text1= "Flight is not delayed. It will depart for next flight at scheduled time"
                st.write(text1)
            elif(res>= -15):
                text2= "Flight is only delayed by "+str(abs(res))+". Delays upto 15 minutes are considered as not delay. FLIGHT IS NOT DELAYED"
                st.write(text2)
            else:
                text3= "Flight is delayed by "+str(res)+". Delays by more than 15 minutes are considered to be actual delays. FLIGHT IS DELAY"
                st.write(text3)


if __name__=='__main__':
    main()
