import pandas as pd
import streamlit as st
import joblib
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def tren(dataset):
    data1 = dataset[['dteday', 'casual', 'registered', 'cnt']]
    # Mengubah kolom 'dteday' menjadi format tahun-bulan
    data1['dteday'] = pd.to_datetime(data1['dteday']).dt.to_period('M')
    data1['dteday'] = data1['dteday'].astype('str')

    # Menghitung total bulanan untuk kolom 'cnt'
    data1 = data1.groupby('dteday').agg({
        'casual': 'sum',
        'registered': 'sum',
        'cnt': 'sum'
    }).reset_index()
    return data1

def count_by_season(dataset):
    season = dataset.groupby('season').agg({
    'casual':'sum',
    'registered':'sum',
    'cnt':'sum'}).reset_index()
    return season

def count_by_weathersit(dataset):
    weathersit = dataset.groupby('weathersit').agg({
    'casual':'sum',
    'registered':'sum',
    'cnt':'sum'}).reset_index()
    return weathersit

def count_by_day(dataset):
    day = dataset.groupby('weekday').agg({
    'casual':'sum',
    'registered':'sum',
    'cnt':'sum'})
    day = day.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    return day

def count_by_hour(dataset):
    hour = dataset.groupby('hr').agg({
    'casual':'sum',
    'registered':'sum',
    'cnt':'sum'})
    return hour

# import dataset
df = pd.read_csv('dataset/hour.csv')

# handling dataset
df['dteday'] = pd.to_datetime(df['dteday'])
df['mnth'] = df['dteday'].dt.month_name()
df['weekday'] = df['dteday'].dt.day_name()
df['yr'] = df['dteday'].dt.year

df['temp'] = df['temp'] * 41
df['atemp'] = df['atemp'] * 50
df['hum'] = df['hum'] * 100
df['windspeed'] = df['windspeed'] * 67

mapping_season = {
    1: 'Springer',
    2: 'Summer',
    3: 'Fall',
    4: 'Winter'
}

df['season'] = df['season'].map(mapping_season)

# menyiapkan dataframe
data_tren = tren(df)
data_season = count_by_season(df)
data_weathersit = count_by_weathersit(df)
data_day = count_by_day(df)
data_hour = count_by_hour(df)

st.set_page_config(page_title="Bike-sharing Rides Dashboard",
                   layout="wide")

with st.sidebar:
    st.markdown(
        "<div style='display: flex; justify-content: center;'><h1>Nugroho Adi Wirapratama</h1></div>",
        unsafe_allow_html=True
    )

    # Menambahkan foto
    st.markdown(
        "<div style='display: flex; justify-content: center;'>"
        "<img src='https://github.com/adiwira09/bike-sharing-analysis-streamlit/blob/main/photo.png?raw=true' width='180' style='border-radius: 50%;'>"
        "</div>",
        unsafe_allow_html=True
    )
    st.sidebar.header("Click for Profile:")

    col1,col2,col3,col4,col5 = st.sidebar.columns(5)
    with col1:
        st.markdown(
            "<div style='display: flex;'>"
            "<a href='https://www.linkedin.com/in/nug-adiwira/' target='_blank'>"
            "<img src='https://upload.wikimedia.org/wikipedia/commons/thumb/8/81/LinkedIn_icon.svg/2048px-LinkedIn_icon.svg.png' width='50'>"
            "<a/>"
            "</div>",
            unsafe_allow_html=True
        )
    with col2: 
        st.markdown(
            "<div style='display: flex;'>"
            "<a href='https://www.github.com/adiwira09/' target='_blank'>"
            "<img src='https://cdn-icons-png.flaticon.com/512/25/25231.png' width='50'>"
            "<a/>"
            "</div>",
            unsafe_allow_html=True
            )
    # show_pages_from_config()
dashboard , model_report, prediction  = st.tabs(['Dashboard Analysis 🔎','Model Report' , 'Prediction 📈'])
with dashboard:
    st.title(":bicyclist: Bike-sharing Rides Dashboard :bicyclist:")
    col1, col2, col3 = st.columns(3)
    with col1:
        total_all_rides = df['cnt'].sum()
        st.metric("Total Rides", value=total_all_rides)
    with col2:
        total_casual_rides = df['casual'].sum()
        st.metric("Total Casual Rides", value=total_casual_rides)
    with col3:
        total_registered_rides = df['registered'].sum()
        st.metric("Total Registered Rides", value=total_registered_rides)
##################################################################################################
    fig1 = px.bar(data_tren, 
                x='dteday', 
                y=['casual', 'registered'],
                labels={'value': 'Count', 'dteday':'Date'},
                color_discrete_sequence=["blue", "green"],
                title='Monthly Number of Casual and Registered Users (2 years)',
                template='plotly_white',
                barmode='group')

    # Menambahkan line plot untuk 'cnt'
    fig1.add_scatter(x=data_tren['dteday'], y=data_tren['cnt'], mode='lines+markers', name='Cnt', line=dict(color='red'))
    fig1.update_layout(legend=dict(orientation="h", yanchor="bottom", y=0.87, xanchor="right", x=0.4))

    fig2 = px.bar(x=['Casual', 'Registered'], y=[df['casual'].sum(), df['registered'].sum()],
                 color=['Casual', 'Registered'], color_discrete_map={'Casual': 'blue', 'Registered': 'green'},
                 labels={'y': 'Count', 'x':''})
    fig2.update_traces(showlegend=False)
    fig2.update_layout(title='Number of Registered and Casual<br>Bike-share Rides')

    left_column, right_column = st.columns((2,1))
    left_column.plotly_chart(fig1, use_container_width=True)
    right_column.plotly_chart(fig2, use_container_width=True)
##################################################################################################    
    fig3 = px.bar(data_day, 
                  x=data_day.index, 
                  y=['casual', 'registered'],
                 color_discrete_sequence=["blue", "green"],
                 labels={'value': 'Count', 'weekday': 'Day'},
                 title='Casual vs Registered Count by Weekday',
                 template='plotly_white',
                 barmode='group')

    # Line plot untuk 'cnt'
    fig3.add_scatter(x=data_day.index, y=data_day['cnt'], mode='lines+markers', name='Cnt', line=dict(color='red'))
    st.plotly_chart(fig3, use_container_width=True)
##################################################################################################
    fig4 = px.bar(data_hour, 
                  x=data_hour.index, 
                  y=['casual', 'registered'],
                 color_discrete_sequence=["blue", "green"],
                 labels={'value': 'Count', 'hr': 'Hour'},
                 title='Casual vs Registered Count by Hour',
                 template='plotly_white',
                 barmode='group')

    # Line plot untuk 'cnt'
    fig4.add_scatter(x=data_hour.index, y=data_hour['cnt'], mode='lines+markers', name='Cnt', line=dict(color='red'))
    fig4.update_layout(xaxis=dict(tickmode='linear', dtick=1))
    st.plotly_chart(fig4, use_container_width=True)
##################################################################################################
    fig5 = px.pie(data_season, values='cnt', names='season',
                    title='Percentage of Bike-share Users by Season',
                    color_discrete_sequence=['gold', 'tomato', 'cornflowerblue', 'orchid'],
                    labels={'cnt': 'Percentage'})
    fig5.update_layout(legend=dict(orientation="v", yanchor="bottom", y=0.8, xanchor="right", x=0.2))
    fig6 = px.bar(data_season, 
                  x=data_season['season'], 
                  y=['casual', 'registered'],
                  color_discrete_sequence=["blue", "green"],
                  labels={'value': 'Count', 'season': 'Season'},
                  title='Casual vs Registered Count by Season',
                  template='plotly_white',
                  barmode='group')


    left_column, right_column = st.columns((1.5,2))
    left_column.plotly_chart(fig5, use_container_width=True)
    right_column.plotly_chart(fig6, use_container_width=True)

##################################################################################################
    weather_descriptions = {
        "1": "Clear, Few clouds, Partly cloudy, Partly cloudy",
        "2": "Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist",
        "3": "Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds",
        "4": "Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog"
    }
    data_weathersit['weathersit'] = data_weathersit['weathersit'].astype(str)
    data_weathersit['weathersit_legend'] = data_weathersit['weathersit'].map(weather_descriptions)
    color_map = {
        'Clear, Few clouds, Partly cloudy, Partly cloudy': 'blue',
        'Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist': 'green',
        'Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds': 'orange',
        'Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog': 'red'
    }
    fig7 = px.bar(data_weathersit, x='weathersit', y='cnt',
                 color='weathersit_legend',
                 color_discrete_map=color_map,
                 labels={'cnt': 'Count', 'weathersit':'Weathersit'},
                 title='Weather Situation Count',
                 text='cnt')
    st.plotly_chart(fig7, use_container_width=True)

##################################################################################################
df_copy = df.copy()
kolom = ['season',	'mnth',	'hr', 'weekday', 'weathersit', 'temp', 'atemp', 'hum', 'cnt']
df_model = df_copy[kolom]
mnth_encoder = {
    "January": 0,
    "February": 1,
    "March": 2,
    "April": 3,
    "May": 4,
    "June": 5,
    "July": 6,
    "August": 7,
    "September": 8,
    "October": 9,
    "November": 10,
    "December": 11
}

weekday_encoder = {
    "Sunday": 0,
    "Monday": 1,
    "Tuesday": 2,
    "Wednesday": 3,
    "Thursday": 4,
    "Friday": 5,
    "Saturday": 6,
}
df_model["mnth"] = df_model["mnth"].apply(lambda x: mnth_encoder[x])
df_model["weekday"] = df_model["weekday"].apply(lambda x: weekday_encoder[x])
df_model = pd.get_dummies(df_model, prefix='season')
df_model = pd.get_dummies(df_model, columns=['weathersit'], prefix='weathersit')

# normalisasi kolom
scaler = MinMaxScaler()
# fitur yang akan dinormalisasi
features_to_normalize = ['temp', 'atemp', 'hum']
# Normalisasi fitur
df_model[features_to_normalize] = scaler.fit_transform(df_model[features_to_normalize])

# memisahkan atribut fitur dan target
X = df_model.drop('cnt', axis=1)
y = df_model['cnt']

# membagi dataset train dan test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model = joblib.load("models.pkl")
y_pred = model.predict(X_test)
y_pred = np.ceil(y_pred).astype(int) # membulatkan hasil nya menjadi ke atas

# Menghitung metrik evaluasi
mae_xgb = mean_absolute_error(y_test, y_pred)
mse_xgb = mean_squared_error(y_test, y_pred)
rmse_xgb = np.sqrt(mse_xgb)
r2_xgb = r2_score(y_test, y_pred)

with model_report:
    st.subheader("Model Algoritma: XBG Regressor")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Absolute Error (MAE): ", value=round(mae_xgb, 3))
    with col2:
        st.metric("Mean Squared Error (MSE): ", value=round(mse_xgb, 3))
    with col3:
        st.metric("Root Mean Squared Error (RMSE): ", value=round(rmse_xgb, 3))
    with col4:
        st.metric("R-squared (R2) XGBoost: ", value=round(r2_xgb, 3))


    # Perbandingan Data Aktual dengan Data Prediksi
    y_predict = model.predict(X)
    y_predict = np.ceil(y_predict).astype(int)
    data = pd.DataFrame({'Actual': y, 'Predicted': y_predict})

    fig8 = px.scatter(data, x=data.index, y='Actual', labels={'x': 'index', 'y': 'cnt'}, title='Data Actual')
    fig8.update_traces(marker=dict(color='blue'))

    fig9 = px.scatter(data, x=data.index, y='Predicted', labels={'x': 'index', 'y': 'cnt'}, title='Data Predicted')
    fig9.update_traces(marker=dict(color='red'))

    col1,col2 = st.columns([1,4])
    with col1:
        correlations = df.drop(columns=["casual", "registered", "cnt"])
        correlations = correlations.select_dtypes(include='number').corrwith(df['cnt'])
        correlation_df = pd.DataFrame(correlations, columns=["cnt"]).sort_values(by='cnt', ascending=False)
        plt.figure(figsize=(1, 5))
        sns.heatmap(correlation_df, annot=True, cmap='RdYlGn', fmt=".3f")
        title = plt.title("Korelasi cnt")
        title.set_weight('bold')
        st.pyplot(plt)
    with col2:
        feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_,X_train.columns)), columns=['Value','Feature'])
        plt.figure(figsize=(10, 5))
        sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
        plt.title(label= 'XGBRegressor Feature Importance', size = 20)
        plt.tight_layout()
        st.pyplot(plt)

    st.subheader("Actual Data VS Predicted Data for the Initial Dataset")
    col1,col2 = st.columns(2)
    col1.plotly_chart(fig8, use_container_width=True)
    col2.plotly_chart(fig9, use_container_width=True)

with prediction:
    col1,col2,col3 = st.columns(3)
    with col1:
        min_date = pd.Timestamp("2011-01-01")
        max_date = pd.Timestamp("2100-12-31")
        date = st.date_input('Date' , min_value=min_date , max_value=max_date , value=pd.Timestamp.today())
        weekday = date.day
        month_name = date.strftime("%B")
        day_name = date.strftime("%A")
    with col2:
        weathersit_list = [
        'Clear, Few clouds, Partly cloudy, Partly cloudy',
        'Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist',
        'Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds',
        'Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog']
        weathersit = st.selectbox ('Select Weathersit :' , weathersit_list)
    with col3:
        hour_list = list(range(24))
        hour = st.selectbox("Select Hour", hour_list)

    col1,col2,col3 = st.columns(3)
    with col1:
        temp = st.slider("Temp", min(df['temp']), max(df['temp']))
    with col2:
        atemp = st.slider("Atemp", min(df['atemp']), max(df['atemp']))
    with col3:
        hum = st.slider("Humidity", min(df['hum']), max(df['hum']))

    # SEASON
    # Spring = 1 Maret - 31 Mei
    # Summer = 1 Juni - 31 Agustus
    # Fall = 1 September - 30 November
    # Winter = 1 Desember - 30 Februari

    if date.month in range(3, 6):
        season = 'Springer'
    elif date.month in range(6, 9):
        season = 'Summer'
    elif date.month in range(9, 12):
        season = 'Fall'
    else:
        season = 'Winter'
        
    weathersit_mapping = {
        'Clear, Few clouds, Partly cloudy, Partly cloudy': 1,
        'Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist': 2,
        'Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds': 3,
        'Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog': 4}
    
    df_user = pd.DataFrame({
        'mnth':[month_name],
        'hr':[hour],
        'weekday':[day_name],
        'temp':[temp],
	    'atemp':[atemp],
        'hum':[hum],
        'season':[season],
    	'weathersit':[weathersit]
    })
    
    df_predict = df_user.copy()
    df_predict['mnth'] = df_predict['mnth'].apply(lambda x: mnth_encoder[x])
    df_predict['weekday'] = df_predict["weekday"].apply(lambda x: weekday_encoder[x])
    df_predict['weathersit'] = df_predict["weathersit"].apply(lambda x: weathersit_mapping[x])
    
    df_predict['season_Fall'] = 0 
    df_predict['season_Springer'] = 0 
    df_predict['season_Summer'] = 0 
    df_predict['season_Winter'] = 0 

    df_predict['weathersit_1'] = 0
    df_predict['weathersit_2'] = 0
    df_predict['weathersit_3'] = 0
    df_predict['weathersit_4'] = 0

    df_predict['season_Fall'] = 1 if df_predict['season'][0] == 'Fall' else 0
    df_predict['season_Springer'] = 1 if df_predict['season'][0] == 'Springer' else 0
    df_predict['season_Summer'] = 1 if df_predict['season'][0] == 'Summer' else 0
    df_predict['season_Winter'] = 1 if df_predict['season'][0] == 'Winter' else 0

    df_predict['weathersit_1'] = 1 if df_predict['weathersit'][0] == 1 else 0
    df_predict['weathersit_2'] = 1 if df_predict['weathersit'][0] == 2 else 0
    df_predict['weathersit_3'] = 1 if df_predict['weathersit'][0] == 3 else 0
    df_predict['weathersit_4'] = 1 if df_predict['weathersit'][0] == 4 else 0

    # Menghapus kolom yang tidak diperlukan
    df_predict = df_predict.drop(['season', 'weathersit'], axis=1)

    result = model.predict(df_predict)

    if st.button("Predict"):
        st.header(f"Result: {int(result)} Rides")
