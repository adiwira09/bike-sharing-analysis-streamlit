import pandas as pd
import streamlit as st
import plotly.express as px
# from st_pages import show_pages_from_config

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

# # Visual 2
fig2 = px.bar(x=['Casual', 'Registered'], y=[df['casual'].sum(), df['registered'].sum()],
             color=['Casual', 'Registered'], color_discrete_map={'Casual': 'blue', 'Registered': 'green'},
             labels={'y': 'Count', 'x':''})
fig2.update_traces(showlegend=False)
fig2.update_layout(title='Number of Registered and Casual<br>Bike-share Rides')

left_column, right_column = st.columns((2,1))
left_column.plotly_chart(fig1, use_container_width=True)
right_column.plotly_chart(fig2, use_container_width=True)




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

#######
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

# Show the plot
st.plotly_chart(fig7, use_container_width=True)
