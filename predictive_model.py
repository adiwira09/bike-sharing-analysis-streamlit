import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# import xgboost
from xgboost import XGBRegressor

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
kolom = ['season',	'mnth',	'hr', 'weekday', 'weathersit', 'temp', 'atemp', 'hum', 'cnt']
df_model = df[kolom]
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
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
best_params_ = {'colsample_bytree': 0.9,
                'gamma': 0.2,
                'learning_rate': 0.1,
                'max_depth': 7,
                'min_child_weight': 1,
                'n_estimators': 300,
                'subsample': 0.9}
model = XGBRegressor(**best_params_)
model.fit(X_train, y_train)
import joblib
joblib.dump(model, "models.pkl")