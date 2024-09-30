import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# read data
df = pd.read_csv("C:/Users/boran/Desktop/Dosyalar/python/sleep_detection/blink_data.csv")


X = df[['RatioAvg']]  
y = df['BlinkCount'] 

# train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

with open("C:/Users/boran/Desktop/Dosyalar/python/sleep_detection/model/random_forest_model.pkl", "wb") as f:
    pickle.dump(model, f)
