import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
import pandas as pd
from datetime import datetime
import pickle
import numpy as np

# create dataframe
df = pd.DataFrame(columns=["Time", "BlinkCount", "RatioAvg"])

def log_blink(blink_counter, ratio_avg):
    current_time = datetime.now().strftime("%H:%M:%S")
    new_row = pd.DataFrame({"Time": [current_time], "BlinkCount": [blink_counter], "RatioAvg": [ratio_avg]})
    global df
    df = pd.concat([df, new_row], ignore_index=True)

def interpret_prediction(prediction):
    if prediction == 0:
        return "Not Sleeping"
    elif prediction == 1:
        return "Light Sleep"
    elif prediction == 2:
        return "Deep Sleep"
    elif prediction == 3:
        return "Very Deep Sleep"
    elif prediction == 4:
        return "Unknown"
    else:
        return "Invalid Prediction"

def predict_sleep(ratioAvg):
    # upload model
    with open("C:/Users/boran/Desktop/Dosyalar/python/sleep_detection/model/random_forest_model.pkl", "rb") as f:
        model = pickle.load(f)
    # prediction
    prediction = model.predict([[ratioAvg]])[0]  
    interpreted_result = interpret_prediction(prediction)  
    return interpreted_result

cap = cv2.VideoCapture(0)
detector = FaceMeshDetector()
plotY = LivePlot(540,360,[10,60])

idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 24]
color = (0,0,255)
counter = 0
blickCounter = 0  
ratioList = []

BLINK_THRESHOLD = 30  
BLINK_DURATION = 10  

while True:
    success, img = cap.read()
    img, faces = detector.findFaceMesh(img, draw=False)
    
    if faces:
        face = faces[0]
        for id in idList:
            cv2.circle(img, face[id], 3, color, cv2.FILLED)

        leftUp = face[159]
        leftDown = face[23]
        leftLeft = face[130]
        leftRight = face[243]

        lengthVer, _ = detector.findDistance(leftUp, leftDown)
        lengthHor, _=  detector.findDistance(leftLeft, leftRight)

        cv2.line(img, leftUp, leftDown, (0,255,0),2)
        cv2.line(img, leftLeft, leftRight, (0,255,0),2)

        ratio = int((lengthVer/lengthHor)*100)
        ratioList.append(ratio)
        if len(ratioList) > 3:
            ratioList.pop(0)
        ratioAvg = sum(ratioList) / len(ratioList)

        if ratioAvg < BLINK_THRESHOLD:
            if counter == 0:
                counter = 1
                blickCounter += 1  
                color = (0,255,0)
                log_blink(blickCounter, ratioAvg)
        else:
            if counter > 0:
                counter += 1
                if counter > BLINK_DURATION:
                    counter = 0
                    color = (0,0,255)

        cvzone.putTextRect(img, f'Blink Count: {blickCounter}', (50,100), colorR=color)  
        cvzone.putTextRect(img, f'Sleep Prediction: {predict_sleep(ratioAvg)}', (50,150), colorR=color)

        imgPlot = plotY.update(ratioAvg, color)
        img = cv2.resize(img, (640,360))
        imgStack = cvzone.stackImages([img, imgPlot], 2,1)

    cv2.imshow("img", imgStack)
    cv2.waitKey(25)

    df.to_csv('C:/Users/boran/Desktop/Dosyalar/python/sleep_detection/blink_data.csv', index=False)
