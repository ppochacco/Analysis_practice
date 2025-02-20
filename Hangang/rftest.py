"""
author      : ws
Description : yeoui_test
Date        : 11~
Usage       : 
"""


from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

def average_closest_values(df, pk_name, day, temp_col, curr_temp, target_cols):
    # 주어진 조건을 만족하는 값 필터링
    filtered_df = df[(df['주차장명']==pk_name)&(df['요일']==day)]
    # 입력된 파라미터와의 차이 계산
    differences = np.abs(filtered_df[temp_col] - curr_temp)
    # 차이가 가장 작은 5개의 값 인덱스 선택
    closest_indices = differences.nsmallest(5).index
    
    # 가장 가까운 5개의 값 평균 계산
    average_value = filtered_df.loc[closest_indices, target_cols].mean()
    return average_value

# @app.get("/startup")
def load_model():
    app.model = joblib.load("../Python/Data/rf_yeoui.h5")
    app.yeo_set = pd.read_csv('../Python/Data/1118_yeodata')

@app.get("/predict")
def predict(pk_name: int, day: int, temp: float):
    load_model()
    ex = average_closest_values(app.yeo_set, pk_name, day, '평균기온(°C)', temp, ['주차대수(아침)','주차대수(낮)','주차대수(저녁)']).tolist()
    prediction = app.model.predict([[pk_name, ex[0], ex[1], ex[2], 11, day, temp, 462]])
    return {"prediction": prediction.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
