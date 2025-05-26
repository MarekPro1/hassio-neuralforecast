from fastapi import FastAPI
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS

app = FastAPI()

@app.post("/forecast")
async def forecast(data: dict):
    df = pd.DataFrame(data['df'])
    df['ds'] = pd.to_datetime(df['ds'])
    nf = NeuralForecast(models=[NBEATS(input_size=24, h=12, max_steps=100)], freq='M')
    nf.fit(df=df)
    forecasts = nf.predict()
    return forecasts.to_dict()