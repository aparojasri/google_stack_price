
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI()
model = load_model("lstm_model.h5")

class InputData(BaseModel):
    features: list

@app.post("/predict")
def predict(data: InputData):
    x_input = np.array(data.features).reshape(1, *np.array(data.features).shape)
    prediction = model.predict(x_input)
    return {"prediction": prediction.tolist()}
