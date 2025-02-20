from fastapi import FastAPI # type: ignore
from pydantic import BaseModel # type: ignore
import joblib # type: ignore
import numpy as np # type: ignore

# Load the trained model
model = joblib.load("model/iris_model.pkl")

# Initialize FastAPI
app = FastAPI(title="Iris Classifier API")

# Define request model
class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Define endpoint
@app.post("/predict")
def predict(iris: IrisRequest):
    data = np.array([[iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]])
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}
