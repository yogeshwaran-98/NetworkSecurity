import os,sys
from dotenv import load_dotenv
import pandas as pd
import certifi
import pymongo

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile,Request
from uvicorn import run as app_run
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse

from network_security.pipelines.training_pipeline import TrainingPipeline
from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging
from network_security.utils.main_utils.utils import load_object
from network_security.utils.ml_utils.model.estimator import NetworkModel

load_dotenv()

ca = certifi.where()

mongo_db_url = os.getenv("MONGO_URI")

client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)

templates = Jinja2Templates(directory="./templates")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"],
    allow_credentials = True,
)

@app.get("/")
async def index():
    return RedirectResponse(url = "/docs")

@app.get("/train")
async def train_route():
    try:
        training_pipeline = TrainingPipeline()
        training_pipeline.run_pipeline()
        return Response("Training is successful")
    except Exception as e:
        raise NetworkSecurityException(e,sys)

@app.post("/predict")
async def predict_route(request: Request , file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        if 'Result' in df.columns:
            df = df.drop(columns=['Result'])
        preprocessor = load_object("final_model/preprocessor.pkl")
        final_model=load_object("final_model/model.pkl")
        network_model = NetworkModel(preprocessor = preprocessor ,model = final_model)

        y_pred = network_model.predict(df)
        df['predicted_col'] = y_pred
        df.to_csv('prediction_output/output.csv')

        #table_html = df.to_html(classes='table table-striped')
        #return templates.TemplateResponse("table.html", {"request": request, "table": table_html})
    except Exception as e:
        raise NetworkSecurityException(e,sys)


   
if __name__=="__main__":
    app_run(app,host="localhost",port=8000)

    

        