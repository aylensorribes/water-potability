from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

import pickle
import pandas as pd
from fastapi.encoders import jsonable_encoder


app = FastAPI()

MODEL_PATH = '/home/aylen/Desktop/DataScience_Bootcamp/semana6/water_potability/model/rf.pkl'
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

COLUMNS_OHE='/home/aylen/Desktop/DataScience_Bootcamp/semana6/water_potability/model/categories_ohe.pkl'
with open(COLUMNS_OHE, "rb") as f:
    columns_ohe = pickle.load(f)

BIN_PH='/home/aylen/Desktop/DataScience_Bootcamp/semana6/water_potability/model/saved_bins_ph.pkl'
with open(BIN_PH, "rb") as f:
    bins_ph = pickle.load(f)
    
BIN_SULF='/home/aylen/Desktop/DataScience_Bootcamp/semana6/water_potability/model/saved_bins_Sulf.pkl'
with open(BIN_SULF, "rb") as f:
    bins_sulf = pickle.load(f)
    
BIN_TRIHA='/home/aylen/Desktop/DataScience_Bootcamp/semana6/water_potability/model/saved_bins_Triha.pkl'
with open(BIN_TRIHA, "rb") as f:
    bins_triha = pickle.load(f)
    

class Answer(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity:float



@app.get("/")
async def root():
    return {"message": "Water Potability"}


@app.post("/prediction")
def predict_water_potability(answer: Answer):

    answer_dict = jsonable_encoder(answer) #convertimos el json de data a diccionario
    
    for key, value in answer_dict.items():
        answer_dict[key] = [value]
    
    single_instance = pd.DataFrame.from_dict(answer_dict) #creamos un df con el diccionario de los datos, ya que nuestro modelo entrenado recibe dataframes como entrada.
    
    # Manejar puntos de corte o bins
    col_names=['ph','Trihalomethanes','Sulfate']
    binsname=[bins_ph,bins_triha,bins_sulf]
    for colname,saved_bin in zip(col_names,binsname):
        single_instance[f"{colname}_cat"]=pd.cut(single_instance[colname],bins=saved_bin,include_lowest=True)
        single_instance[f"{colname}_cat"]=single_instance[f"{colname}_cat"].cat.add_categories('desconocido')
        single_instance[f"{colname}_cat"]=single_instance[f"{colname}_cat"].fillna(value='desconocido')
        
    single_instance=single_instance.drop(['ph','Trihalomethanes','Sulfate'],axis=1)
    # One hot encoding
    single_instance_ohe = pd.get_dummies(single_instance,dtype=int).reindex(columns = columns_ohe).fillna(0)

    prediction = model.predict(single_instance_ohe)
    # Cast numpy.int64 to just a int
    # pred_probs=model.predict_proba(single_instance_ohe) #matriz de probabilidades: para cada dato del dataset de entrenamiento (row)  tenemos la probabilidad de que el agua no sea potable (col1) y prob de que el agua sea potable (col2)
    # y_prob_tr=pred_probs[:,1]
    # score = y_prob_tr[0]
    # if score>0.39:
    #     response = {"Potable_water": 'Yes'}
    # else:
    #     response = {"Potable_water": 'No'}
    score = int(prediction[0])
    
    response = {"score": score}
    
    return response


# Corre en http://127.0.0.1:8000 o http://0.0.0.0:8000
if __name__ == '__main__':

    # 0.0.0.0 o 127.0.0.1
    uvicorn.run(app, host='127.0.0.1', port=8000)