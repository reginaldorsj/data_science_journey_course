import pandas as pd
import pickle as pk
from flask import Flask, request
from preparation import Preparation

model = pk.load(open('./model.pkl','rb'))

app = Flask(__name__)

@app.route("/predict", methods=["GET"])
def predict():
    test_json = request.get_json()
    if test_json:
        if isinstance(test_json,dict):
            df = pd.DataFrame(test_json, index=[0])
        else:
            df = pd.DataFrame(test_json, columns=test_json[0].keys())
    else:
        return dict()

    df_result = Preparation().execute(df)
    pred = model.predict(df_result)
    df["selling_price_pred"] = pred
    return df.to_json(orient="records")

if __name__ == '__main__':
    app.run(host="localhost",port='5000')