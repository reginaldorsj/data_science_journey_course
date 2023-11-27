import pandas as pd
import pickle as pk
from flask import Flask, request

model = pk.load(open('..//Project 2 - Machine Learning - Recommendation System//model.pkl','rb'))
books_df = pk.load(open('..//Project 2 - Machine Learning - Recommendation System//books_df.pkl','rb'))
df = pk.load(open('..//Project 2 - Machine Learning - Recommendation System//books_user.pkl','rb'))

app = Flask(__name__)
@app.route("/sugest", methods=["GET"])
def sugest():
    test_json = request.get_json()
    if test_json:
        if isinstance(test_json,dict):
            input_df = pd.DataFrame(test_json, index=[0])
        else:
            input_df = pd.DataFrame(test_json, columns=test_json[0].keys())
    else:
        return dict()

    sugested = df.loc[df.index.str.contains(input_df.iloc[0].values[0])].index
    t = []
    u = []
    cont = 0
    for title in sugested:
        pos = df.index.get_loc(title)
        distance, indices = model.kneighbors([df.iloc[pos, :].values], n_neighbors=6)
        for i, b in enumerate(indices.flatten()):
            if len([tit for tit in t if tit==df.index[b]])!=0:
                continue
            u.append(books_df.loc[books_df["title"]==df.index[b],"image"].values[0])
            t.append(df.index[b])

    ret_df = pd.DataFrame({"title":t,"url":u})
    return ret_df.to_json(orient="records")

if __name__ == '__main__':
    app.run(host="localhost",port='5000')