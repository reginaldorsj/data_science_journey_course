import pickle as pk
import pandas as pd

class Preparation( object ):
    def __init__(self):
        self.ohe = pk.load(open('.//ohe.pkl' ,'rb'))
        self.scaler = pk.load(open('.//scaler.pkl', 'rb'))
    def execute(self, df):
        categorical_cols = [col for col in df.columns if df[col].dtype == "O"]
        df_ohe = self.ohe.transform(df[categorical_cols])
        df_ohe = pd.DataFrame(data=df_ohe, columns=self.ohe.get_feature_names_out(), index=df.index)
        df = df.drop(categorical_cols+["selling_price"], axis=1)
        df = pd.concat(
            [df, df_ohe],
            axis=1)
        df[df.columns] = self.scaler.transform(df)
        return df
       
   
    
    
    