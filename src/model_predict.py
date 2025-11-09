import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.config import MODEL_PATH,PROCESSED_DATA_DIR
import pickle
import pandas as pd


def load_model():
    '''
    load a model pickel file and predict 
    '''
    try:
        logging.info("loading a model")
        with open(MODEL_PATH,'rb') as f:
            loaded_model=pickle.load(f)

        logging.info("model loaded sucessfuly")
        return loaded_model

    except Exception as e:
        CustomException(e,sys)



def make_predictions():
    
    try:
        model=load_model()
        y_test=pd.read_csv(os.path.join(PROCESSED_DATA_DIR,'X_test.csv')).squeeze()

        y_pred=model.predict(y_test)
        logging.info(f"Prediction completed successfully. Sample output: {y_pred[:5]}")
        return y_pred
    
    except Exception as e:
        CustomException(e,sys)


if __name__=='__main__':
    print(make_predictions())



        





