from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
import os
from src.config import RAW_DATA_PATH,PROCESSED_DATA_DIR,MODEL_DIR,MODEL_PATH
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
import pickle

def train_model():
    '''
    loading a train dataset and training a model
    '''

    try:
        logging.info("starting a training the model")

        X_train=pd.read_csv(os.path.join(PROCESSED_DATA_DIR,'X_train.csv'))
        X_test=pd.read_csv(os.path.join(PROCESSED_DATA_DIR,'X_test.csv'))
        y_train=pd.read_csv(os.path.join(PROCESSED_DATA_DIR,'y_train.csv')).squeeze()
        y_test=pd.read_csv(os.path.join(PROCESSED_DATA_DIR,'y_test.csv')).squeeze()
        logging.info(y_train.shape)
        logging.info("Traing a model with RandomForest Algo")

        model=RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=1000,
            criterion='gini',
            class_weight='balanced'
            )
        
        model.fit(X_train,y_train)

        y_pred=model.predict(X_test)

        acc=accuracy_score(y_pred,y_test)

        logging.info(f"accuracy of model is {acc}")
        logging.info(f"classification report is :{classification_report(y_pred,y_test)}")
        

        with open(MODEL_PATH,'wb') as f:
            pickle.dump(model,f)
            logging.info("model loaded successfully")


    except Exception as e:
        logging.info("Error in the training a model")
        CustomException(e,sys)

if __name__=="__main__":
    train_model()
    



