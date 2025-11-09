from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
import os
from src.config import RANDOM_STATE,RAW_DATA_PATH,PROCESSED_DATA_DIR,TEST_SIZE


def load_data():
    '''
    Load a dataset and return as Pandas dataframe
    '''

    try:
        logging.info("loading a raw data from :%s",RAW_DATA_PATH)
        df=pd.read_csv(RAW_DATA_PATH)
        logging.info("loaded a data from sucessfully:%s",df.shape)
        return df


    except Exception as e:
        raise CustomException(e,sys)


def preprocess_data(df):
    '''
    processing a data: handling a missing values,duplicates etc
    '''
    try:
        logging.info("starting a data processing ")

        df.drop_duplicates(inplace=True)

        missing=df.isnull().sum().sum()
        if missing>0:
            logging.warning("found a missing values filling with zero")
            df.fillna(0,inplace=True)
        else:
            logging.info("No missing values found")

        logging.info("preprocessing is done")
        return df
    except Exception as e:
        CustomException(e,sys)

def split_and_save(df):
    '''
    split a dataset into train and test dataset
    '''

    try:
        logging.info("split a dataset into train and test")
        os.makedirs(PROCESSED_DATA_DIR,exist_ok=True)
        X=df.drop('Class',axis=1)
        y=df['Class']

        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=TEST_SIZE,random_state=RANDOM_STATE)

        X_train.to_csv(os.path.join(PROCESSED_DATA_DIR,'X_train.csv'),index=False)
        X_test.to_csv(os.path.join(PROCESSED_DATA_DIR,'X_test.csv'),index=False)
        y_train.to_csv(os.path.join(PROCESSED_DATA_DIR,'y_train.csv'),index=False)
        y_test.to_csv(os.path.join(PROCESSED_DATA_DIR,'y_test.csv'),index=False)

        logging.info("data split a sucessfully and strored in path")

    except Exception as e:
        CustomException(e,sys)

def main():
    '''
    Main pipeline for data processing and loading
    '''


    try:
        logging.info("data loading and preprocessing is started")
        df=load_data()
        df=preprocess_data(df)
        split_and_save(df)
        logging.info("data loading and preprocessing is done")

    except Exception as e:
        logging.error("error is data loading pipeline:%s",e)
        CustomException(e,sys)




if __name__=="__main__":
    main()





