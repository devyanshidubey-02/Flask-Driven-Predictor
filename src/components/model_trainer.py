import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import customexception
import os
import sys
from src.utils.utils import save_object, evaluate_model
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting dependent and independent variables from train and test data.')

            # Splitting the features (X) and target (y) from the training and testing arrays
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # All columns except the last column as features
                train_array[:, -1],   # Last column as target
                test_array[:, :-1], 
                test_array[:, -1]    
            )

            models = {
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'ElasticNet': ElasticNet()
            }

            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models)

            logging.info(f"Model Performance Report:\n{model_report}")
            print(f"Model Performance Report:\n{model_report}")

            logging.info('\n====================================================================================')

            best_model_score = max(model_report.values()) 
            best_model_name = [k for k, v in model_report.items() if v == best_model_score][0] 

            logging.info(f"Best Model Found: Model Name: {best_model_name}, R2 Score: {best_model_score}")
            print(f"Best Model Found, Model Name: {best_model_name}, R2 Score: {best_model_score}")

            best_model = models[best_model_name]

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            logging.error(f"Exception occurred during model training: {str(e)}")
            raise customexception(e, sys)

if __name__ == "__main__":
    # Load the train and test data from CSV files
    train_data_path = 'artifacts/train.csv'
    test_data_path = 'artifacts/test.csv'
    
    # Load CSV files using pandas
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    
    # Convert DataFrames to numpy arrays
    train_array = train_data.values
    test_array = test_data.values

    # Create an instance of the ModelTrainer class and initiate training
    trainer = ModelTrainer()
    trainer.initiate_model_training(train_array, test_array)