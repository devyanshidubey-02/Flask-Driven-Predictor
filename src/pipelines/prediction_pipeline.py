import os
import sys
import pandas as pd
from src.exception import customexception
from src.logger import logging
from src.utils.utils import load_object


class PredictPipeline:

    def __init__(self):
        logging.info("Initializing the prediction pipeline object...")

    def predict(self, features, target=None):
        try:
            # Load preprocessor and model
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model.pkl")

            logging.info("Loading preprocessor and model from the artifacts...")
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            # Apply preprocessor transformation and predict
            logging.info("Applying preprocessing to the input data...")
            scaled_fea = preprocessor.transform(features)
            pred = model.predict(scaled_fea)

            # Skip model score for single sample predictions
            if target is not None and len(target) > 1:
                score = model.score(scaled_fea, target)
                logging.info(f"Model Score: {score}")
            else:
                logging.info("Model score calculation skipped for single sample prediction.")

            # Log actual vs predicted values
            logging.info("Actual vs Predicted values:")
            if target is not None:
                for actual, predicted in zip(target, pred):
                    logging.info(f"Actual: {actual}, Predicted: {predicted}")
            else:
                for p in pred:
                    logging.info(f"Predicted: {p}")

            logging.info("Prediction completed successfully.")
            return pred

        except Exception as e:
            raise customexception(e, sys)


class CustomData:
    def __init__(self,
                 store_area: float,
                 items_available: float,
                 daily_customer_count: float):

        self.store_area = store_area
        self.items_available = items_available
        self.daily_customer_count = daily_customer_count

    def get_data_as_dataframe(self):
        try:
            # Prepare input data dictionary
            custom_data_input_dict = {
                'Store_Area': [self.store_area],
                'Items_Available': [self.items_available],
                'Daily_Customer_Count': [self.daily_customer_count]
            }
            # Convert to DataFrame
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe created successfully.')
            return df
        except Exception as e:
            logging.info('Exception occurred while creating the dataframe.')
            raise customexception(e, sys)


# if __name__ == "__main__":
#     try:
#         # Example: Predicting a custom data sample
#         # custom_data = CustomData(
#         #     store_area=1650,
#         #     items_available=1850,
#         #     daily_customer_count=500
#         # )

#         # # Get the data as a dataframe
#         # df = custom_data.get_data_as_dataframe()

#         # # Assuming target (Store_Sales) is known for comparison
#         # target = [50000]  

#         # # Initialize the prediction pipeline and get prediction
#         # predict_pipeline = PredictPipeline()
#         # prediction = predict_pipeline.predict(df, target)

#         # # Output the prediction result
#         # print(f"Predicted Store Sales: {prediction[0]}")


#         # actual_value = 66490
#         # predicted_value = 70629.79985872863

#         # Calculate the error and percentage error
#         # error = abs(actual_value - predicted_value)
#         # percentage_error = (error / actual_value) * 100

#         # print(f"Error: {error}")
#         # print(f"Percentage Error: {percentage_error:.2f}%")


#     except Exception as e:
#         logging.error(f"Error in executing the prediction pipeline: {e}")