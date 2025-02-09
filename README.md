# Store Sales Prediction App 📊💡

## Overview 🌟

The **Store Sales Prediction App** is a web application built with **Streamlit** that predicts store sales based on three key parameters:
- **Store Area (sq. ft)** 🏢
- **Items Available** 📦
- **Daily Customer Count** 👥

This app uses machine learning to forecast store sales, allowing store managers and owners to plan better for inventory and operations. 

## Features ✨

- **User-friendly Interface**: Intuitive and easy-to-use interface for inputting data.
- **Instant Predictions**: Get predictions on sales with just one click! ⏳
- **Real-time Results**: Predict sales based on your inputs and see the output instantly! 💡

## Technologies Used 🛠️

- **Frontend**: Streamlit
- **Backend**: Python, Custom Machine Learning Pipeline
- **Data Processing**: Pandas
- **Machine Learning Models**: Custom models for training and prediction

## Installation 🚀

### Prerequisites 📋

To run this app locally, make sure you have **Python 3.x** and **pip** installed.

### Steps 🏃‍♂️

1. **Clone the repository**:

    ```bash
    git clone https://github.com/devyanshidubey-02/Flask-Driven-Predictor.git
    ```

2. **Navigate to the project directory**:

    ```bash
    cd Flask-Driven-Predictor
    ```

3. **Create and activate a virtual environment (optional but recommended)**:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

4. **Install the required dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

5. **Run the Streamlit application**:

    ```bash
    streamlit run app.py
    ```

6. **Access the app**: 
    Open your browser and visit `http://localhost:8501` to interact with the app.

## Usage 💻

### Input Fields 📝:
- **Store Area (sq. ft)** 🏢: Enter the area of the store in square feet.
- **Items Available** 📦: Enter the total number of items in the store.
- **Daily Customer Count** 👥: Enter the daily customer footfall.

### Prediction 📈:
Once you've filled out the fields, click on the **"Predict Store Sales"** button. The app will predict the store sales based on the inputs provided.

### Example:
1. **Store Area**: 1500 sq. ft
2. **Items Available**: 500
3. **Daily Customer Count**: 150

**Predicted Store Sales:** 25,000.50 💵

---

## Code Structure 📂

### `app.py` 📲

The core Streamlit app, `app.py`, defines the structure and flow of the application:
- Configures the app's title and layout.
- Accepts user input for **store area**, **items available**, and **daily customer count**.
- Triggers the prediction pipeline when the "Predict Store Sales" button is clicked.
- Displays the predicted store sales.

---

## Components 🧩

### 1. **`data_ingestion.py` 🗂️**

This component handles the ingestion of data. It collects raw data from various sources (e.g., CSV, database) and prepares it for transformation and processing. The goal is to collect relevant data that can be fed into the ML pipeline for prediction.

- **Functions**: Data loading, handling missing values, basic data cleaning.
- **Example usage**: Ensures that all required data is available before proceeding to the transformation stage.

### 2. **`data_transformation.py` 🔄**

The `data_transformation.py` component is responsible for transforming the ingested data into a format suitable for model training and prediction. This includes operations such as feature scaling, encoding categorical variables, and splitting the data into training and testing datasets.

- **Functions**: Feature engineering, scaling, encoding, and data splitting.
- **Example usage**: Converts raw data into numeric features for machine learning models to process.

### 3. **`model_trainer.py` 🏋️‍♂️**

This component trains the machine learning models on the preprocessed data. It takes transformed data and fits the model to it. After training, it saves the model for later use in predictions.

- **Functions**: Model selection, training, hyperparameter tuning, and model saving.
- **Example usage**: Trains a regression or classification model and saves it as a file (e.g., `.pkl` or `.joblib`).

### 4. **`model_evaluation.py` 📊**

This component evaluates the performance of the trained model. It calculates performance metrics like **accuracy**, **mean squared error (MSE)**, **R-squared**, and more. It helps to assess how well the model is performing and if it can make reliable predictions.

- **Functions**: Model evaluation metrics, comparison of models, plotting of performance charts.
- **Example usage**: Evaluates the trained model and visualizes metrics such as **MSE** and **R2 score**.

---

## Pipelines 🚄

### 1. **`training_pipeline.py` 🛠️**

This pipeline orchestrates the entire process of model training:
- Ingests data.
- Transforms data.
- Trains the model.
- Saves the trained model for future use.

The `training_pipeline.py` ensures a clean flow from raw data to a trained model ready for deployment.

- **Steps**: Data ingestion → Data transformation → Model training → Model saving
- **Functions**: Integrates all components (data ingestion, transformation, model training).

### 2. **`prediction_pipeline.py` 🔮**

Once the model is trained and saved, the **prediction pipeline** handles the process of loading the trained model and using it to predict new data.

- **Steps**: Load trained model → Process new data → Make predictions.
- **Functions**: Loading the saved model, making predictions, formatting results for output.

---

## Contributing 🤝
We welcome contributions to improve this app! To contribute, follow these steps:

- Fork the repository.
- Create a new branch (git checkout -b feature-branch).
- Make your changes and improvements.
- Commit your changes (git commit -am 'Add new feature').
- Push your branch to the repository (git push origin feature-branch).
- Open a pull request on GitHub.
- Please ensure that your code adheres to the project’s coding conventions and includes tests where necessary.

---

## Acknowledgments 🏆
- Streamlit: For enabling easy deployment of interactive data apps.
- Machine Learning Pipeline: For predictive modeling and data transformation.
- GitHub: For hosting and version control.
- Everyone contributing: This project is possible because of the contributions of all the open-source developers and tools.

---
