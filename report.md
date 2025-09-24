# Project Report: Simple Linear Regression Web Application

## 1. Project Overview

The objective of this project was to develop a web application that demonstrates a simple linear regression model. The application was built following the CRISP-DM methodology, allowing users to interact with the model by adjusting parameters and visualizing the results.

## 2. CRISP-DM Process

The project adhered to the Cross-Industry Standard Process for Data Mining (CRISP-DM) as follows:

*   **Business Understanding:** The initial goal was to understand the relationship between an independent variable (X) and a dependent variable (y) and to build a model for prediction.
*   **Data Understanding:** Synthetic data was generated using a linear equation with configurable parameters for slope (`a`), noise, and the number of data points.
*   **Data Preparation:** The generated data was split into training and testing sets to prepare for model training and evaluation.
*   **Modeling:** A simple linear regression model from the scikit-learn library was used to learn the relationship between X and y.
*   **Evaluation:** The model's performance was evaluated using the Mean Squared Error (MSE) on the test data.
*   **Deployment:** The model was deployed as a web application using Flask, providing a user interface for interaction.

## 3. Implementation Details

*   **`main.py`:** This Python script contains the core logic of the application. It includes functions for data generation, model training, and evaluation. It also implements the Flask web server that renders the user interface and handles user requests.
*   **`README.md`:** A comprehensive `README.md` file was created to document the entire CRISP-DM process, explaining each step of the project.

## 4. Execution Summary

The Flask application was successfully developed and executed. The user was guided on how to run the application and access it through a web browser. The application, with a CSS-styled interface for a better user experience, allows users to:

*   Set the slope (`a`), noise level, and number of data points.
*   View the generated data and the fitted regression line on a plot.
*   See the Mean Squared Error (MSE) of the model.

## 5. Final Status

The project is complete. The web application is running and accessible, and the project is well-documented in the `README.md` file.
