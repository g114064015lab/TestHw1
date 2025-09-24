import numpy as np
from flask import Flask, render_template_string, request
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

def generate_data(a, n_points, noise):
    """Generate synthetic data for linear regression."""
    X = np.linspace(0, 10, n_points)
    y = a * X + np.random.normal(0, noise, n_points)
    return X.reshape(-1, 1), y

def train_and_evaluate_model(X, y):
    """Train a linear regression model and evaluate it."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return model, mse, X_train, y_train, X_test, y_test, y_pred

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        a = float(request.form['a'])
        noise = float(request.form['noise'])
        n_points = int(request.form['n_points'])
    else:
        a = 2
        noise = 1
        n_points = 50

    X, y = generate_data(a, n_points, noise)
    model, mse, X_train, y_train, X_test, y_test, y_pred = train_and_evaluate_model(X, y)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train, y_train, color='blue', label='Training data')
    plt.scatter(X_test, y_test, color='green', label='Testing data')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression line')
    plt.title('Simple Linear Regression')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template_string('''
        <!doctype html>
        <html>
            <head>
                <title>Simple Linear Regression</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        margin: 2em;
                        background-color: #f4f4f4;
                    }
                    h1, h2 {
                        color: #333;
                    }
                    form {
                        margin-bottom: 2em;
                        background: #fff;
                        padding: 1em;
                        border-radius: 5px;
                    }
                    label {
                        margin-right: 1em;
                    }
                    input[type="text"] {
                        padding: 5px;
                        border-radius: 3px;
                        border: 1px solid #ccc;
                    }
                    input[type="submit"] {
                        padding: 5px 15px;
                        background-color: #007bff;
                        color: white;
                        border: none;
                        border-radius: 3px;
                        cursor: pointer;
                    }
                    img {
                        max-width: 100%;
                        height: auto;
                        border: 1px solid #ddd;
                        border-radius: 5px;
                    }
                </style>
            </head>
            <body>
                <h1>Simple Linear Regression</h1>
                <form method="post">
                    <label for="a">Slope (a):</label>
                    <input type="text" name="a" value="{{ a }}">
                    <label for="noise">Noise:</label>
                    <input type="text" name="noise" value="{{ noise }}">
                    <label for="n_points">Number of points:</label>
                    <input type="text" name="n_points" value="{{ n_points }}">
                    <input type="submit" value="Run">
                </form>
                <h2>Results</h2>
                <p>Mean Squared Error: {{ mse }}</p>
                <img src="data:image/png;base64,{{ plot_url }}">
            </body>
        </html>
    ''', a=a, noise=noise, n_points=n_points, mse=mse, plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
