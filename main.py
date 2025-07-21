from flask import Flask, request, render_template, send_file, redirect, flash
import pandas as pd
import numpy as np
from prophet import Prophet
import io
import os
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error

app = Flask(__name__)
app.secret_key = 'secret-key'



@app.route('/')
def index():
    all_comments = load_comments()
    positive_keywords = ['love', 'great', 'awesome', 'helpful', 'amazing', 'nice', 'good', 'best']
    comments = [c for c in all_comments if any(word in c['message'].lower() for word in positive_keywords)]
    testimonials = load_testimonials()
    return render_template('index.html', testimonials=testimonials, comments=comments)


def load_comments():
    if os.path.exists('comments.txt'):
        with open('comments.txt', 'r', encoding='utf-8') as f:
            return [parse_comment(line) for line in f.readlines()]
    return []

def parse_comment(line):
    parts = line.split(':', 1)
    if len(parts) == 2:
        return {'name': parts[0].strip(), 'message': parts[1].strip()}
    return {'name': 'Anonymous', 'message': line.strip()}


@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/download_csv')
def download_csv():
    return send_file("sales_data.csv", as_attachment=True)

@app.route('/comment', methods=['POST'])
def comment():
    name = request.form.get('name', 'Anonymous')
    content = request.form.get('message', '')


    if content.strip():
        with open('comments.txt', 'a', encoding='utf-8') as f:
            f.write(f"{name}: {content}\n")
        if any(word in content.lower() for word in ['love', 'great', 'helpful', 'awesome', 'best']):
            with open('testimonials.txt', 'a', encoding='utf-8') as f:
                f.write(f"{name}: {content}\n")
    return redirect('/')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file:
        flash("No file uploaded")
        return redirect('/upload')

    try:
        periods = int(request.form['periods'])
    except ValueError:
        flash("Invalid forecast period")
        return redirect('/upload')

    try:
        # Load CSV
        df = pd.read_csv(file)
        if df.shape[1] < 2:
            flash("CSV must have 2 columns: date and value.")
            return redirect('/upload')

        df = df.iloc[:, :2]
        df.columns = ['ds', 'y']
        df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        df.dropna(subset=['ds', 'y'], inplace=True)

        if df.empty:
            flash("Uploaded data is empty or invalid.")
            return redirect('/upload')

        # Train
        model = Prophet()
        model.fit(df)

        # Predict future
        future = model.make_future_dataframe(periods=periods, freq='M')
        forecast = model.predict(future)
        future_forecast = forecast[['ds', 'yhat']].tail(periods)
        future_forecast = future_forecast.rename(columns={'ds': 'date', 'yhat': 'forecast'})

        # Save to CSV
        future_forecast.to_csv("sales_data.csv", index=False)

        # Model Evaluation — not shown to user
        eval_df = pd.merge(df, forecast[['ds', 'yhat']], on='ds', how='inner')
        y_true = eval_df['y']
        y_pred = eval_df['yhat']

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = (abs((y_true - y_pred) / y_true).mean()) * 100

        print(f"[EVALUATION] MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

        # Send to frontend
        forecast_data = future_forecast.to_dict(orient='records')
        return render_template('result.html', forecast=forecast_data, periods=periods)

    except Exception as e:
        flash(f"Error during forecasting: {str(e)}")
        return redirect('/upload')


def load_testimonials():
    if os.path.exists('testimonials.txt'):
        with open('testimonials.txt', 'r', encoding='utf-8') as f:
            return [parse_comment(line) for line in f.readlines()]
    return []




if __name__ == '__main__':
    app.run(debug=True)
    '''
from flask import Flask, request, render_template, send_file, redirect, flash
import pandas as pd
import numpy as np
from prophet import Prophet
import io
import os
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error

app = Flask(__name__)
app.secret_key = 'secret-key'

POSITIVE_KEYWORDS = ['love', 'great', 'awesome', 'helpful', 'amazing', 'nice', 'good', 'best']

@app.route('/')
def index():
    all_comments = load_comments()
    comments = [c for c in all_comments if any(word in c['message'].lower() for word in POSITIVE_KEYWORDS)]
    testimonials = load_testimonials()
    return render_template('index.html', testimonials=testimonials, comments=comments)

def load_comments():
    if os.path.exists('comments.txt'):
        with open('comments.txt', 'r', encoding='utf-8') as f:
            return [parse_comment(line) for line in f.readlines()]
    return []

def parse_comment(line):
    parts = line.split(':', 1)
    if len(parts) == 2:
        return {'name': parts[0].strip(), 'message': parts[1].strip()}
    return {'name': 'Anonymous', 'message': line.strip()}

def load_testimonials():
    if os.path.exists('testimonials.txt'):
        with open('testimonials.txt', 'r', encoding='utf-8') as f:
            return [parse_comment(line) for line in f.readlines()]
    return []

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/download_csv')
def download_csv():
    return send_file("sales_data.csv", as_attachment=True)

@app.route('/comment', methods=['POST'])
def comment():
    name = request.form.get('name', 'Anonymous')
    content = request.form.get('message', '').strip()

    if content:
        # Save to all comments
        with open('comments.txt', 'a', encoding='utf-8') as f:
            f.write(f"{name}: {content}\n")

        # Save only positive ones to testimonials
        if any(word in content.lower() for word in POSITIVE_KEYWORDS):
            with open('testimonials.txt', 'a', encoding='utf-8') as f:
                f.write(f"{name}: {content}\n")

    return redirect('/')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file:
        flash("No file uploaded")
        return redirect('/upload')

    try:
        periods = int(request.form['periods'])
    except ValueError:
        flash("Invalid forecast period")
        return redirect('/upload')

    try:
        # Load and clean CSV
        df = pd.read_csv(file)
        if df.shape[1] < 2:
            flash("CSV must have 2 columns: date and value.")
            return redirect('/upload')

        df = df.iloc[:, :2]
        df.columns = ['ds', 'y']
        df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        df.dropna(subset=['ds', 'y'], inplace=True)

        if df.empty:
            flash("Uploaded data is empty or invalid.")
            return redirect('/upload')

        # Forecast
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=periods, freq='M')
        forecast = model.predict(future)
        future_forecast = forecast[['ds', 'yhat']].tail(periods)
        future_forecast = future_forecast.rename(columns={'ds': 'date', 'forecast': 'forecast'})

        # Save forecast
        future_forecast.to_csv("sales_data.csv", index=False)

        # Evaluation (not shown to user)
        eval_df = pd.merge(df, forecast[['ds', 'yhat']], on='ds', how='inner')
        y_true = eval_df['y']
        y_pred = eval_df['yhat']
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = (abs((y_true - y_pred) / y_true).mean()) * 100
        print(f"[EVALUATION] MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

        # Send result
        forecast_data = future_forecast.to_dict(orient='records')
        return render_template('result.html', forecast=forecast_data, periods=periods)

    except Exception as e:
        flash(f"Error during forecasting: {str(e)}")
        return redirect('/upload')

if __name__ == '__main__':
    app.run(debug=True)'''

