from flask import Flask, render_template, url_for, request
import joblib
import pandas as pd

app = Flask(__name__)


RandomForestModel = joblib.load('RandomForestClassifierModel.joblib')
ExtraTreeModel = joblib.load('ExtraTreesClassifierModel.joblib')


@app.route('/', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol
        X = {
            'alcohol': [float(request.form.get('alcohol'))],
            'chlorides': [float(request.form.get('chlorides'))],
            'citric acid': [float(request.form.get('citric acid'))],
            'density': [float(request.form.get('density'))],
            'fixed acidity': [float(request.form.get('fixed acidity'))],
            'free sulfur dioxide': [float(request.form.get('free sulfur dioxide'))],
            'pH': [float(request.form.get('pH'))],
            'residual sugar': [float(request.form.get('residual sugar'))],
            'sulphates': [float(request.form.get('sulphates'))],
            'total sulfur dioxide': [float(request.form.get('total sulfur dioxide'))],
            'volatile acidity': [float(request.form.get('volatile acidity'))],
        }

        sample = pd.DataFrame.from_dict(X)
        res = [
            RandomForestModel.predict(sample),
            ExtraTreeModel.predict(sample)]
        return render_template('index.html', res=res)
    elif request.method == 'GET':
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
