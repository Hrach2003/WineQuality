from flask import Flask, render_template
import joblib

app = Flask(__name__)


@app.route('/')
def index():
    return str(RandomForestModel)


if __name__ == '__main__':
    RandomForestModel = joblib.load('RandomForestClassifierModel.pkl')
    ExtraTreeModel = joblib.load('ExtraTreesClassifierModel.pkl')

    app.run(debug=True)
