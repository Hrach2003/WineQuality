# %%
from models import RandomForestClassifier, ExtraTreesClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

# %%
df = pd.read_csv('./data/wineQuality(cleaned).csv')
X = df[df.columns.difference(['quality'])]
y = df['quality']

# %%
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

pkl_filenames = [("ExtraTreesClassifierModel.pkl", ExtraTreesClassifier.getExtraTreesModel(x_train, y_train)),
                 ("RandomForestClassifierModel.pkl", RandomForestClassifier.getRandForestModel(x_train, y_train))]

if __name__ == '__main__':
    for filename, model in pkl_filenames:
        joblib.dump(model, filename)
