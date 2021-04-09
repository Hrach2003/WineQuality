from sklearn.ensemble import ExtraTreesClassifier


def getExtraTreesModel(X, y):
    et_final = ExtraTreesClassifier()
    et_final.fit(X, y)
    return et_final
