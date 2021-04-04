from sklearn.ensemble import RandomForestClassifier


def getRandForestModel(X, y):
    rf_final = RandomForestClassifier(bootstrap=False, max_features='sqrt', min_samples_leaf=2,
                                      n_estimators=450)
    rf_final.fit(X, y)
    return rf_final
