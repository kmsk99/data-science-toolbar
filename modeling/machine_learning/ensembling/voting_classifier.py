from sklearn.ensemble import VotingClassifier

ensemble_lin_rbf = VotingClassifier(estimators=[('KNN', KNeighborsClassifier(n_neighbors=10)),
                                                ('RBF', svm.SVC(probability=True,
                                                                kernel='rbf', C=0.5, gamma=0.1)),
                                                ('RFor', RandomForestClassifier(
                                                    n_estimators=500, random_state=0)),
                                                ('LR', LogisticRegression(C=0.05)),
                                                ('DT', DecisionTreeClassifier(
                                                    random_state=0)),
                                                ('NB', GaussianNB()),
                                                ('svm', svm.SVC(
                                                    kernel='linear', probability=True))
                                                ],
                                    voting='soft').fit(train_X, train_Y)
print('The accuracy for ensembled model is:',
      ensemble_lin_rbf.score(test_X, test_Y))
cross = cross_val_score(ensemble_lin_rbf, X, Y, cv=10, scoring="accuracy")
print('The cross validated score is', cross.mean())


vc = VotingClassifier(estimators=[
    ("lr", lr_gs.best_estimator_),
    ("rf", rf_gs.best_estimator_),
    ("svc", svc_gs.best_estimator_)
], voting="soft")

vc.fit(train_scaled_poly, train_y)
