from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

train_scaled = pd.DataFrame(ss.fit_transform(
    train_data), index=train_data.index)
test_scaled = pd.DataFrame(ss.transform(test_data), index=test_data.index)

poly = PolynomialFeatures(degree=2)

combined = pd.DataFrame(poly.fit_transform(combined), index=combined.index)

train_data_poly = combined.xs(0)
test_data_poly = combined.xs(1)

ss = StandardScaler()

train_scaled_poly = pd.DataFrame(ss.fit_transform(
    train_scaled), index=train_scaled.index)
test_scaled_poly = pd.DataFrame(
    ss.transform(test_scaled), index=test_scaled.index)

# SVM
C = [0.05, 0.1, 0.2, 0.3, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
gamma = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
kernel = ['rbf', 'linear']
hyper = {'kernel': kernel, 'C': C, 'gamma': gamma}
gd = GridSearchCV(estimator=svm.SVC(), param_grid=hyper, verbose=True)
gd.fit(X, Y)
print(gd.best_score_)
print(gd.best_estimator_)

# Random Forests
n_estimators = range(100, 1000, 100)
hyper = {'n_estimators': n_estimators}
gd = GridSearchCV(estimator=RandomForestClassifier(
    random_state=0), param_grid=hyper, verbose=True)
gd.fit(X, Y)
print(gd.best_score_)
print(gd.best_estimator_)

# Logistic Regression

lr_model = LogisticRegression(random_state=42, max_iter=1000)

test_params = {
    'penalty': ["l1", "l2", "none"],
    'C': [x for x in np.linspace(0, 10, 500)]
}

lr_gs = RandomizedSearchCV(lr_model, test_params, cv=5, n_jobs=4, n_iter=500)

lr_gs.fit(train_data, train_y)

print(lr_gs.best_params_)
print(lr_gs.best_score_)

lr_gs.fit(train_scaled_poly, train_y)
print(lr_gs.best_params_)
print(lr_gs.best_score_)

# Random Forest Model
rf_model = RandomForestClassifier(random_state=42)

rf_params = {
    'bootstrap': [True, False],
    'max_depth': [int(x) for x in np.linspace(1, 50, 50)],
    'max_features': ['auto', 'sqrt', "none"],
    'min_samples_leaf': [int(x) for x in np.linspace(1, 8, 8)],
    'min_samples_split': [int(x) for x in np.linspace(2, 30, 30)],
    'n_estimators': [int(x) for x in np.linspace(2, 50, 50)]}

rf_gs = RandomizedSearchCV(rf_model, rf_params, cv=5, n_jobs=4, n_iter=500)

rf_gs.fit(train_data, train_y)

print(rf_gs.best_params_)
print(rf_gs.best_score_)

rf_gs.fit(train_scaled_poly, train_y)

print(rf_gs.best_params_)
print(rf_gs.best_score_)

# Support Vector Machine

svc_model = SVC(random_state=42, max_iter=1000, probability=True)

test_params = {
    "kernel": ["linear", "rbf", "poly", "sigmoid"],
    "degree": [2],
    "C": [x for x in np.linspace(0, 10, 500)]
}

svc_gs = RandomizedSearchCV(svc_model, test_params, cv=5, n_jobs=4, n_iter=500)

svc_gs.fit(train_scaled, train_y)

print(svc_gs.best_params_)
print(svc_gs.best_score_)

svc_gs.fit(train_scaled_poly, train_y)

print(svc_gs.best_params_)
print(svc_gs.best_score_)

print(lr_gs.best_score_)
print(rf_gs.best_score_)
print(svc_gs.best_score_)
