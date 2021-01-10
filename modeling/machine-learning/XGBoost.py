from sklearn.model_selection import cross_val_score
import xgboost
from sklearn.model_selection import train_test_split
from sklearn import metrics

X_train = df_train.drop("SalePrice_Log", axis=1).values
target_label = df_train["SalePrice_Log"].values
X_test = df_test.values
X_tr, X_vld, y_tr, y_vld = train_test_split(
    X_train, target_label, test_size=0.2, random_state=2000)

# Test하기 전 Validation 과정을 겨처줍니다.
# train데이터의 20%를 validation으로 주고 80%을 train으로 남겨주어 분리해줍니다.

regressor = xgboost.XGBRegressor(colsample_bytree=0.4603, learning_rate=0.06, min_child_weight=1.8,
                                 max_depth=3, subsample=0.52, n_estimators=2000,
                                 random_state=7, ntrhead=-1)
regressor.fit(X_tr, y_tr)

# XGBoost 모델을 만들어줍니다.

y_hat = regressor.predict(X_tr)

plt.scatter(y_tr, y_hat, alpha=0.2)
plt.xlabel('Targets (y_tr)', size=18)
plt.ylabel('Predictions (y_hat)', size=18)
plt.show()

# 예측 된 y 값 (y_hat)에 대한 Scatter Plot을 그려봅니다.

regressor.score(X_tr, y_tr)

y_hat_test = regressor.predict(X_vld)


plt.scatter(y_vld, y_hat_test, alpha=0.2)
plt.xlabel('Targets (y_vld)', size=18)
plt.ylabel('Predictions (y_hat_test)', size=18)
plt.show()

# validation으로 예측해봅니다.

regressor.score(X_vld, y_vld)

accuracies = cross_val_score(estimator=regressor, X=X_tr, y=y_tr, cv=10)

# k-fold validation을 수행합니다.

print(accuracies.mean())
print(accuracies.std())

# 정확도를 확인해봅니다.
