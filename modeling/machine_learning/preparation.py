from sklearn.model_selection import train_test_split


X_train = df_train.drop("SalePrice_Log", axis=1).values
target_label = df_train["SalePrice_Log"].values
X_test = df_test.values

X_train.shape, X_test.shape

X_tr, X_vld, y_tr, y_vld = train_test_split(
    X_train, target_label, test_size=0.2, random_state=2000)

# Test하기 전 Validation 과정을 겨처줍니다.
# train데이터의 20%를 validation으로 주고 80%을 train으로 남겨주어 분리해줍니다.

y_tr.shape, y_vld.shape