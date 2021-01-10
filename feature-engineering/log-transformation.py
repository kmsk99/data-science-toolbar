f, ax = plt.subplots(1, 1, figsize=(10, 6))
g = sns.distplot(df_train["SalePrice"], color="b", label="Skewness: {:2f}".format(
    df_train["SalePrice"].skew()), ax=ax)
g = g.legend(loc="best")

print("Skewness: %f" % df_train["SalePrice"].skew())
print("Kurtosis: %f" % df_train["SalePrice"].kurt())

# Target Feature인 SalePrice의 비대칭도와 첨도를 확인합니다.
# 그래프와 수치를 확인하면 정상적으로 분포되지 않는것을 확인할 수 있습니다.
# 예측의 정확도를 높히기 위해 로그 변환을 수행합니다.

df_train["SalePrice_Log"] = df_train["SalePrice"].map(
    lambda i: np.log(i) if i > 0 else 0)

f, ax = plt.subplots(1, 1, figsize=(10, 6))
g = sns.distplot(df_train["SalePrice_Log"], color="b", label="Skewness: {:2f}".format(
    df_train["SalePrice_Log"].skew()), ax=ax)
g = g.legend(loc="best")

print("Skewness: %f" % df_train['SalePrice_Log'].skew())
print("Kurtosis: %f" % df_train['SalePrice_Log'].kurt())

df_train.drop('SalePrice', axis=1, inplace=True)

# kewness, Kurtosis를 없애주기 위해 로그를 취해줍니다.
# Log변환을 수행한 새로운 feature "SalePrice_Log"를 만들고 전 Feature인 "Saleprice"를 버려줍니다.
# 로그를 취해준 그래프와 수치가 바뀐 모습을 볼 수 있습니다. (정규근사화)
