id_test = df_test['Id']

to_drop_num = num_weak_corr
to_drop_catg = catg_weak_corr

cols_to_drop = ['Id'] + to_drop_num + to_drop_catg

for df in [df_train, df_test]:
    df.drop(cols_to_drop, inplace=True, axis=1)

# SalePrice와의 상관관계가 약한 모든 변수를 삭제합니다.

df_train.head()
df_train.dtypes
df_test.head()
df_test.dtypes

# 삭제가 잘 진행되었는지 확인합니다.
