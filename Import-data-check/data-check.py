
df_train.head()

# 잘 불러졌는지 확인해봅니다.

df_train.shape, df_test.shape

# train 데이터는 1460개의 데이터와 81개의 feature
# test 데이터는 1459개의 데이터와 80개의 feature가 있습니다.

numerical_feats = df_train.dtypes[df_train.dtypes != "object"].index
print("Number of Numerical features: ", len(numerical_feats))

categorical_feats = df_train.dtypes[df_train.dtypes == "object"].index
print("Number of Categorical features: ", len(categorical_feats))

# 편의상 수치형 변수와 명목형 변수를 나눠줍니다.
# 수치형 변수는 38개, 명목형 변수는 43개가 있습니다.

print(df_train[numerical_feats].columns)
print("*"*80)
print(df_train[categorical_feats].columns)

# 변수명을 확인해봅니다.
