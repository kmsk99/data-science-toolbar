for col in df_train.columns:
    msperc = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(
        col, 100 * (df_train[col].isnull().sum() / df_train[col].shape[0]))
    print(msperc)

# train 데이터 각 column의 결측치가 몇 %인지 확인합니다.
# df_train[col].isnull().sum() : 해당 열의 결측치가 몇개인지 알 수 있게하는 문장입니다. (TRUE=1(결측치), FALSE=0으로 계산)
# df_train[col].shape[0] : 해당 열의 차원 (열이 지정되어 있으므로 행의 갯수를 보여줍니다.)
# 100 * (df_train[col].isnull().sum() / df_train[col].shape[0] : 위의 설명을 통해 %를 출력해주는 문장임을 알 수 있습니다.

for col in df_test.columns:
    msperc = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(
        col, 100 * (df_test[col].isnull().sum() / df_test[col].shape[0]))
    print(msperc)

# test 데이터도 확인해줍니다.
# train, test 모두 PoolQc 데이터가 가장 결측치가 많습니다.

missing = df_train.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar(figsize=(12, 6))

# 직관적으로 확인하기 위해 barplot을 그려봅니다.
