df_submit = pd.read_csv('../input/sample_submission.csv')
df_submit.head()

prediction = model.predict(X_test)
df_submit['Survived'] = prediction

df_submit.to_csv('my_first_submission.csv', index=False)
