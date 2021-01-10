submission = pd.read_csv('../input/sample_submission.csv')
submission.head()

prediction = model.predict(X_test)
submission['Survived'] = prediction

submission.to_csv('my_first_submission.csv', index=False)
