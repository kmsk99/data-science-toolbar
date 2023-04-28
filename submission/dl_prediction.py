submission = pd.read_csv('../input/sample_submission.csv')
prediction = nn_model.predict(X_test)
prediction = prediction > 0.5
prediction = prediction.astype(np.int)
prediction = prediction.T[0]
prediction.shape

submission['Survived'] = prediction
submission.to_csv('my_nn_submission.csv', index=False)
