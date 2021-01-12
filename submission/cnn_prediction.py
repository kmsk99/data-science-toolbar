# predict results
results = model.predict(test)

# select the indix with the maximum probability
results = np.argmax(results, axis=1)

results = pd.Series(results, name="Label")

submission = pd.concat(
    [pd.Series(range(1, 28001), name="ImageId"), results], axis=1)

submission.to_csv("cnn_mnist_datagen.csv", index=False)
