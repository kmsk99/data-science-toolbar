# recreate the train_x array and val_x array
train_x = train[list(train.columns)[1:]].values
train_x, val_x = train_test_split(train_x, test_size=0.2)

## normalize and reshape
train_x = train_x/255.
val_x = val_x/255.

train_x = train_x.reshape(-1, 28, 28, 1)
val_x = val_x.reshape(-1, 28, 28, 1)

# Noisy Images

# Lets add sample noise - Salt and Pepper
noise = augmenters.SaltAndPepper(0.1)
seq_object = augmenters.Sequential([noise])

train_x_n = seq_object.augment_images(train_x * 255) / 255
val_x_n = seq_object.augment_images(val_x * 255) / 255

# Before adding noise
f, ax = plt.subplots(1, 5)
f.set_size_inches(80, 40)
for i in range(5, 10):
    ax[i-5].imshow(train_x[i].reshape(28, 28))
plt.show()

# After adding noise
f, ax = plt.subplots(1, 5)
f.set_size_inches(80, 40)
for i in range(5, 10):
    ax[i-5].imshow(train_x_n[i].reshape(28, 28))
plt.show()

# input layer
input_layer = Input(shape=(28, 28, 1))

# encoding architecture
encoded_layer1 = Conv2D(64, (3, 3), activation='relu',
                        padding='same')(input_layer)
encoded_layer1 = MaxPool2D((2, 2), padding='same')(encoded_layer1)
encoded_layer2 = Conv2D(32, (3, 3), activation='relu',
                        padding='same')(encoded_layer1)
encoded_layer2 = MaxPool2D((2, 2), padding='same')(encoded_layer2)
encoded_layer3 = Conv2D(16, (3, 3), activation='relu',
                        padding='same')(encoded_layer2)
latent_view = MaxPool2D((2, 2), padding='same')(encoded_layer3)

# decoding architecture
decoded_layer1 = Conv2D(16, (3, 3), activation='relu',
                        padding='same')(latent_view)
decoded_layer1 = UpSampling2D((2, 2))(decoded_layer1)
decoded_layer2 = Conv2D(32, (3, 3), activation='relu',
                        padding='same')(decoded_layer1)
decoded_layer2 = UpSampling2D((2, 2))(decoded_layer2)
decoded_layer3 = Conv2D(64, (3, 3), activation='relu')(decoded_layer2)
decoded_layer3 = UpSampling2D((2, 2))(decoded_layer3)
output_layer = Conv2D(1, (3, 3), padding='same')(decoded_layer3)

# compile the model
model_2 = Model(input_layer, output_layer)
model_2.compile(optimizer='adam', loss='mse')

model_2.summary()

early_stopping = EarlyStopping(
    monitor='val_loss', min_delta=0, patience=10, verbose=5, mode='auto')
history = model_2.fit(train_x_n, train_x, epochs=20, batch_size=2048,
                      validation_data=(val_x_n, val_x), callbacks=[early_stopping])

f, ax = plt.subplots(1, 5)
f.set_size_inches(80, 40)
for i in range(5, 10):
    ax[i-5].imshow(val_x_n[i].reshape(28, 28))
plt.show()

preds = model_2.predict(val_x_n[:10])
f, ax = plt.subplots(1, 5)
f.set_size_inches(80, 40)
for i in range(5, 10):
    ax[i-5].imshow(preds[i].reshape(28, 28))
plt.show()
