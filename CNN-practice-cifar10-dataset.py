import tensorflow as tf


def load_dataset():
    (trainX, trainY), (testX, testY) = tf.keras.datasets.cifar10.load_data()

    trainX = trainX.astype('float32')
    testX = testX.astype('float32')

    trainX = trainX / 255.0
    testX = testX / 255.0

    trainY = tf.keras.utils.to_categorical(trainY)
    testY = tf.keras.utils.to_categorical(testY)

    return trainX, trainY, testX, testY


def define_model():
    model = tf.keras.models.Sequential()

    # layer_1
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    # layer_2
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    # layer_3
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    return model


trainX, trainY, testX, testY = load_dataset()

opt = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)

model = define_model()

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(trainX, trainY, epochs=20, batch_size=64, validation_data=(testX, testY))
sloss, acc = model.evaluate(testX, testY)

print(' > %.3f' % (acc * 100.0))
