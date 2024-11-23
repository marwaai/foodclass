from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
def create_model():
    model = models.Sequential([
        layers.Input(shape=(25, 25, 3)),
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(3, 3)),

        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dense(101, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
def corrector():
  model = models.Sequential()

  model.add(layers.InputLayer(input_shape=(101,)))

  model.add(layers.Dense(512, activation='relu'))  # Example hidden layer with 512 units
  model.add(layers.Dense(256, activation='relu'))  # Another hidden layer with 256 units
  model.add(layers.Dense(128, activation='relu'))  # Another hidden layer with 128 units
  model.add(layers.Dense(128, activation='relu'))  # Another hidden layer with 128 units

  model.add(layers.Dropout(0.8))
  # Output layer: 101 units (for 101 classes) and softmax activation
  model.add(layers.Dense(101, activation='softmax'))
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  return model