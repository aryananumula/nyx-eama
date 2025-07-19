from tensorflow import keras
from tensorflow.keras import layers

def create_vit_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    # Vision Transformer architecture
    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    return model

def compile_model(model):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

def train_model(model, train_data, train_labels, validation_data, validation_labels, epochs=10):
    history = model.fit(train_data, train_labels, 
                        validation_data=(validation_data, validation_labels), 
                        epochs=epochs)
    return history

def evaluate_model(model, test_data, test_labels):
    test_loss, test_accuracy = model.evaluate(test_data, test_labels)
    print(f'Test accuracy: {test_accuracy:.2f}')
    return test_loss, test_accuracy