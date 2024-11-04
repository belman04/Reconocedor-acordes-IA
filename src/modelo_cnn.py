import numpy as np                                          # manejo de matrices
import tensorflow as tf                                     # crear y entrenar modelo de red neuronal
import pandas as pd                                         # manejo de datos
from sklearn.model_selection import train_test_split as tts # dividir datos en entrenamiento y prueba
import preprocesamiento as prep
import matplotlib.pyplot as plt


def crear_modelo(input_shape, num_classes):
    modelo = tf.keras.Sequential([
        # primera capa convolucional, extrae caracteristicas del espectograma
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),         
        # capa de agrupación (submuestreo) reduce las dimensiones de la salida anterior
        tf.keras.layers.MaxPooling2D((2, 2)), 
        
        # segunda capa convolucional, extrae caracteristicas del espectograma
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        # capa de agrupación (submuestreo) reduce las dimensiones de la salida anterior
        tf.keras.layers.MaxPooling2D((2, 2)), 
        
        tf.keras.layers.Flatten(),                               # capa de aplanamiento, convierte la salida anterior en un vector
        tf.keras.layers.Dense(128, activation='relu'),           # capa densa, aprende la relación entre las caracteristicas extraidas y las etiquetas
        tf.keras.layers.Dense(num_classes, activation='softmax') # capa de salida, clasifica las entradas en las clases 
        #softmax es una función de activación que convierte las salidas en probabilidades
    ])


# Graficar pérdida y precisión
def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    # Pérdida
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Precisión
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


if __name__ == "__main__":
    a, b = prep.tomar_datos("../datos")
    a = np.array([a.flatten() for A in a])                                       # aplanar los MFCCs
    b = pd.factorize(b)[0]                                                       # convertir las etiquetas en valores numéricos

    a_train, a_test, b_train, b_test = tts(a, b, test_size=0.2, random_state=42) # dividir los datos en entrenamiento y prueba

    modelo = crear_modelo(a_train.shape[1:], len(np.unique(b)))           # crear el modelo
    history = modelo.fit(a_train, b_train, epochs=10, batch_size=32)      # entrenar el modelo
    modelo.save("/modelos/modelo.h5")                                     # guardar el modelo
    test_loss, test_acc = modelo.evaluate(a_test, b_test)                 # evaluar el modelo
    print("Precisión del modelo:", test_acc)
    plot_training_history(history)                                       # graficar la pérdida y precisión