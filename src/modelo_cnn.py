import numpy as np                                          # manejo de matrices
import tensorflow as tf                                     # crear y entrenar modelo de red neuronal
import pandas as pd                                         # manejo de datos
from sklearn.model_selection import train_test_split as tts # dividir datos en entrenamiento y prueba
import preprocesamiento as prep


def crear_modelo(input_shape, num_classes):
    modelo = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),  # Usar Input(shape) como la primera capa
        # primera capa convolucional, extrae caracteristicas del espectograma
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),         
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
    modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # compilar modelo
    return modelo


if __name__ == "__main__":
    print("Cargando datos...")
    try:
        a, b = prep.tomar_datos("../datos") # Cargar los datos
        print("Datos cargados correctamente.")
    except Exception as e:
        print(f"Error al cargar los datos: {e}")

    print("Aplanando los MFCCs...")
    try:
        a_flat = np.array([x.flatten() for x in a]) # Aplanar los MFCCs
        print("MFCCs aplanados exitosamente.")
    except Exception as e:
        print(f"Error al aplanar los MFCCs: {e}")
    
    print("Redimensionando los MFCCs...")
    try:
        a = a.reshape(a.shape[0], a.shape[1], a.shape[2], 1) # Redimensionar los MFCCs
        print(f"MFCCs redimensionados a la forma: {a.shape}")
    except Exception as e:
        print(f"Error al redimensionar los MFCCs: {e}")

    print("Convirtiendo etiquetas en valores numéricos...")
    try:
        b = pd.factorize(b)[0] # Convertir etiquetas en valores numéricos
        print("Etiquetas convertidas exitosamente.")
    except Exception as e:
        print(f"Error al convertir etiquetas: {e}")

    print("Dividiendo datos en conjuntos de entrenamiento y prueba...")
    try:
        a_train, a_test, b_train, b_test = tts(a, b, test_size=0.2, random_state=42) # Dividir los datos
        print("Datos divididos correctamente en conjuntos de entrenamiento y prueba.")
    except Exception as e:
        print(f"Error al dividir los datos: {e}")

    print("Creando el modelo...")
    try:
        modelo = crear_modelo(a_train.shape[1:], len(np.unique(b)))  # Crear el modelo
        print("Modelo creado correctamente.")
        
        # compilando el modelo
        modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
        print("Modelo compilado correctamente.")
    except Exception as e:
        print(f"Error al crear o compilar el modelo: {e}")

    print("Entrenando el modelo...")
    try:
        modelo.fit(a_train, b_train, epochs=10, batch_size=32) # Entrenar el modelo
        print("Modelo entrenado exitosamente.")
    except Exception as e:
        print(f"Error al entrenar el modelo: {e}")

    print("Guardando el modelo...")
    try:
        modelo.save("modelos/modelo.keras")   # Guardar el modelo
        print("Modelo guardado exitosamente.")
    except Exception as e:
        print(f"Error al guardar el modelo: {e}")

    print("Evaluando el modelo...")
    try:
        test_loss, test_acc = modelo.evaluate(a_test, b_test)  # Evaluar el modelo
        print("Evaluación completa.")
        print("Precisión del modelo:", test_acc)
    except Exception as e:
        print(f"Error al evaluar el modelo: {e}")