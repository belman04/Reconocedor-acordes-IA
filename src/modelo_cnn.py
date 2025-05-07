import os
import numpy as np                                          # manejo de matrices
import tensorflow as tf                                     # crear y entrenar modelo de red neuronal
import pandas as pd                                         # manejo de datos
from sklearn.model_selection import train_test_split as tts # dividir datos en entrenamiento y prueba
from sklearn.preprocessing import StandardScaler            # escalar los datos
from sklearn.metrics import classification_report, confusion_matrix # evaluar el modelo
import matplotlib.pyplot as plt                             # visualización de datos
import seaborn as sns                                       # visualización de datos
import preprocesamiento as prep                             # preprocesamiento de datos

# diccionario de acordes
dic_acordes = {0: "A", 1: "Am", 2: "B", 3: "Bb", 4: "Bm", 5: "C", 6: "Cm", 7: "D", 8: "Dm", 9: "E", 10: "Em", 11: "F", 12: "Fm", 13: "G", 14: "Gm"}

def crear_modelo(input_shape, num_classes):
    modelo = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),  # usar Input(shape) como la primera capa
        # primera capa convolucional, extrae caracteristicas del espectograma
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
        tf.keras.layers.BatchNormalization(), # capa de normalización
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'), # capa de agrupación (submuestreo) reduce las dimensiones de la salida anterior
        
        # segunda capa convolucional, extrae caracteristicas del espectograma
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
        tf.keras.layers.BatchNormalization(), # capa de normalización
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'), # capa de agrupación (submuestreo) reduce las dimensiones de la salida anterior
        
        # tercera capa convolucional, extrae caracteristicas del espectograma
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
        tf.keras.layers.BatchNormalization(), # capa de normalización
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'), # capa de agrupación (submuestreo) reduce las dimensiones de la salida anterior

        # cuarta capa convolucional, extrae caracteristicas del espectograma
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'),
        tf.keras.layers.BatchNormalization(), # capa de normalización
        tf.keras.layers.MaxPooling2D((1, 2), padding='same'), # capa de agrupación (submuestreo) reduce las dimensiones de la salida anterior
      
        tf.keras.layers.Flatten(),                               # capa de aplanamiento, convierte la salida anterior en un vector
        tf.keras.layers.Dropout(0.5),                            # capa de dropout, previene el sobreajuste
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1)), # aumentar a 0.1 si no mejora
        tf.keras.layers.Dropout(0.3),                            # capa de dropout, previene el sobreajuste
        tf.keras.layers.Dense(num_classes, activation='softmax') # capa de salida, clasifica las entradas en las clases 
        #softmax es una función de activación que convierte las salidas en probabilidades
    ])
    return modelo


# Función para graficar entrenamiento
def graficar_entrenamiento(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

# Función para graficar la precisión y la pérdida durante el entrenamiento
def graficar_precision_perdida(history):
    plt.figure(figsize=(12, 6))

    # Graficar Pérdida
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Graficar Precisión
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Función para mostrar matriz de confusión
def mostrar_matriz_confusion(b_test, predicciones_clase, dic_acordes):
    cm = confusion_matrix(b_test, predicciones_clase)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, xticklabels=dic_acordes.values(), yticklabels=dic_acordes.values(), cmap="Blues")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


# Función para graficar la distribución de clases reales y predichas
def graficar_distribucion_clases(b_test, predicciones_clase, dic_acordes):
    plt.figure(figsize=(12, 6))

    # Graficar la distribución de clases reales
    plt.subplot(1, 2, 1)
    plt.hist(b_test, bins=np.arange(len(dic_acordes)+1)-0.5, alpha=0.7, color='b', rwidth=0.8, label='True Classes')
    plt.xticks(np.arange(len(dic_acordes)), dic_acordes.values())
    plt.xlabel('Classes')
    plt.ylabel('Frequency')
    plt.title('Distribution of True Classes')
    plt.legend()

    # Graficar la distribución de clases predichas
    plt.subplot(1, 2, 2)
    plt.hist(predicciones_clase, bins=np.arange(len(dic_acordes)+1)-0.5, alpha=0.7, color='g', rwidth=0.8, label='Predicted Classes')
    plt.xticks(np.arange(len(dic_acordes)), dic_acordes.values())
    plt.xlabel('Classes')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predicted Classes')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
# Función para graficar la curva de aprendizaje
def curva_de_aprendizaje(history):
    plt.figure(figsize=(12, 6))

    # Pérdida
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Precisión
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


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
        scaler = StandardScaler()
        a_flat = scaler.fit_transform(a_flat)
        print("MFCCs aplanados y normalizados exitosamente.")
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
        a_train, a_test, b_train, b_test = tts(a, b, test_size=0.2, stratify=b, random_state=42) # Dividir los datos
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
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)  # Detener el entrenamiento si no hay mejoras
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-5) # Reducir la tasa de aprendizaje si no hay mejoras

        history = modelo.fit(a_train, b_train, epochs=150, batch_size=32, validation_split=0.2, callbacks=[early_stopping, reduce_lr])
        print("Modelo entrenado exitosamente.")
        graficar_entrenamiento(history)
        graficar_precision_perdida(history)
        curva_de_aprendizaje(history)
    except Exception as e:
        print(f"Error al entrenar el modelo: {e}")

    print("Guardando el modelo...")
    try:
        carpeta_modelos = "modelos/"
        contador = len([archivo for archivo in os.listdir(carpeta_modelos) if archivo.startswith("modelo_")])
        nombre_archivo = f"{carpeta_modelos}modelo_{contador + 1}.keras"

        modelo.save(nombre_archivo)   # Guardar el modelo
        print("Modelo guardado exitosamente.")
    except Exception as e:
        print(f"Error al guardar el modelo: {e}")

    print("Evaluando el modelo...")
    try:
        # Evaluar el modelo
        test_loss, test_acc = modelo.evaluate(a_test, b_test)
        print("Evaluación completa.")
        print(f"Pérdida en el conjunto de prueba: {test_loss}")
        print(f"Precisión en el conjunto de prueba: {test_acc}")
        
        # Predicciones
        predicciones = modelo.predict(a_test)
        predicciones_clase = np.argmax(predicciones, axis=1)

        # Reporte de clasificación
        print("Reporte de clasificación:")
        print(classification_report(b_test, predicciones_clase, target_names=[dic_acordes[i] for i in range(len(dic_acordes))]))

        # Matriz de confusión
        print("Generando matriz de confusión...")
        mostrar_matriz_confusion(b_test, predicciones_clase, dic_acordes)

        # Comprobar clases no predichas (opcional)
        clases_predichas = np.unique(predicciones_clase)
        clases_no_predichas = np.setdiff1d(np.unique(b_test), clases_predichas)
        graficar_distribucion_clases(b_test, predicciones_clase, dic_acordes)
        if clases_no_predichas.size > 0:
            print(f"Las siguientes clases no están siendo predichas: {[dic_acordes[c] for c in clases_no_predichas]}")
        else:
            print("El modelo está prediciendo todas las clases correctamente.")
    except Exception as e:
        print(f"Error al evaluar el modelo: {e}")