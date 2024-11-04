import numpy as np                                       # manejo de matrices 
import librosa as lib                                    # procesamiento de audio
import tensorflow as tf                                   # crear y entrenar modelo de red neuronal
import pandas as pd                                       # manejo de datos


def tomar_etiquetas(ruta_etiquetas):
    etiquetas_df = pd.read_csv(ruta_etiquetas)  # leer  CSV de etiquetas
    # Crear un diccionario que mapea índices a acordes
    acorde_mapping = dict(zip(etiquetas_df.index, etiquetas_df['etiquetas']))
    return acorde_mapping


def tomar_modelo(ruta_modelo):
    try:
        print("Cargando modelo desde:", ruta_modelo)    # Mensaje de carga
        modelo = tf.keras.models.load_model(ruta_modelo)  # cargar el modelo
        print("Modelo cargado exitosamente.")              # Mensaje de éxito
        return modelo
    except Exception as e:
        print("Error al cargar el modelo:", e)   # Mensaje de error
        raise  # Vuelve a lanzar la excepción


def predecir(ruta_audio, modelo, acorde_mapping):
    print(f"Cargando archivo de audio desde: {ruta_audio}")  # Imprimir ruta
    try:
        audio, sr = lib.load(ruta_audio, sr=None)  # cargar el archivo de audio
        print("Archivo de audio cargado correctamente.")
    except Exception as e:
        print(f"Error al cargar el audio: {e}")  # Imprimir error
        return None  # Salir de la función si hay un error
    
    mfccs = lib.feature.mfcc(y=audio, sr=sr, n_mfcc=13)  # extraer los MFCCs
    print("MFCCs extraídos con forma:", mfccs.shape)  # Mostrar la forma de los MFCCs
    
    # Determinar la longitud deseada
    desired_length = 173  # o el valor que necesites

    # Normalizar la longitud
    if mfccs.shape[1] > desired_length:
        mfccs = mfccs[:, :desired_length]  # Truncar si es mayor
    elif mfccs.shape[1] < desired_length:
        padding_width = desired_length - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, padding_width)), 'constant')  # Rellenar con ceros

    mfccs = np.expand_dims(mfccs, axis=-1)  # Añadir una dimensión para cumplir con la forma (13, 173, 1)
    mfccs = mfccs.reshape(1, 13, desired_length, 1)  # Cambiar la forma a (1, 13, 173, 1)

    prediccion = modelo.predict(mfccs)  # predecir la clase del acorde
    clase_predecida = np.argmax(prediccion, axis=1)[0]  # obtener la clase con mayor probabilidad

    # Usar el mapeo de etiquetas que has cargado
    acorde = acorde_mapping.get(clase_predecida, "Desconocido")
    return acorde


if __name__ == "__main__":
    modelo = tomar_modelo("modelos/modelo.keras")       # cargar el modelo
    acorde_mapping = tomar_etiquetas("../datos/etiquetas.csv")  # cargar el mapeo de acordes
    acorde = predecir("../datos/F_prueba.wav", modelo, acorde_mapping)   # predecir el acorde
    print("Acorde predecido:", acorde)
