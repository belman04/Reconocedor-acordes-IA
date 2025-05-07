import numpy as np                                       # manejo de matrices 
import librosa as lib                                    # procesamiento de audio
import tensorflow as tf                                   # crear y entrenar modelo de red neuronal

dic_acordes = {0: "A", 1: "Am", 2: "B", 3: "Bb", 4: "Bm", 5: "C", 6: "Cm", 7: "D", 8: "Dm", 9: "E", 10: "Em", 11: "F", 12: "Fm", 13: "G", 14: "Gm"}

def tomar_modelo(ruta_modelo):
    try:
        print("Cargando modelo desde:", ruta_modelo)    # Mensaje de carga
        modelo = tf.keras.models.load_model(ruta_modelo)  # cargar el modelo
        print("Modelo cargado exitosamente.")              # Mensaje de éxito
        return modelo
    except Exception as e:
        print("Error al cargar el modelo:", e)   # Mensaje de error
        raise  # Vuelve a lanzar la excepción


def predecir(ruta_audio, modelo):
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
    desired_length = 200   # o el valor que necesites

    # Normalizar la longitud
    if mfccs.shape[1] > desired_length:
        mfccs = mfccs[:, :desired_length]  # Truncar si es mayor
    elif mfccs.shape[1] < desired_length:
        padding_width = desired_length - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, padding_width)), 'constant')  # Rellenar con ceros

    mfccs = np.expand_dims(mfccs, axis=-1)  # Añadir una dimensión para cumplir con la forma (13, 200 , 1)
    # mfccs = mfccs.reshape(1, 13, desired_length, 1)  # Cambiar la forma a (1, 13, 200, 1)

    prediccion = modelo.predict(np.expand_dims(mfccs, axis=0))  # Añadir la dimensión de batch (1, 13, 200 , 1)
    clase_predecida = np.argmax(prediccion, axis=1)[0]  # obtener la clase con mayor probabilidad

    acorde_predecido = dic_acordes[clase_predecida]  # obtener la etiqueta correspondiente
    return acorde_predecido


if __name__ == "__main__":
    modelo = tomar_modelo("modelos/modelo_7.keras")       # cargar el modelo
    acorde = predecir("../datos/acordes/C/audio(10).wav", modelo)   # predecir el acorde
    print("Acorde predecido:", acorde)
