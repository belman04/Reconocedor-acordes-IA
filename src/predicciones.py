import numpy as np                                       # manejo de matrices
import librosa as lib                                    # procesamiento de audio
import tensorflow as tf                                  # crear y entrenar modelo de red neuronal

def tomar_modelo(ruta_modelo):
    return tf.keras.models.tomar_modelo(ruta_modelo)       # cargar el modelo


def predecir(ruta_audio, modelo):
    audio, sr = lib.load(ruta_audio, sr=None)            # cargar el archivo de audio
    mfccs = lib.feature.mfcc(y=audio, sr=sr, n_mfcc=13)  # extraer los MFCCs
    mfccs = mfccs.flatten().shape(1, -1)                 # cambia segun la forma de los datos de entrada

    prediccion = modelo.predecir(mfccs)                  # predecir la clase del acorde
    clase_predecida = np.argmax(prediccion, axis=1)      # obtener la clase con mayor probabilidad
    return clase_predecida


if __name__ == "__main__":
    modelo = tomar_modelo("../modelo/modelo.h5")         # cargar el modelo, cambia segun la ruta del modelo
    acorde = predecir("../datos/audio.wav", modelo)      # predecir el acorde, cambia segun el audio que quieras
    print("Acorde predecido:", acorde)
