import librosa
import tensorflow as tf
import keras
import numpy as np

SAMPLES_TO_CONSIDER = 22050
input_duration = 3

class _Stress_Level:

    model = None
    _mapping = ["not_stress", "stress"]
    _instance = None


    def predict(self, file_path):
        # extract features
        features = [self.preprocess(file_path)]

        # we need a 4-dim array to feed to the model for prediction: (# samples, # time steps, # coefficients, 1)
        features = np.expand_dims(features, axis=2)

        # get the predicted label
        preds = self.model.predict(features, batch_size=16, verbose=1)
        preds1=preds.argmax(axis=1).item()
        predicted_level = self._mapping[preds1]

        return predicted_level


    def preprocess(self, file_path, hop_length=512, frame_length=1024):

        # load audio file
        X, sample_rate = librosa.load(file_path, res_type='kaiser_fast',duration=input_duration,sr=22050*2,offset=0.5)
        
        # preprocess
        stft = np.abs(librosa.stft(X))
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
        rms = np.mean(librosa.feature.rms(X, frame_length=frame_length, hop_length=hop_length), axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        feature = np.concatenate((rms, mfccs, chroma), axis=0)
        return feature


def Stress_API():

    # ensure an instance is created only the first time the factory function is called
    if _Stress_Level._instance is None:
        _Stress_Level._instance = _Stress_Level()
        _Stress_Level.model = tf.keras.models.load_model("2stress_model_ravdess.h5")
    return _Stress_Level._instance




'''if __name__ == "__main__":

    # create 2 instances of the keyword spotting service
    kss = Keyword_Spotting_Service()
    kss1 = Keyword_Spotting_Service()

    # check that different instances of the keyword spotting service point back to the same object (singleton)
    assert kss is kss1

    # make a prediction
    keyword = kss.predict("03-01-04-01-02-01-02.wav")
    keyword1 = kss.predict("03-01-08-02-02-01-02.wav")
    print(keyword)
    print(keyword1)'''