import keras
def predict(data):
    #rf = joblib.load('pipe_rf_model.sav')
    model = keras.models.load_model('my_model.h5')
    return model.predict(data)
