import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
sc_pred = pickle.load(open('scaler.pkl','rb'))
sc_x = pickle.load(open('scaler_x.pkl','rb'))

model = load_model('lstm_bitcoin.h5')
#df = pd.read_csv('df_test.csv')
df.drop(columns = 'index', inplace = True)

def create_lookback(dataset, look_back=30):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 1:8]
        X.append(a)
        Y.append(dataset[i + look_back, 1])
    return np.array(X), np.array(Y)

def predict(df): #argument df is a pandas dataframe
  X_test = df.drop(columns = 'Bitcoin Core (BTC) Price')
  X_test = pd.DataFrame(sc_x.transform(X_test), columns = X_test.columns)

  Y_test = df['Bitcoin Core (BTC) Price']
  Y_test = pd.DataFrame(sc_pred.transform(Y_test.values.reshape(-1,1)))

  X_test.insert(value = Y_test.values, column = 'Bitcoin Core (BTC) Price', loc = 1  )

  X_test, Y_test = create_lookback(X_test.values, 30)

  X_test = np.reshape(X_test, (len(X_test), 30, X_test.shape[2]))

  # Predict
  predictions = model.predict(X_test)
  predictions = sc_pred.inverse_transform(predictions)
  return predictions.flatten().tolist() #return will be a list of strings