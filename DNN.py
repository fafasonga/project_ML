import numpy as np
np.random.seed(123)

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score


scaler = MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)

model = Sequential()

# Add an input layer 
model.add(Dense(50, activation='relu', input_shape=(100,)))
# Add one hidden layer 
model.add(Dense(50, activation='relu'))
# Add an output layer 
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train,epochs=20, batch_size=1, verbose=1)
y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
