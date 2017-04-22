import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

def basic_network(X_train,y_train):
    model = Sequential();
    model.add(Lambda(lambda x: x/255.0,input_shape=(160,320,3)))
    model.add(Flatten(input_shape=(160,320,3)))
    model.add(Dense(1))
    model.compile(loss='mse',optimizer='adam')
    model.fit(X_train,y_train, validation_split=0.2,shuffle=True,nb_epoch=7)
    model.save('my_model2.h5')

def lenet_network(X_train,y_train,model_name='lenet_default.h5'):
    model = Sequential();
    model.add(Lambda(lambda x: x/255.0,input_shape=(160,320,3)))
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dense(84))
    model.add(Dense(1))
    model.compile(loss='mse',optimizer='adam')
    model.fit(X_train,y_train, validation_split=0.2,shuffle=True,nb_epoch=7)
    model.save(model_name)
   
def main():
    lines = []
    with open('./myData/data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    
    images = []
    measurements = []
    for line in lines:
        source_path = line[0]
        filename = source_path.split('\\')[-1]
        current_path = './myData/data/IMG/'+filename
        if not (line[3] == 'steering'):
            image = cv2.imread(current_path)
            images.append(image)
            measurement = float(line[3])
            measurements.append(measurement)
        else:
            print(line)
    X_train = np.array(images)
    y_train = np.array(measurements)
    print("Training :")
    print(X_train.shape)
    lenet_network(X_train,y_train,'lenet_2204171447.h5')

if __name__ == "__main__":
    main()
