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
    STEERING_FACTOR=[0,-0.2,0.2]
    lines = []
    with open('./myData/data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    
    images = []
    measurements = []
    for line in lines:
      for i in range(3):
        source_path = line[i]
        filename = source_path.split('\\')[-1]
        current_path = './myData/data/IMG/'+filename
        if not (line[3] == 'steering'):
            image = cv2.imread(current_path)
            images.append(image)
            measurement = float(line[3])
            measurements.append(measurement+STEERING_FACTOR[i])
        else:
            print(line)
    augmented_images, augmented_measurements = [],[]
    for image,measurement in zip(images,measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        augmented_images.append(cv2.flip(image,1))
        augmented_measurements.append(measurement*-1.0)
    X_train = np.array(augmented_images)
    y_train = np.array(augmented_measurements)
    print("Training :")
    print(X_train.shape)
    lenet_network(X_train,y_train,'lenet_data_2304171256.h5')

if __name__ == "__main__":
    main()
