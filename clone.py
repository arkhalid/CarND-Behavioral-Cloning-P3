import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout

def basic_network(X_train,y_train):
    model = Sequential();
    model.add(Lambda(lambda x: x/255.0,input_shape=(160,320,3)))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(loss='mse',optimizer='adam')
    model.fit(X_train,y_train, validation_split=0.2,shuffle=True,nb_epoch=7)
    model.save('my_model2.h5')

def lenet_network(X_train,y_train,model_name='lenet_default.h5'):
    model = Sequential();
    model.add(Lambda(lambda x: x/255.0,input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,0),(0,0))))
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dense(84))
    model.add(Dense(1))
    model.compile(loss='mse',optimizer='adam')
    model.fit(X_train,y_train, validation_split=0.2,shuffle=True,nb_epoch=10)
    model.save(model_name)

def nvidia_network(X_train,y_train,existing_weights,use_weights,model_name="nvidia_default.h5"):
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,0),(0,0)))) 
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))  
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))   
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))   
    model.add(Convolution2D(64,3,3,activation="relu"))   
    model.add(Convolution2D(64,3,3,activation="relu"))   
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    if use_weights:
        model.load_weights(existing_weights)
    model.compile(loss='mse',optimizer='adam')
    model.fit(X_train,y_train, validation_split=0.2,shuffle=True,nb_epoch=5)
    model.save(model_name)

def main():
    STEERING_FACTOR=[0,-0.2,0.2]
    dir_list = ['myData','turn2','dirt','correction']
    augmented_images, augmented_measurements = [],[]
    for curr_dir in dir_list:
        lines = []
        with open(curr_dir+'/data/driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                lines.append(line)
    
        images = []
        measurements = []
        for line in lines:
          for i in range(3):
            source_path = line[i]
            filename = source_path.split('\\')[-1]
            current_path = curr_dir + '/data/IMG/'+filename
            if not (line[3] == 'steering'):
                image = cv2.imread(current_path)
                images.append(image)
                measurement = float(line[3])
                measurements.append(measurement+STEERING_FACTOR[i])
            else:
                print(line)
        
        for image,measurement in zip(images,measurements):
            augmented_images.append(image)
            augmented_measurements.append(measurement)
            augmented_images.append(cv2.flip(image,1))
            augmented_measurements.append(measurement*-1.0)
    X_train = np.array(augmented_images)
    y_train = np.array(augmented_measurements)
    print("Training :")
    print(X_train.shape)
    nvidia_network(X_train,y_train,'nvidia_data_2604172338_cropping.h5',False,'nvidia_data_0205170212_from_scratch.h5')
    #nvidia_network(X_train,y_train,'nvidia_data_0205170059_cropping_turns_refined.h5','nvidia_data_0205170130_cropping_turns__dirt_refined.h5')
    #nvidia_network(X_train,y_train,'','nvidia_data_test.h5')

if __name__ == "__main__":
    main()
