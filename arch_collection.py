
import tensorflow.keras
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

def nvidia_model():
    
    model = tensorflow.keras.Sequential()
    
    model.add(Conv2D(24, kernel_size=(5,5), strides=(2,2), input_shape=(66,200,3),activation='elu'))
    
    model.add(Conv2D(36, kernel_size=(5,5), strides=(2,2), activation='elu'))
    model.add(Conv2D(48, kernel_size=(5,5), strides=(2,2), activation='elu'))
    model.add(Conv2D(64, kernel_size=(3,3), activation='elu'))
    model.add(Conv2D(64, kernel_size=(3,3), activation='elu'))
    #model.add(Dropout(0.5))
    
    
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    # model.add(Dropout(0.5))
    
    
    model.add(Dense(50, activation='elu'))
    # model.add(Dropout(0.5))

    model.add(Dense(10, activation ='elu'))
    # model.add(Dropout(0.5))
    
    model.add(Dense(1)) #This is the vehicle Control *****

    
    optimizer= Adam(lr=2e-4)
    model.compile(loss='mse', optimizer=optimizer)
    return model