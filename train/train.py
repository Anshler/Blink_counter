import csv
import numpy as np
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, Flatten, Dense, Activation, Dropout, MaxPooling2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import imageio
import os
import fnmatch

#we will use images 26x34x1 (1 is for grayscale images)
height = 26
width = 34
dims = 1
#read images from folder
def readImg():
    dir='dataset'
    count = len(fnmatch.filter(os.listdir(dir), '*.*'))
    imgs = np.empty((count, height, width, dims), dtype=np.uint8)
    tgs = np.empty((count, 1))
    for i in range(0,count):
     im = imageio.imread('dataset\\'+os.listdir(dir)[i])
     im = im.reshape((26, 34))
     im = np.expand_dims(im, axis=2)
     imgs[i]=im
     if os.listdir(dir)[i].__contains__("open"):
         tgs[i]=1
     else:
         tgs[i]=0
    imgs = np.array(imgs, dtype=np.uint8)
    tgs = np.array(tgs, dtype=np.uint8)
    index = np.random.permutation(imgs.shape[0])
    imgs = imgs[index]
    tgs = tgs[index]
    return imgs, tgs


def readCsv(path):

    f=open(path,'r')
    reader = csv.DictReader(f)
    rows = list(reader)

    #imgs is a numpy array with all the images
    #tgs is a numpy array with the tags of the images
    imgs = np.empty((len(list(rows)),height,width, dims),dtype=np.uint8)
    tgs = np.empty((len(list(rows)),1))

    for row,i in zip(rows,range(len(rows))):

        #convert the list back to the image format
        img = row['image']
        img = img.strip('[').strip(']').split(', ')
        im = np.array(img,dtype=np.uint8)
        im = im.reshape((26,34))
        im = np.expand_dims(im, axis=2)
        imgs[i] = im

        #the tag for open is 1 and for close is 0
        tag = row['state']
        if tag == 'open':
            tgs[i] = 1
        else:
            tgs[i] = 0

    #shuffle the dataset
    index = np.random.permutation(imgs.shape[0])
    imgs = imgs[index]
    tgs = tgs[index]

    #return images and their respective tags
    return imgs,tgs

#make the convolution neural network
def makeModel():
    model = Sequential()

    model.add(Conv2D(32, (3,3), padding = 'same', input_shape=(height,width,dims)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (2,2), padding= 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (2,2), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))


    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy',metrics=['accuracy'])

    return model

def main():

    X_train ,Y_train = readImg()
    print (X_train.shape[0])
    #scale the values of the images between 0 and 1
    X_Train = X_train.astype('float32')
    X_Train /= 255

    model = makeModel()

    #do some data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        )
    datagen.fit(X_train)

    #train the model
    history = model.fit(datagen.flow(X_train,Y_train,batch_size=32), steps_per_epoch=len(X_train) / 32, epochs=50)
    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()

    #save the model
    model.save('blinkModel.hdf5')

if __name__ == '__main__':
    main()
