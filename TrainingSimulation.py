import os

import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Accuracy

import random

import cv2
import matplotlib.pyplot as plt
import numpy.random
import pandas as pd
import numpy as np
import os
import matplotlib.image as mpimg
from sklearn.utils import shuffle
import  imgaug.augmenters as iaa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D,Flatten,Dense
from tensorflow.keras.optimizers import Adam



def getName(filePath):
    return filePath.split('\\')[-1]

def importDataInfo(path):
    columns=['Center','Left','Right','Steering','Throttle','Brake','Speed']
    data=pd.read_csv(os.path.join(path,'driving_log.csv'),names=columns)
    # print(data.head())
    # print(data['Center'][0])
    # print(getName(data['Center'][0]))
    data['Center']=data['Center'].apply(getName)
    # print(data.head())
    print('Total image in Data', data.shape[0])
    return data

def balanceData(data, display=True):
    nBins=31
    samplesPerBin=800
    hist, bins = np.histogram(data['Steering'],nBins)
    if display:
        center = (bins[:-1]+bins[1:])*0.5
        plt.bar(center,hist,width=0.06)
        plt.plot((-1,1),(samplesPerBin,samplesPerBin))
        plt.show()

    removeIndexList=[]
    for j in range(nBins):
        binDataList=[]
        for i in range(len(data['Steering'])):
            if data['Steering'][i]>=bins[j] and data['Steering'][i]<=bins[j+1]:
                binDataList.append(i)

        binDataList=shuffle(binDataList)
        binDataList=binDataList[samplesPerBin:]
        removeIndexList.extend(binDataList)

    print('Removed Images', len(removeIndexList))
    data.drop(data.index[removeIndexList],inplace=True)
    print('Remaining Data', len(data))


    if display:
        hist, _ = np.histogram(data['Steering'], nBins)
        plt.bar(center, hist, width=0.06)
        plt.plot((-1, 1), (samplesPerBin, samplesPerBin))
        plt.show()


    return data


def loadData(path,data):
    imagesPath=[]
    steering=[]

    for i in range(len(data)):
        indexedData=data.iloc[i]
        imagesPath.append(os.path.join(path,'IMG',indexedData.iloc[0]))
        steering.append(float(indexedData.iloc[3]))

    imagesPath=np.asarray(imagesPath)
    steering=np.asarray(steering)
    return imagesPath,steering

def augmentImage(imgPath,steering):
    img=mpimg.imread(imgPath)
    ##PAN
    if numpy.random.rand()<0.5:
        pan=iaa.Affine(translate_percent={'x':(-0.1,0.1),'y':(-0.1,0.1)})
        img=pan.augment_image(img)

    ##ZOOM
    if numpy.random.rand() < 0.5:
        zoom=iaa.Affine(scale=(1,1.2))
        img=zoom.augment_image(img)

    #BRIGHTNESS
    if numpy.random.rand() < 0.5:
        brightness=iaa.Multiply((0.4,1.2))
        img=brightness.augment_image(img)

    ##Flip
    if numpy.random.rand() < 0.5:
        img=cv2.flip(img,1)
        steering=-steering

    return img, steering


def preprocessing(img):
    img=img[60:135,:,:]
    img=cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
    img=cv2.GaussianBlur(img,(3,3),0)
    img=cv2.resize(img,(200,66))
    img=img/255
    return img

def batchGen(imagesPath,steeringList,batchSize,trainFlag):
    while True:
        imgBatch=[]
        steeringBatch=[]

        for i in range(batchSize):
            index=random.randint(0,len(imagesPath)-1)
            if trainFlag:
                img,steering = augmentImage(imagesPath[index],steeringList[index])
            else:
                img=mpimg.imread(imagesPath[index])
                steering=steeringList[index]
            img=preprocessing(img)
            imgBatch.append(img)
            steeringBatch.append(steering)
        yield (np.asarray(imgBatch),np.asarray(steeringBatch))


def createModel():
    model=Sequential()
    model.add(Convolution2D(24,(5,5),(2,2),input_shape=(66,200,3),activation='elu'))
    model.add(Convolution2D(36,(5,5),(2,2),activation='elu'))
    model.add(Convolution2D(48,(5,5),(2,2),activation='elu'))
    model.add(Convolution2D(64,(3,3),activation='elu'))
    model.add(Convolution2D(64,(3,3),activation='elu'))
    model.add(Flatten())
    model.add(Dense(100,activation='elu'))
    model.add(Dense(50,activation='elu'))
    model.add(Dense(10,activation='elu'))
    model.add(Dense(1))

    model.compile(Adam(learning_rate=0.0001),loss='mse')

    return model



###data
path='E:/self-drive-data'

data=importDataInfo(path)

data = balanceData(data,display=False)

#images and steering data
imagesPath,steerings =loadData(path,data)

## train_test_split
x_train,x_val,y_train,y_val=train_test_split(imagesPath,steerings,test_size=0.2,random_state=5)
print('total train images',x_train.shape)
print('total val images',x_val.shape)


#augmentation - 1000 images converted to 3000 values


##preprocessing image


##model
model=createModel()
model.summary()

#model fit
history=model.fit(batchGen(x_train,y_train,100,1),steps_per_epoch=300,epochs=10,
          validation_data=batchGen(x_val,y_val,100,0),validation_steps=200)

## model save

model.save('model.h5')
print('Model Saved')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training','Validation'])
plt.ylim([0,1])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()