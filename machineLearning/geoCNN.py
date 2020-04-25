'''
Creator: Nirvan S P Theethira
Date: 04/24/2020
Purpose:  CSCI 5922 Spring Group Project: GeoGuessr
PLEASE HAVE THE DATA IN THE SAME FOLDER AS THIS FILE BEFORE RUNNING
SAMPLE TRAIN RUN: 
python Mimic.py --trainCharachter joey --epochs 2 --batchSize 20 --saveEpochs 1 --modelSaveFile joey2
SAMPLE LOAD:
python Mimic.py --modelLoadFile joey2
'''


from shapely.geometry import Point, Polygon
from matplotlib import pyplot as plt
from math import sin, cos, sqrt, atan2, radians

# !pip install gmaps
import shapely
import pickle
import random
import numpy as np
import gmaps, os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from ipywidgets.embed import embed_minimal_html
import webbrowser
import argparse

class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
    '''
    Custom callback used to save model every few epochs.
    '''
    def __init__(self, gGuessr, saveFolder, saveEpoch):
        super(LossAndErrorPrintingCallback, self).__init__()
        self.gGuessr = gGuessr
        self.saveEpoch =  saveEpoch
        self.saveFolder = saveFolder

    def on_epoch_end(self, epoch, logs=None):
        '''
        Save model every few epochs
        '''
        self.gGuessr.accuracy = logs['val_accuracy']
        if epoch%self.saveEpoch==0:
            self.gGuessr.save(self.saveFolder)

    def on_train_end(self, logs={}):
        '''
        Save model at the end of training
        '''
        print("Training sucessfull!!")
        self.gGuessr.save(self.saveFolder)
        
def rgb(minimum, maximum, value):
        minimum, maximum = float(minimum), float(maximum)
        ratio = 2 * (value-minimum) / (maximum - minimum)
        b = int(max(0, 255*(1 - ratio)))
        r = int(max(0, 255*(ratio - 1)))
        g = 255 - b - r
        return r, g, b

class Geoguessr:
    def __init__(self, model=None, accuracy=-1, 
                 inputShape=(300,600,3), gridCount=243,
                 modelOptimizer=tf.keras.optimizers.Adam()):
        if model==None:
            # load restnet model
            restnet = tf.keras.applications.resnet50.ResNet50(include_top=False, 
                                                              weights='imagenet', 
                                                              input_shape=inputShape)
            self.model = tf.keras.models.Sequential()
            self.model.add(restnet)
            # freeze restnet model, will not be trained
            self.model.layers[0].trainable = False

            #MNIST cnn model structure to be trained
            self.model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
            self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
            self.model.add(tf.keras.layers.Dropout(0.25))
            self.model.add(tf.keras.layers.Flatten())
            self.model.add(tf.keras.layers.Dense(128, activation='relu'))
            self.model.add(tf.keras.layers.Dropout(0.5))

            # model_finetuned.add(tf.keras.layers.GlobalAveragePooling2D())
            # softmaxed output layer over all grids
            self.model.add(tf.keras.layers.Dense(gridCount, activation="softmax"))
            self.model.compile(loss=tf.keras.losses.categorical_crossentropy,
                              optimizer=modelOptimizer,
                              metrics=['accuracy'])
        else:
            self.model = model
            
        self.model.summary()
        self.accuracy = accuracy
    
    def dataGen(self, X, y, dataDir, batchSize=10):
        totalBatches = len(X)/batchSize
        counter=0
        numClasses = self.model.layers[-1].output_shape[-1]
        inputShape = self.model.layers[0].input_shape[1:3]
        while(True):
            prev = batchSize*counter
            nxt = batchSize*(counter+1)
            counter+=1
            yield np.array(list(map(lambda x:np.array(load_img(dataDir+x, 
                                                               target_size=inputShape)), 
                                    X[prev:nxt]))), \
                            tf.keras.utils.to_categorical(y[prev:nxt], num_classes=numClasses), [None]
            if counter>=totalBatches:
                counter=0
    
    def fit(self, trainFiles, dataDir, saveFolder,
            valSplit=0.97, batchSize = 10, 
            epochs = 20, stepEpoch=5, saveEpoch=1, 
            plot=False):
        # list of image file names
        # eg: <gridNo>+<lat,long>+<imageNo_date>.jpg 
        # eg: 60+48.4271513,-110.5611851+0_2009-06.jpg
        xTrain = trainFiles[:int(len(trainFiles)*valSplit)]
        yTrain = np.array(list(map(lambda x: int(x.split('+')[0]),xTrain)))
        xVal = trainFiles[int(len(trainFiles)*valSplit):]
        yVal = np.array(list(map(lambda x: int(x.split('+')[0]),xTrain)))
        print("Getting data from directory: {}".format(dataDir))
        print("Using {} for training and {} for validation".format(len(xTrain),len(xVal)))
        
        callBack = [LossAndErrorPrintingCallback(self, saveFolder, saveEpoch)]

        evalutaion = self.model.fit(self.dataGen(xTrain, yTrain, dataDir, batchSize=batchSize),
                                      epochs=epochs, steps_per_epoch = stepEpoch,
                                      callbacks=callBack, 
                                      validation_data = self.dataGen(xVal, yVal, 
                                                                     dataDir, batchSize=batchSize),
                                      validation_steps = 1
                                     )
        self.accuracy = evalutaion.history['accuracy'][-1]

        if plot:
            plt.plot(evalutaion.history['val_accuracy'])
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epochs')
            plt.show()

            plt.plot(evalutaion.history['val_accuracy'])
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.show()
            
    def save(self, saveFolder):
        if self.accuracy==-1:
            print("Cannot save untrained model!!!")
        else:
            print("\nSaving model with accuracy {} at {}".format(int(self.accuracy*100), 
                                                               saveFolder))
            self.model.save(saveFolder + '/model_{}.h5'.format(int(self.accuracy*100)))

    @classmethod
    def load(cls, loadFile):
        print("Loading model from {}".format(loadFile))
        model = tf.keras.models.load_model(loadFile)
        modelFile = loadFile.split('/')[-1]
        accuracy = int(modelFile.split('.')[0].split('_')[-1])
        print("Loaded model accuracy {}".format(accuracy))
        return cls(model=model, accuracy=accuracy)
    
    def haversine(self, lati1, long1, lati2, long2):
        # approximate radius of earth in miles
        R = 3958.8

        lat1 = radians(lati1)
        lon1 = radians(long1)
        lat2 = radians(lati2)
        lon2 = radians(long2)

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        return R * c
    
    def gridDist(self, gridPoly1, gridPoly2):
        c1 = Polygon(np.flip(gridPoly1)).centroid
        lati1, long1 = c1.y, c1.x
        c2 = Polygon(np.flip(gridPoly2)).centroid
        lati2, long2 = c2.y, c2.x
        h = self.haversine(lati1, long1, lati2, long2)
        return h, [lati1, long1], [lati2, long2]
    
    def evaluate(self, imgFiles, expectedGrids, ployGrid, checkPoint=50):
        inputShape = self.model.layers[0].input_shape[1:3]
        dists = []
        ln = len(imgFiles)
        for idx, (imgFile,expected) in enumerate(zip(imgFiles,expectedGrids)):
            xx = np.array([np.array(load_img(imgFile, target_size=inputShape))])
            yp = self.model.predict(xx)[0]
            yn = list(map(lambda x:x/max(yp), yp))
            dist, _, _ = self.gridDist(ployGrid[expected],ployGrid[np.argmax(yp)])
            dists.append(dist)
            if idx%checkPoint==0:
                print("Evaluated {} out of {} points".format(idx, ln))
        return np.average(dists)
            
    def predictSingle(self, imgFile, ployGrid=None, expected=None):
        inputShape = self.model.layers[0].input_shape[1:3]
        xx = np.array([np.array(load_img(imgFile, target_size=inputShape))])
        yp = self.model.predict(xx)[0]
        yn = list(map(lambda x:x/max(yp), yp))
        dist, start, end = self.gridDist(ployGrid[expected],ployGrid[np.argmax(yp)])
        if ployGrid and expected:
            mx = max(yn)
            mn = min(yn)
            plt.plot([start[1],end[1]], [start[0],end[0]], color='black', 
                     label="Distance: {} miles".format(round(dist,3)))
            for k,i in ployGrid.items():
                if k==expected:
                    plt.plot(i[:,1],i[:,0],color='blue',label="Actual Grid", alpha=1)
                else:
                    plt.plot(i[:,1],i[:,0],color='black', alpha=0.7)
                plt.fill(i[:,1],i[:,0],color='red', alpha=yn[k])
            plt.legend(loc="lower left")
            plt.show()
            
            gPoly = []
            gLine = gmaps.Line(
                start=start,
                end=end,
                stroke_color = 'blue'
            )
            for grid, polygon in ployGrid.items():
                gPoly.append(gmaps.Polygon(
                                        list(polygon),
                                        stroke_color='black',
                                        fill_color='red',#rgb(mn,mx,float(yn[grid])),
                                        fill_opacity=float(yn[grid])
                                        ))
            fig = gmaps.figure(center=(39.50,-98.35), zoom_level=4)
            fig.add_layer(gmaps.drawing_layer(features=gPoly))
            fig.add_layer(gmaps.drawing_layer(features=[gLine]))
            fig.add_layer(gmaps.symbol_layer([start], scale=3, 
                                 fill_color='green',stroke_color='green', info_box_content='Expected'))
            fig.add_layer(gmaps.symbol_layer([end], scale=3, 
                                             fill_color='yellow', stroke_color='yellow', 
                                             info_box_content='Predicted: {}'.format(dist)))
            embed_minimal_html('gmap.html', views=fig)
            webbrowser.open('gmap.html',new=1)
        return dist

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and test a seq2seq personality chatbot')
    parser.add_argument('--trainCharachter', type=str, help='Name of character to train on. \
                         This helps load data to train model on. eg:joey')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train on. eg:100')
    parser.add_argument('--batchSize', type=int, help='Number of batches to split data into. eg:10')
    parser.add_argument('--saveEpochs', type=int, help='Number of epochs to save after. eg:10')
    parser.add_argument('--modelSaveFile', type=str, help='File to save model to. eg:joey400')

    parser.add_argument('--modelLoadFile', type=str, help='File to load model from. eg:joey400')

    args = parser.parse_args()