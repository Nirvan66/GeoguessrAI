'''
Creator: Nirvan S P Theethira
Date: 04/24/2020
Purpose:  CSCI 6502 Spring Project: GeoGuessr
Note: This file can be only run on sample data locally.

SAMPLE TRAIN RUN
python geoCNN.py

SAMPLE TEST RUN
python geoCNN.py --testModelName model_0.083_6mill.h5
'''

from shapely.geometry import Point, Polygon
from matplotlib import pyplot as plt
from math import sin, cos, sqrt, atan2, radians

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
    def __init__(self, gGuessr, saveFolder, modelNumber):
        '''
        gGuess: Instance of Geoguessr
        saveFolder: Name of floder to save to
        modelNumber: A number to add to saved model file nam
        '''
        super(LossAndErrorPrintingCallback, self).__init__()
        self.gGuessr = gGuessr
        self.modelNumber =  modelNumber
        self.saveFolder = saveFolder

    def on_epoch_end(self, epoch, logs=None):
        '''
        Save model every few epochs
        '''
        self.gGuessr.loss = round(float(logs['loss']),3)

    def on_train_end(self, logs={}):
        '''
        Save model at the end of training
        '''
        print("Training sucessfull!!")
        self.gGuessr.save(self.saveFolder, self.modelNumber)
        


class Geoguessr:
    '''
    The class has all the functions required to built, train and test the geoguessr CNN model.
    '''
    def __init__(self, model=None, loss=-1, 
                 inputShape=(300,600,3), gridCount=243,
                 modelOptimizer=tf.keras.optimizers.Adam(), 
                 hidden1=256, hidden2=1024):
        '''
        The function is used to load or initialize a new model with specifies shapes and hidden layers
        '''
        if model==None:
            # load restnet model
            restnet = tf.keras.applications.resnet50.ResNet50(include_top=False, 
                                                              weights='imagenet', 
                                                              input_shape=inputShape)
            self.model = tf.keras.models.Sequential()
            self.model.add(restnet)

            # freeze resnet model
            self.model.layers[0].trainable = False

            ##### MNIST cnn model structure to be trained 
            self.model.add(tf.keras.layers.Conv2D(hidden1, (3, 3), activation='relu', 
                                                    input_shape=inputShape))
            self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
            self.model.add(tf.keras.layers.Dropout(0.25))
            self.model.add(tf.keras.layers.Flatten())
            self.model.add(tf.keras.layers.Dense(hidden2, activation='relu'))
            self.model.add(tf.keras.layers.Dropout(0.5))
            self.model.add(tf.keras.layers.Dense(gridCount, activation="softmax"))

            self.model.compile(loss=tf.keras.losses.categorical_crossentropy,
                              optimizer=modelOptimizer,
                              metrics=['categorical_accuracy'])
        else:
            self.model = model
            
        self.model.summary()
        self.loss = loss
    
    def dataGen(self, fileNames, dataDir, batchSize=10, infinite=True):
        '''
        The function is used to generate input ouput pairs in batches.
        Inifinit: Tells the function to stop or keep going once the list of file names has been iterated through
        fileNames should look like: 60+48.4271513,-110.5611851+0_2009-06.jpg
        '''
        totalBatches = len(fileNames)/batchSize
        counter=0
        while(True):
            prev = batchSize*counter
            nxt = batchSize*(counter+1)
            counter+=1
            yield self.readData(fileNames[prev:nxt],dataDir)
            if counter>=totalBatches:
                if infinite:
                    counter=0
                else:
                    break

    def readData(self, fileNames, dataDir):
        '''
        Takes a list of file names and gives a list of input image vector, output one hot grid vector pairs.
        fileNames should look like: 60+48.4271513,-110.5611851+0_2009-06.jpg
        '''
        numClasses = self.model.layers[-1].output_shape[-1]
        inputShape = self.model.layers[0].input_shape[1:3]
        return np.array(list(map(lambda x:np.array(load_img(dataDir+x, 
                                                        target_size=inputShape)), 
                            fileNames))), \
                tf.keras.utils.to_categorical(list(map(lambda x:int(x.split('+')[0]), fileNames)), 
                                              num_classes=numClasses)
    
    def fit(self, trainFiles, dataDir, saveFolder, batchSize = 10, 
            epochs = 20,
            plot=False):
        # list of image file names
        # eg: <gridNo>+<lat,long>+<imageNo_date>.jpg 
        # eg: 60+48.4271513,-110.5611851+0_2009-06.jpg
        print("Getting data from directory: {}".format(dataDir))
        accuracy = []
        loss = []
        cnt = 0
        for X,y in self.dataGen(trainFiles, dataDir, batchSize=batchSize, infinite=False):
            callBack = [LossAndErrorPrintingCallback(self, saveFolder, cnt)]
            print("Read {} points. Training now".format(len(X)))
            evalutaion = self.model.fit(X,y,
                                        epochs=epochs, steps_per_epoch = len(X),
                                        callbacks=callBack)
            accuracy += evalutaion.history['categorical_accuracy']
            loss += evalutaion.history['loss']
            cnt += 1
        if plot:
            plt.plot(accuracy)
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epochs')
            plt.show()

            plt.plot(loss)
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.show()
            
    def save(self, saveFolder, modelNumber=0):
        '''
        Saves model to specified folder with specified number along with loss
        '''
        if self.loss==-1:
            print("Cannot save untrained model!!!")
        else:
            print("\nSaving model {} with loss {} at {}".format(modelNumber,
                                                                self.loss, 
                                                               saveFolder))
            self.model.save(saveFolder + '/model_{}_{}.h5'.format(self.loss,
                                                                  modelNumber))

    @classmethod
    def load(cls, loadFile):
        '''
        Loads model from specified folder with loss
        '''
        print("Loading model from {}".format(loadFile))
        model = tf.keras.models.load_model(loadFile)
        modelFile = loadFile.split('/')[-1]
        loss = float(modelFile.split('_')[1])
        print("Loaded model loss {}".format(loss))
        return cls(model=model, loss=loss)
    
    def haversine(self, lati1, long1, lati2, long2):
        '''
        Gives distance in miles between two points on the planet specified by latitudes and longitudes
        '''
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
        '''
        Gives distance in miles between the centers of two grids or polygons. 
        '''
        c1 = Polygon(np.flip(gridPoly1)).centroid
        lati1, long1 = c1.y, c1.x
        c2 = Polygon(np.flip(gridPoly2)).centroid
        lati2, long2 = c2.y, c2.x
        h = self.haversine(lati1, long1, lati2, long2)
        return h, [lati1, long1], [lati2, long2]
    
    def evaluate(self, imgFiles, dataDir, ployGrid, checkPoint=50):
        '''
        Calculates average of distances between target and predicted grids for a list of files
        imgFile has to look like: 
        # eg: <gridNo>+<lat,long>+<imageNo_date>.jpg 
        # eg: 60+48.4271513,-110.5611851+0_2009-06.jpg
        '''
        dists = []
        ln = len(imgFiles)
        for idx,(xx,yy) in enumerate(self.dataGen(imgFiles, dataDir, batchSize=1, infinite=False)):
            yp = self.model.predict(xx)[0]
            yn = list(map(lambda x:x/max(yp), yp))
            dist, _, _ = self.gridDist(ployGrid[np.argmax(yy[0])],ployGrid[np.argmax(yp)])
            dists.append(dist)
            if idx%checkPoint==0:
                print("Evaluated {} out of {} points".format(idx, ln))
        return np.average(dists)
            
    def predictSingle(self, imgFile, dataDir, ployGrid=None):
        '''
        Predicts softmax ouput by trained model for single image and plots it 
        imgFile has to look like: 
        # eg: <gridNo>+<lat,long>+<imageNo_date>.jpg 
        # eg: 60+48.4271513,-110.5611851+0_2009-06.jpg
        '''
        xx,yy = self.readData([imgFile], dataDir)
        yp = self.model.predict(xx)[0]
        yn = list(map(lambda x:x/max(yp), yp))
        dist, start, end = self.gridDist(ployGrid[np.argmax(yy[0])],ployGrid[np.argmax(yp)])
        if ployGrid:
            mx = max(yn)
            mn = min(yn)
            plt.plot([start[1],end[1]], [start[0],end[0]], color='black', 
                     label="Distance: {} miles".format(round(dist,3)))
            for k,i in ployGrid.items():
                if k==np.argmax(yy[0]):
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
                                        fill_color='red',
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
    parser = argparse.ArgumentParser(description='Train and test a geguessr model')
    parser.add_argument('--testModelName', type=str, help='Name of model to run test on. Required for test mode')

    args = parser.parse_args()
    if args.testModelName!=None:
        print("In testing mode.")
        geoModel = Geoguessr.load("models/" + args.testModelName)
        TESF = np.load('../infoExtraction/dataSample/testFileSample.npy')
        usaPolyGrid = pickle.load(open("../infoExtraction/usaPolyGrid.pkl",'rb'))
        geoModel.predictSingle(TESF[0], "../infoExtraction/dataSample/imageFilesSample/", ployGrid=usaPolyGrid)

    else:
        print("In training mode. Training new model")
        TF = np.load('../infoExtraction/dataSample/trainFileSample.npy')
        geoModel = Geoguessr(hidden1=256,hidden2=256)
        geoModel.fit(trainFiles = TF, 
                      dataDir = "../infoExtraction/dataSample/imageFilesSample/", 
                      saveFolder = "models",
                      epochs=5,
                      batchSize=5,
                      plot=True
                     )



