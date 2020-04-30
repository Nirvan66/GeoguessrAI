# GeoguessrAI
Geolocating google street view images using CNNs.

## Map creation
* The geolocation task for this project was restricted to the map of mainland USA
* The map of mainlad USA was extracted as a polygon from the US shapefile. The map USA mainland polygon is stored in the file `infoExtraction/usaPoly.pkl`. The python script for extracting the map from the shapefile can be found in the notebook `infoExtraction/gridCreation.ipynb`.
* The map was then split into grids of roughly even size and the list of polygon grids was stored in the file `infoExtraction/usaPolyGrid.pkl`. The code for creating the grids can be found in the notebook `infoExtraction/gridCreation.ipynb`

## Dataset
* The dataset for this project was stree view images scraped from random locations per grid. Three images were scraped from a single location using the google street view static API.
* Forty locations were scraped from each grid across 243 grids bringing the total dataset size to 29160. 
* The python scripts for data scraping can be found in the notebook `infoExtraction/DataScraping.ipynb`
* Only a sample of 20 images can be found at `infoExtration/dataSample/imageFilesSample`. 
* The 20 sample image file names are split into train and test set of 15 and 5 respectively. This sample train test split can be found at `infoExtration/dataSample/trainFileSample.npy` and `infoExtration/dataSample/testFileSample.npy` respectively.
* Note: The data found in this repo is just a sample dataset to mock training and testing features of the model. The entire dataset can be found at: https://drive.google.com/drive/folders/1e7RPZEf8-SN9rgsO1oJXE__u04ZEeUXM?usp=sharing

## CNN model
* The class called `Geoguessr` is used for training, testing, saving and loading the model. This class can be found in the file `machineLearning/geoCNN.py`. The file can be run from terminal using the run instructions provided at the begining of the .py file. The training and testing runs will only be run on a sample of data as there the complete dataset is not in this repo
* The class can also be used by running cells in the notebook `machineLearning/geoGuessrCNN_TrainTest.ipynb` to train and test a model.
* The trained models are not in this repo and can be found at: https://drive.google.com/drive/folders/1e7RPZEf8-SN9rgsO1oJXE__u04ZEeUXM?usp=sharing
