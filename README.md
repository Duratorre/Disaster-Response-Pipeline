# Disaster Response Pipeline Project

## Table of Contents
1. [ Installation. ](#inst)
2. [ Motivation. ](#motiv)
3. [ Data. ](#data)
4. [ Instructions. ](#res)
5. [ Licensing, Authors, Acknowledgement. ](#lic)

<a name="inst"></a>
## 1. Installation
There are no requirements needed to run the code in this repository apart from the Anaconda distribution of Python. The code runs with no issues using the Python versions 3 and later ones.

<a name="motiv"></a>
## 2. Motivation
The goal of this project is to classify real messages sent during disaster events into categories. The data is provided from Figure Eight and it will be used to create an ETL and a machine learning pipeline to classify the messages. The result of this process will be a classification model that will be run through a web app, allowing users to input a new message and obtain all the different related categories belonging to the message.

<a name="data"></a>
## 3. Data and File Descriptions
The project is organized into three folders.
  1. _Data_ contains the following files:
      - *disaster_categories.csv* and *disaster_messages.csv* are the datasets respectively about categories and messages
      - *process_data.py* is the ETL pipeline, which takes in the two datasets, merges and cleans them, and outputs a sqlite database
      - *DisasterResponse.db* is the sqlite database
  2. _Model_ contains the following files:
      - *train_classifier.py* is the machine learning pipeline, which loads data from the sqlite database, splits it into train and test sets, fits the training data to a model using text processing techniques, tunes the model via GridSearchCV and outputs the model as a pickle file.
      - *Classifier.pkl* is a pickle file containing the model
  3. _App_ contains the following files:
      - *run.py* is the file used to run the web application
      - *main.html* is the file containing all the code used to display of the main page of the web app
      - *go.html* is the source code of the web page showing classification results once a message is typed in the web app


## 4. Instructions
To run the web app it is necessary to have python installed on your local machine. Then go to the project's root directory.

  1. Run the ETL pipeline with the following command in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

  2. Run the following command in the app's directory to run your web app.
    `python run.py`

  3. Go to http://0.0.0.0:3001/

<a name="data"></a>
## 5. Acknowledgements
Credit to Udacity and Figure Eight for providing the data, and giving precious insights into how to set up the web app. Code used in this repository can be used freely.
