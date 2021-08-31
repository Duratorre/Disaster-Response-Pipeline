# Disaster Response Pipeline Project

## Table of Contents
1. [ Installation. ](#inst)
2. [ Motivation. ](#motiv)
3. [ Data. ](#data)
4. [ Instructions. ](#ins)
5. [ Examples. ](#exs)
6. [ Licensing, Authors, Acknowledgement. ](#lic)

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
      - *DisasterResponse.db* is the sqlite database that will be output from *process_data.py*
  2. _Model_ contains the following files:
      - *train_classifier.py* is the machine learning pipeline, which loads data from the sqlite database, splits it into train and test sets, fits the training data to a model using text processing techniques, tunes the model via GridSearchCV and outputs the model as a pickle file.
      - *classifier.pkl* is a pickle file containing the model that will be output from *train_classifier.py*
  3. _App_ contains the following files:
      - *run.py* is the file used to run the web application
      - *main.html* is the file containing all the code used to display of the main page of the web app
      - *go.html* is the source code of the web page showing classification results once a message is typed in the web app

<a name="ins"></a>
## 4. Instructions
To run the web app it is necessary to have python installed on your local machine. The following are the steps needed for a Linux OS. If you are using a Windows machine, the actions are the same, just remember to write the whole path to the python.exe application on your machine instead of just 'python'.

  1. Go to the project's root directory (i.e. 'cd ../project_root_directory'):
      - Run the ETL pipeline with the following command to clean and set up your database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
      - Run the ML pipeline with the following command to train the classifier and save it in a pickle file
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`      
  2. Go to the app directory ('cd ./app') and run the following command to run the web app:
        `python run.py`
  3. Go to http://localhost:3001/

<a name="exs"></a>
## 5. Examples
Below there are some images taken from the web app.

The first three images come from the main page: the first one displays the box where to insert the message to be classified, the second one shows the distribution of message genres and the top 5 message categories in the training data, while the third shows the distribution of message length used in the training data.

The last screenshot represents the result of a message classification: after the message is typed, the web app shows all the different predicted categories it belongs to.

  1. Main page

  ![WebApp_Insert](https://user-images.githubusercontent.com/48055172/127248512-fa832715-7b56-40d5-8fa8-0e3244d4adbd.JPG)
  ![WebApp_Main1](https://user-images.githubusercontent.com/48055172/127248514-0e99107e-281d-4711-ad1b-5a3ca3b86d22.JPG)
  ![WebApp_Main2](https://user-images.githubusercontent.com/48055172/127248510-023cb5f0-c7a2-4362-b566-ba997e53e368.JPG)

  2. Result page

  ![WebApp_Go](https://user-images.githubusercontent.com/48055172/127248472-219470c9-f77a-47b4-b1ba-77c4eff636dc.JPG)

<a name="data"></a>
## 6. Acknowledgements
Credit to Udacity and Figure Eight for providing the data, and giving precious insights into how to set up the web app. Code used in this repository can be used freely.
