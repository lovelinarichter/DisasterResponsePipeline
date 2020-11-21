# Disaster Response Pipeline Project

## Description

This project is a requirement of Data Science NanoDegree - Data Engineering course. The goal of this project is to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages. 

## Table of contents

- [Installation](#installation)
- [Project motivation](#project-motivation)
- [File Structure](#file-structure)
- [Results](#results)

## Installation

The code is implemented using Python and the following libraries. 
- pandas
- sqlalchemy
- nltk 
- numpy
- sklearn
- pickle 
- Flask 
- plotly

To install go to the following links for download and instruction details.
https://www.python.org/downloads/

https://numpy.org/install/

https://scikit-learn.org/stable/install.html

https://www.nltk.org/install.html

https://pypi.org/project/SQLAlchemy/

To Run the codes: 

1. Run the following commands in the project's root directory to set up database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ or http://localhost:3001/
- if running the codes from Udacity Workspace, run "env|grep WORK" in the terminal
- substitute the output spaceid and spacedomain in this url: https://SPACEID-3001.SPACEDOMAIN
![ScreenShot](https://github.com/lovelinarichter/DisasterResponsePipeline/blob/master/run%20image.PNG)


## Project Motivation

This project has three components. 
1. ETL Pipeline - create a Python script named process_data.py. The script cover the following:
- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

2. ML Pipeline - create a Python script named train_classifier.py. The script cover the following:
- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

3. Flask Web app - the flask web app is provided by Udacity. 

## File Structure
- app <br />
| - template <br />
| |- master.html  # main page of web app <br />
| |- go.html  # classification result page of web app <br />
|- run.py  # Flask file that runs app <br />

- data <br />
|- disaster_categories.csv  # data to process <br />
|- disaster_messages.csv  # data to process <br />
|- process_data.py <br />
|- InsertDatabaseName.db   # database to save clean data to <br />

- models <br />
|- train_classifier.py <br />
|- classifier.pkl  # saved model <br />

- README.md

## Results
![ScreenShot](https://github.com/lovelinarichter/DisasterResponsePipeline/blob/master/output%20chart.PNG)
![ScreenShot](https://github.com/lovelinarichter/DisasterResponsePipeline/blob/master/analyze%20message.PNG)
