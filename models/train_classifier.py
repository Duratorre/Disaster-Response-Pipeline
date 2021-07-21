# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import re
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import pickle


def load_data(database_filepath):
    '''
    This function takes as input the directory of a sqlite database where messages
    and categories data is stored and loads it in variables for modelling purposes

    Input:
    database_filepath - the directory of the sqlite database

    Output:
    X - the messages data contained in the sqlite database, e.g. the independent variable
    y - the categories values of the messages contained in the sqlite database, e.g. the dependent variable
    category_names - the names of the columns containing the categories values in the sqlite database
    '''

    # connect to sqlite database and read in the data
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('etl_data', engine)
    # store messages and categories values and labels in variables
    X = df.message.values
    Y = df.iloc[:,4:].values
    category_names = list(df.iloc[:,4:].columns)

    return X, Y, category_names


def tokenize(text):
    '''
    This function takes as input a string of text and outputs a list containing the single words
    in the input string

    Input:
    text - a string variable

    Output:
    out_tokens - a list containing all the single words from the string input
    '''

    # define regular expression to identify urls
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # get all urls
    urls = re.findall(url_regex, text)
    # remove all urls
    for url in urls:
        text = text.replace(url, '')

    # remove all punctuations
    text = re.sub(r"[^a-zA-Z0-9]", ' ', text)

    # tokenize normalized text
    tokens = word_tokenize(text.lower())

    # inizialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    out_tokens = [WordNetLemmatizer().lemmatize(token).strip() for token in tokens]

    return out_tokens


def build_model():
    '''
    This function builds a machine learning pipeline to process text data and predict
    a message category based on the categories available in the trainig data. It is
    composed of a series of estimators: CountVectorizer to count word occurences,
    TfidfTransformer to count word frequencies, and RandomForestClassifier to predict
    a message category_names. The pipeline best combination of parameters are chosen
    with cross-validation based on the best score

    Output:
    cv - a model that performs gridsearch on the pipeline over a set of determined parameters
    '''

    # create pipeline
    pipeline = Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultiOutputClassifier(RandomForestClassifier()))
                    ])

    # define parameters
    parameters = {
              'clf__estimator__n_estimators':[5,10,20]
             }

    # implement cross validation on pipeline parameters
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    This function takes as input the pipeline model, the test dataset and the different
    message categories and prints out precision, recall and the f1-score for
    each category and the overall accuracy

    Input:
    model - the pipeline model
    X_test - an array containg the independent variable in the test dataset
    Y_test - a dataframe containing the dependent variable in the test dataset
    category_names - a list with all the message categories
    '''

    # predict message categories
    Y_pred = model.predict(X_test)

    # print out precision, recall and f1-score for each category
    for i, column in enumerate(category_names):
        print(column)
        print(classification_report(Y_test[:,i], Y_pred[:,i]))

    # print out model accuracy
    print('Model Accuracy: {}'.format((Y_test==Y_pred).mean()))


def save_model(model, model_filepath):
    '''
    This function takes as input the pipeline model and the desired destination directory
    where a pickle file of the model will be saved

    Input:
    model - the pipeline model
    model_filepath - the directory where the pickle file of the model will be stored
    '''

    # open a pickle file in the desired directory and save the model
    pickle_out = open(model_filepath, 'wb')
    pickle.dump(model, pickle_out)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
