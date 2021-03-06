import json
import plotly
import pandas as pd
import re
import pickle
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Histogram
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

# define tokenize function
def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('etl_data', engine)

# load model
model = joblib.load("../models/Classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # top 5 message categories in training dataset
    top_5 = df.iloc[:,4:].sum(axis=0).sort_values(ascending=False)[:5]
    top_5_labels = [item.replace('_',' ') for item in list(top_5.index)]

    # calculate character length of each message
    msg_length = df['message'].apply(lambda x: len(x))

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data':[
                Bar(
                    x=top_5_labels,
                    y=top_5
                )
            ],

            'layout':{
                'title':'Top 5 Message Categories used for Training',
                'yaxis':{'title':'Count'},
                'xaxis':{'title':'Category'}
            }
        },
        {
            'data':[
                Histogram(
                    x=msg_length,
                    nbinsx=500
                )
            ],

            'layout':{
                'title':'Distribution of Message Length used for Training',
                'yaxis':{'title':'Count'},
                'xaxis':{'title':'Length - N. of characters', 'range':[0, 600]}
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
