# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

# load messages and categories datasets
def load_data(messages_filepath, categories_filepath):
    '''
    The function takes as input the directories of the messages and categories
    with respect to the working directory, loads the data contained in the filepaths
    and then merges the two dataframes together

    Input:
    messages_filepath - the directory of the messages file
    categories_filepath - the directory of the categories file

    Output:
    df - a pandas dataframe which is the result of the merging between the messages and categories dataframes
    '''

    # read in the datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge the two datasets
    df = messages.merge(categories, how='inner', on=['id'])

    return df


def clean_data(df):
    '''
    The function takes as input the result of load_data and performs one hot encoding
    on the categories column of the dataframe

    Input:
    df - a pandas dataframe which is the result of the merging between the messages and the categories dataframes

    Output:
    df - a cleaned pandas dataframe, with one hot encoding for the categories column
    '''

    # extract the different categories from the categories column into a new dataframe
    # and expand them into multiple columns
    categories = df.categories.str.split(';', expand=True)
    # take the first row of the new dataframe
    row = categories.iloc[0]
    # extract the column names from the first row
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    # convert category values to 0 and 1
    for column in categories:

        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # replace all possible values greater than 1 with 1
    categories = categories.apply(lambda x: [y if y<=1 else 1 for y in x])
    # drop categories column from input dataframe and join input dataframe with
    # one hot encoded categories dataframe
    df.drop(columns=['categories'], inplace=True)
    df = df.join(categories)

    # remove all duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filepath):
    '''
    This function takes as input a pandas dataframe and the directory of a sqlite database
    and saves the dataframe into the sqlite database

    Input:
    df - a pandas dataframes
    database_filepath - the directory of the sqlite database where the dataframe will be stored
    '''

    # create connection with the database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    # saave data into the database
    df.to_sql('etl_data', engine, index=False, if_exists='replace')



def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
