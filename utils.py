import os
import pickle

import pandas
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

HOTELS = pd.read_csv('hotels.csv')
HOTELS = HOTELS[HOTELS['Price'] > 0]
HOTELS = HOTELS[HOTELS['AreaSquare'] > 0]

MULTIPLE_CHOICES = {
    'Country name': {
        'options': HOTELS['countyName'].unique().tolist(),
        'name': 'countyName'
    },
    'City name': {
        'options': HOTELS['cityName'].unique().tolist(),
        'name': 'cityName'
    },
    'Rating': {
        'options': HOTELS['HotelRating'].unique().tolist(),
        'name': 'HotelRating'
    },
    'Days': {
        'options': [
            'less than 7',
            'from 7 under 14',
            'from 14 under 21',
            'more than 21'
        ],
        'name': 'AvailDays'
    },
    'Count of rooms': {
        'options': [1, 2, 3, 4],
        'name': 'RoomsCount'
    },
    'Count of floors': {
        'options': [0, 1, 2],
        'name': 'FloorsCount'
    },
    'Price': {
        'options': [
            'less than 200',
            'from 200 under 400',
            'from 400 under 600',
            'more than 600'
        ],
        'name': 'Price'
    },
    'Squares': {
        'options': [
            'less than 40',
            'from 40 under 70',
            'from 70 under 100',
            'more than 100'
        ],
        'name': 'AreaSquare'
    },
    'Count of persons': {
        'options': [1, 2, 3, 4, 5],
        'name': 'PersCount'
    }
}


def correct_config(config):
    """
    Function for correcting users condition: drop non-usable tags

    Parameters:
    ----------
    config : dict[str, list|str]
        Dictionary of tags which symbolized user's filters

    Returns:
    -------
    correct_config : dict[str, list|str]
        Corrected user's filters
    """
    correct_config = {}
    for key, item in config.items():
        if len(item) != 0:
            correct_config[key] = item

    return correct_config


def plot_content(config):
    """
    Function for building plots

    Parameters:
    ----------
    config : dict[str, list|str]
        Corrected user's filters

    Returns:
    -------
    cur_hotels : pd.DataFrame
        Filtered dataframe with info about hotels
    """
    bucket_attributes = [
        'AvailDays',
        'Price',
        'AreaSquare'
    ]
    embedding_attributes = [
        'Description',
        'HotelFacilities'
    ]

    cur_hotels = deepcopy(HOTELS)
    for col in cur_hotels.columns:
        if col in config.keys():
            if col in bucket_attributes:
                tmp = config[col][0].split(' ')
                if tmp[0] == 'more':
                    cur_hotels = cur_hotels[cur_hotels[col] > float(tmp[-1])]

                elif tmp[0] == 'less':
                    cur_hotels = cur_hotels[cur_hotels[col] < float(tmp[-1])]

                else:
                    cur_hotels = cur_hotels[
                        (cur_hotels[col] < float(tmp[-1])) &
                        (cur_hotels[col] > float(tmp[1]))
                        ]

            elif col not in embedding_attributes:
                cur_hotels = cur_hotels[cur_hotels[col].isin(config[col])]

    plt_number = 1
    diagram_cols = [
        'HotelRating', 'AvailDays', 'RoomsCount', 'FloorsCount',
        'Price', 'AreaSquare', 'PersCount', 'countyName'
    ]

    for col in diagram_cols:
        if col not in config and col not in bucket_attributes:
            path = f'srcs/plot{plt_number}.png'
            # colors_count = [
            #     len(elem[0]) for elem in MULTIPLE_CHOICES.items()
            #     if elem[1] == col
            # ]
            # colors = sns.color_palette('pastel')[0:colors_count[0]]
            tmp = pd.DataFrame(
                cur_hotels[col].value_counts()).reset_index()
            ax = plt.subplot()
            ax.pie(
                x=tmp.iloc[:, 1],
                labels=tmp.iloc[:, 0],
                # colors=colors,
                autopct='%.1f%%'
            )
            ax.set_title(f'{col}')
            plt.savefig(path)
            plt.close()

            plt_number += 1

    return cur_hotels


def destroy_plots(list_of_content):
    """
    Function for deleting built plots

    Parameters:
    ----------
    list_of_content : list
        List with paths of deleting files
    """
    for path in list_of_content:
        try:
            os.remove('srcs/' + path)
        except:
            print(f"Fatal error! Doesn't delete {'srcs/' + path}")


def to_pickle(path, data):
    """
    Function for writing dict-formed data

    Parameters:
    ----------
    path : str
        Path to written file

    data : dict[str, list|str]
        Corrected user's filters
    """
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def from_pickle(path):
    """
    Function for reading data from pickle

    Parameters:
    ----------
    path : str
        Path to read file

    Returns:
    -------
    dict[str, list|str]
        Corrected user's filters
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def to_parquet(df: pandas.DataFrame, path: str):
    """
    Function for writing df into parquet file

    Parameters:
    ----------
    df : pandas.DataFrame
        Dataframe for writing

    path : str
        path for writing
    """
    df.to_parquet(path)
