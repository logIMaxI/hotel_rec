import utils
import streamlit as st
import os
from TfIdfTransformer import TfIdfTransformer


def main():
    """
    Function for starting web-application
    """
    build_ui()


def build_ui():
    """
    Function for building initial UI with entering user's data
    """
    with st.sidebar:
        with st.container():
            input_area = st.text_input(
                placeholder='Enter description (optional)', label=''
            )
            cols = st.columns(11)
            col_number = 0
            for option, values in utils.MULTIPLE_CHOICES.items():
                cols[col_number] = st.multiselect(
                    options=values['options'], placeholder=option, label=''
                )
                col_number += 1

            cols[col_number] = st.text_input(
                placeholder='Enter facilities (optional)', label=''
            )
            col_number += 1

            config = {
                utils.MULTIPLE_CHOICES[name]['name']: cols[i] for i, name
                in enumerate(utils.MULTIPLE_CHOICES.keys())
            }
            config['Description'] = input_area
            config['HotelFacilities'] = cols[-2]

            correct_config = utils.correct_config(config)

            cols[col_number] = st.button(
                label='Find hotels!',
                key=1
            )

    if cols[col_number]:
        switch_page(correct_config)
        return


def switch_page(config):
    """
    Function for switching between pages

    Parameters:
    ----------
    config : dict[str, list[str] | list[float]]
        Filters of user's search
    """
    utils.to_pickle('srcs/user_conditions.pkl', config)
    build_analitics()


def build_analitics():
    """
    Function for viewing plots
    """
    correct_config = utils.from_pickle('srcs/user_conditions.pkl')
    df = utils.plot_content(correct_config)

    list_of_content = os.listdir('srcs/')
    plot_cols = st.columns(len(list_of_content) + 1)
    for i, path in enumerate(list_of_content):
        if path[-4:] == '.png':
            try:
                plot_cols[i] = st.image('srcs/' + path)
            except Exception:
                print(f"Fatal error! Doesn't open {'srcs/' + path}")

    tfidf_transformer = TfIdfTransformer(utils.HOTELS)
    recommends_count = 5
    wish = ''
    if correct_config.get('Description', False):
        wish = correct_config['Description']

    if correct_config.get('HotelFacilities', False):
        if len(wish) == 0:
            wish = correct_config['HotelFacilities']

        else:
            wish += ' ' + correct_config['HotelFacilities']

    idx = tfidf_transformer.get_recommends(wish)
    sorted_df = df.iloc[idx].sort_values(by=['PersCount'], ascending=False)
    sorted_df = sorted_df.join(df, how='inner', lsuffix='', rsuffix='_df')
    right_cols = [col for col in sorted_df.columns if col[-3:] != '_df']
    sorted_df = sorted_df[right_cols]
    utils.to_parquet(sorted_df, 'srcs/recommendations.parquet')

    sorted_df = sorted_df.iloc[idx][:recommends_count]
    plot_cols[len(list_of_content)] = st.dataframe(sorted_df)

    plot_cols[len(list_of_content) + 1] = st.button(
        label='Renew filters',
        key=2
    )

    if plot_cols[len(list_of_content)]:
        back_to_choice(list_of_content)
        return


def back_to_choice(list_of_content):
    """
    Function for back user to initial page

    Parameters:
    ----------
    list_of_content : list[str]
        List with paths of files to delete
    """
    utils.destroy_plots(list_of_content)
    return


if __name__ == '__main__':
    main()
