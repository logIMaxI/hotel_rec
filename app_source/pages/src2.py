import app_source.utils as utils
import streamlit as st
import os


def build_analitics():
    """
    Function for viewing plots
    """
    correct_config = utils.from_pickle('../../srcs/user_conditions.pkl')
    utils.plot_content(correct_config)

    list_of_content = os.listdir('../../srcs/')
    for path in list_of_content:
        try:
            st.image(path)
        except:
            print(f"Fatal error! Doesn't open {path}")

    st.button(
        label='Find hotels!',
        on_click=back_to_choice(list_of_content)
    )


def back_to_choice(list_of_content):
    """
    Function for back user to initial page

    Parameters:
    ----------
    list_of_content : list[str]
        List with paths of files to delete
    """
    utils.destroy_plots(list_of_content)
    st.switch_page('app_source/src.py')
