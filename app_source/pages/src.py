import app_source.utils as utils
import streamlit as st


def build_ui():
    """
    Function for building initial UI with entering user's data
    """
    input_area = st.text_input(
        placeholder='Enter description (optional)', label=''
    )
    with st.container():
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
        utils.to_pickle('../../srcs/user_conditions.pkl', correct_config)

        cols[col_number] = st.button(
            label='Find hotels!',
            on_click=switch_page()
        )


def switch_page():
    """
    Function for switching between pages
    """
    st.switch_page('app_source/src2.py')
