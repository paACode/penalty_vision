import streamlit as st

pages = [
    st.Page("pages/home.py", title="Overview"),
]

pg = st.navigation(pages)

pg.run()