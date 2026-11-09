import streamlit as st

st.set_page_config(
    page_title="Gestão do Conhecimento",
    layout="wide" ,
    page_icon=":material/model_training:",
)

st.header("Knowledge Management")
st.divider()

pages = {
    "KNOWLEDGE MANAGEMENT": [],
    "Data Engeneering": [
        st.Page(page='etl.py', title='ETL - Extract Transform Load', icon=':material/pivot_table_chart:'),
        st.Page(page='dimensional.py', title='Dimensional Schema', icon=':material/pivot_table_chart:'),
        st.Page(page='exploratoria.py', title='EDA - Exploratory Data Analysis', icon=':material/pivot_table_chart:'),
        st.Page(page='avancado.py', title='Advanced Charts', icon=':material/pivot_table_chart:'),
    ],
    "Data Analysis": [
        st.Page(page='descritiva.py', title='Descriptive Analysis', icon=':material/query_stats:'),
        st.Page(page='diagnostica.py', title='Diagnostic Analysis', icon=':material/troubleshoot:'),
        st.Page(page='preditiva.py', title='Preditive Analysis', icon=':material/monitoring:'),
        st.Page(page='prescritiva.py', title='Prescritive Analysis', icon=':material/simulation:'),
    ],
    "AI for SQL/DB": [
        st.Page(page='inteligencia.py', title='Chat with Data', icon=':material/chat:'),
    ],
}

pg = st.navigation(
    pages,
    #position="top"
)

pg.run()