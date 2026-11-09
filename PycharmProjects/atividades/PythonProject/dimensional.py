import streamlit as st
import plotly.express as px
from function import load_base

st.session_state['dw'], st.session_state['dims'], st.session_state['meds'] = load_base()
with st.container(horizontal=True):
    st.badge(label="Dimensional Schema", color="orange", icon=":material/pivot_table_chart:")
    st.badge(label="Cube", color="green", icon=":material/pivot_table_chart:")
dimensao = st.segmented_control(
    label='Dimensão',
    options=st.session_state['dims'],
    selection_mode='multi'
)
medida = st.segmented_control(
    label='Medida',
    options=st.session_state['meds'],
    selection_mode='single'
)
st.divider()
if (len(dimensao) > 0) & (len(dimensao) < 5) & (medida is not None):
    base = st.session_state['dw'].groupby(dimensao)[medida].sum().reset_index()
    opcao = st.pills(
        label='Visualização',
        options=[
            'Tabela',
            'Pizza',
            'Árvore'
        ],
        default='Tabela')
    if opcao == 'Tabela':
        st.dataframe(base)
    elif opcao == 'Pizza':
        st.plotly_chart(
            px.sunburst(
                base,
                values=medida,
                path=dimensao
            )
        )
    elif opcao == 'Árvore':
        st.plotly_chart(
            px.treemap(
                base,
                path=dimensao,
                values=medida
            )
        )