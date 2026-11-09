import streamlit as st
import plotly.express as px
import altair as alt
from function import load_base

st.session_state['dw'], st.session_state['dims'], st.session_state['meds'] = load_base()
with st.container(horizontal=True):
    st.badge(
        label="EDA - Exploratory Data Amalysis",
        color="orange",
        icon=":material/pivot_table_chart:"
    )
    st.badge(
        label="Visual Reports",
        color="green",
        icon=":material/pivot_table_chart:"
    )
dimensao = st.segmented_control(
    label='Dimensão',
    options=st.session_state['dims'],
    selection_mode='single'
)
medida = st.segmented_control(
    label='Medida',
    options=st.session_state['meds'],
    selection_mode='single'
)
st.divider()
if (dimensao is not None) & (medida is not None):
    tipo = st.pills(
        label='Tipos de Visualização',
        options=st.secrets['tipos'],
        selection_mode='single'
    )
    if tipo == 'Distribuição':
        st.subheader(f"Distribuição - Histograma - {medida}")
        st.plotly_chart(
            px.histogram(
                st.session_state['dw'],
                x=medida
            )
        )
    elif tipo == 'Relacionamento':
            cols = st.columns(2)
            mediday = cols[0].segmented_control(
                label='Medida Y',
                options=st.session_state['meds'],
                selection_mode='single'
            )
            if mediday is not None:
                cols[0].subheader('Relacionamento - Pontos/Dispersão')
                cols[0].plotly_chart(
                    px.scatter(
                        st.session_state['dw'],
                        x=medida,
                        y=mediday,
                        color=dimensao
                    )
                )
            medidaz = cols[1].segmented_control(
                label='Medida Z',
                options=st.session_state['meds'],
                selection_mode='single'
            )
            if mediday is not None and medidaz is not None:
                cols[1].subheader('Relacionamento - Bolhas')
                cols[1].plotly_chart(
                    px.scatter(
                        st.session_state['dw'],
                        x=medida,
                        y=mediday,
                        size=medidaz,
                        color=dimensao
                    )
                )
    elif tipo == 'Composição':
        base = st.session_state['dw'].groupby(dimensao)[medida].sum().reset_index()
        tab1, tab2 = st.tabs(['Estático', 'Dinâmico'])
        grafico = tab1.pills(
            label='Selecione a Visualização',
            options=['Setor', 'Colunas'],
            selection_mode='single'
        )
        if grafico == 'Setor':
            tab1.subheader('Distribuição - Histograma')
            tab1.plotly_chart(
                px.pie(
                    st.session_state['dw'],
                    names=dimensao,
                    values=medida
                )
            )
        elif grafico == 'Colunas':
            tab1.plotly_chart(
                px.bar(
                    base,
                    x=dimensao,
                    y=medida
                )
            )
        grafico = tab2.pills(
            label='Selecione a Visualização',
            options=['Colunas 100%', 'Colunas Empilhadas'],
            selection_mode='single'
        )
        dimensaoy = tab2.segmented_control(
            label='Dimensão Y',
            options=[c for c in st.session_state['dims'] if c not in dimensao],
            selection_mode='single'
        )
        if dimensaoy is not None:
            base2 = st.session_state['dw'].groupby([dimensao] + [dimensaoy])[medida].sum().reset_index()
            if grafico == 'Colunas 100%':
                tab2.altair_chart(
                    alt.Chart(base2).mark_bar().encode(
                        y = alt.Y(f"sum({medida})").stack("normalize"),
                        x=f"{dimensao}:O",
                        color=dimensaoy
                    )
                )
            elif grafico == 'Colunas Empilhadas':
                tab2.altair_chart(
                    alt.Chart(base2).mark_bar().encode(
                        y = alt.Y(f"sum({medida})"),
                        x=f"{dimensao}:O",
                        color=dimensaoy
                    )
                )
    elif tipo == 'Comparação':
        grafico = st.pills(
            label='Selecione a Visualização',
            options=['Polar', 'Linha', 'Colunas'],
            selection_mode='single'
        )
        base = st.session_state['dw'].groupby(dimensao)[medida].sum().reset_index()
        dimensaoy = st.segmented_control(
            label='Dimensão Y',
            options=[c for c in st.session_state['dims'] if c not in dimensao],
            selection_mode='single'
        )
        base2 = st.session_state['dw'].groupby([dimensao] + [dimensaoy])[medida].sum().reset_index()
        if dimensaoy is not None:
            if grafico == 'Polar':
                st.plotly_chart(
                    px.line_polar(
                        data_frame=base2,
                        r=medida, theta=dimensao,
                        color=dimensaoy,
                        line_close=True
                    )
                )
            elif grafico == 'Linha':
                st.plotly_chart(
                    px.line(
                        data_frame=base2,
                        y=medida,
                        x=dimensao,
                        color=dimensaoy
                    )
                )
            elif grafico == 'Colunas':
                st.plotly_chart(
                    px.bar(
                        data_frame=base2,
                        y=medida,
                        x=dimensao,
                        color=dimensaoy,
                        barmode='group'
                    )
                )