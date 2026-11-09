import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_folium import st_folium
from scipy.stats import zscore
import folium
import altair as alt
import branca.colormap as cm

from function import load_base
from function import deteccao_anomalia, rfm, clusterizacao, principalcomponenteanalysis, cores_cluster, centroides, \
    treino_similares

st.session_state['dw'], st.session_state['dims'], st.session_state['meds'] = load_base()
with st.container(horizontal=True):
    st.badge(
        label="Data Analysis",
        color="orange",
        icon=":material/troubleshoot:"
    )
    st.badge(
        label="Diagnostic Analysis",
        color="green",
        icon=":material/troubleshoot:"
    )
dimensao = st.segmented_control(
    label='Dimensão',
    options=st.session_state['dims'],
    selection_mode='single'
)
st.divider()
if (dimensao is not None):
    base = st.session_state['dw'].groupby(dimensao)[st.session_state['meds']].sum().reset_index()
    if len(base[dimensao]) < 10:
        st.badge(
            label="Valores Padronizados (Z scores)",
            icon=":material/query_stats:",
            color="primary"
        )
        # Correção do erro do zscore aplicada aqui
        basez = zscore(base[st.session_state['meds']], nan_policy='omit')
        basez = pd.DataFrame(basez, columns=[f"{c}z" for c in st.session_state['meds']])

        basez['ranking'] = basez.sum(axis=1)
        base = pd.merge(
            base,
            basez,
            left_index=True,
            right_index=True
        )
        opcao = st.pills(
            label='Visualização',
            options=['Tabela', 'Graficos Individuais', 'Valores Padronizados'],
            selection_mode='single',
            default='Tabela'
        )
        if opcao == 'Tabela':
            st.dataframe(
                base.sort_values(by='ranking', ascending=False),
                hide_index=True
            )
        elif opcao == 'Graficos Individuais':
            cols = st.columns(len(st.session_state['meds']))
            c = 0
            for medida in st.session_state['meds']:
                cols[c].subheader(medida)
                cols[c].plotly_chart(
                    px.pie(
                        base,
                        names=dimensao,
                        values=medida,
                        color_discrete_sequence=px.colors.qualitative.Antique
                    )
                )
                c += 1
        elif opcao == 'Valores Padronizados':
            st.plotly_chart(
                px.bar(
                    base,
                    x=dimensao,
                    y=[f"{c}z" for c in st.session_state['meds']],
                    text_auto=True,
                    color_discrete_sequence=px.colors.qualitative.Antique
                )
            )
    elif len(base[dimensao]) < 100:
        st.badge(
            label="Análise RFM - Clusters/Detecção de Anomalias",
            icon=":material/query_stats:",
            color="primary"
        )
        base = rfm(
            st.session_state['dw'],
            dimensao
        )
        st.badge(
            label="Centróides",
            icon=":material/troubleshoot:",
            color="primary"
        )
        st.dataframe(
            centroides(base[st.secrets.rfm]).style.apply(cores_cluster, axis=1),
            hide_index=True
        )
        base['cluster'] = clusterizacao(base[st.secrets.rfm])
        base['outlier'] = deteccao_anomalia(base[st.secrets.rfm])
        base['escalaX'], base['escalaY'] = principalcomponenteanalysis(base[st.secrets.rfm])
        st.badge(
            label="Cluster (Grupos)",
            icon=":material/troubleshoot:",
            color="primary"
        )
        opcao = st.pills(
            label='Visualização',
            options=['Tabela', 'Plano Cartesiano', 'Mapa'],
            selection_mode='single',
            default='Tabela'
        )
        if opcao == 'Tabela':
            st.dataframe(
                base.style.apply(cores_cluster, axis=1),
                hide_index=True
            )
        elif opcao == 'Plano Cartesiano':
            ed = alt.Chart(base).mark_circle(size=100).encode(
                alt.X(shorthand='escalaX', scale=alt.Scale(zero=False)),
                alt.Y(shorthand='escalaY', scale=alt.Scale(zero=False, padding=1)),
                color='cluster:N',
            ).encode(tooltip=[dimensao]).properties(width=800, height=600)
            tx = alt.Chart(base).mark_text(dy=-10).encode(
                alt.X(shorthand='escalaX', scale=alt.Scale(zero=False)),
                alt.Y(shorthand='escalaY', scale=alt.Scale(zero=False, padding=1)),
                text=dimensao
            ).encode(tooltip=[dimensao]).properties(width=800, height=600)
            st.altair_chart(ed + tx)
        elif opcao == 'Mapa':
            if dimensao in st.secrets.geografico:
                geografico = st.session_state['dw'].groupby(dimensao)[['latitude', 'longitude']].mean().reset_index()
                base = pd.merge(base, geografico, on=dimensao)
                linear = cm.linear.Paired_11.scale(0, 12).to_step(12)
                linear.caption = 'Clusters (Grupos)'
                mapcluster = folium.Map(
                    location=[base['latitude'].mean(), base['longitude'].mean()],
                    tiles='cartodbpositron',
                    zoom_start=4,
                )
                linear.add_to(mapcluster)
                for idx, row in base.iterrows():
                    mapcluster.add_child(
                        folium.CircleMarker(
                            location=[row['latitude'], row['longitude']],
                            popup=f"Dimensao: {row[dimensao]} Cluster {row['cluster']}",
                            radius=3,
                            color=linear(row['cluster'])
                        )
                    )
                st_folium(mapcluster, height=600, width=800)
            else:
                st.write('Dimensão não é geográfica')
    else:
        st.badge(
            label="Análise RFM - Similares(KNN)",
            icon=":material/query_stats:",
            color="primary"
        )
        base = (rfm(st.session_state['dw'], dimensao))
        treino = treino_similares(base[st.secrets.rfm], numero=10)
        evento = st.dataframe(
            data=base,
            hide_index=True,
            on_select='rerun',
            selection_mode='single-row'
        )
        if len(evento['selection']['rows']) > 0:
            st.badge(
                label="Similares",
                icon=":material/troubleshoot:",
                color="primary"
            )
            dados = base[base.index == evento['selection']['rows'][0]]
            similar = treino.kneighbors(dados[st.secrets.rfm], return_distance=False)[0]
            st.dataframe(
                base[base.index.isin(similar)],
                hide_index=True
            )