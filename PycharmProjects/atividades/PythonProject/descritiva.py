import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_folium import st_folium
import folium

from function import load_base
from function import analise_variancia, classes, cores_posicao_final, cor_classe, curvaabc, cores_abc, cor_abc

st.session_state['dw'], st.session_state['dims'], st.session_state['meds'] = load_base()
with st.container(horizontal=True):
    st.badge(
        label="Data Analysis",
        color="orange",
        icon=":material/query_stats:"
    )
    st.badge(
        label="Descriptive Analysis",
        color="green",
        icon=":material/query_stats:"
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
    base = st.session_state['dw'].groupby(dimensao)[medida].sum().reset_index()
    if len(base[dimensao]) < 10:
        st.badge(
            label="Análise de Variância (ANOVA)",
            icon=":material/query_stats:",
            color="primary"
        )
        anova = (
            analise_variancia(
                st.session_state['dw'],
                medida,
                dimensao
            )
        )
        if anova.empty:
            st.warning("A dimensão selecionada possui apenas 1 grupo. Selecione outra dimensão (como 'modal' ou 'segmento') para calcular a ANOVA.")
        else:
            cols = st.columns(3)
            cols[0].plotly_chart(
                px.bar(
                    base,
                    x=dimensao,
                    y=medida
                )
            )
            cols[1].plotly_chart(
                px.bar(
                    anova,
                    x='grupo',
                    y='valor',
                    color='diferente',
                    color_discrete_map={
                        False: "green",
                        True: "red"
                    }
                )
            )
            cols[2].dataframe(
                anova[['grupo', 'valor', 'diferente']],
                hide_index=True
            )
    elif len(base[dimensao]) < 100:
        st.badge(
            label="Intervalo de Confiança da Média",
            icon=":material/query_stats:",
            color="primary"
        )
        classe, li, ls, fl, fu = classes(base, medida)
        opcao = st.pills(
            label='Visualização',
            options=['Tabela', 'Classes', 'Média', 'Faixas', 'Mapa'],
            selection_mode='single',
            default='Tabela'
        )
        if opcao == 'Tabela':
            st.dataframe(
                classe.style.apply(cor_classe, axis=1),
                hide_index=True,
                column_config={
                    "estrelas": st.column_config.ProgressColumn(
                        label="estrelas",
                        format="⭐%f",
                        min_value=0,
                        max_value=7,
                    )
                }
            )
        if opcao == 'Classes':
            st.plotly_chart(
                px.bar(
                    base.sort_values(by=medida, ascending=False),
                    x=dimensao,
                    y=medida,
                    color='posicao_final',
                    color_discrete_map=cores_posicao_final
                )
            )
        if opcao == 'Média':
            st.plotly_chart(
                px.bar(
                    base.sort_values(by=medida, ascending=False),
                    x=dimensao,
                    y=['relacao_media'],
                    color='positivo',
                    color_discrete_map={
                        True: "green", False: "red"
                    }
                )
            )
        if opcao == 'Faixas':
            if base[medida].min() < 0:
                intervals = [[base[medida].min(), fl], [fl, li], [li, ls], [ls, fu], [fu, base[medida].max()]]
            else:
                if fl < 0:
                    intervals = [[0, fl], [0, li], [li, ls], [ls, fu], [fu, base[medida].max()]]
                else:
                    intervals = [[0, fl], [fl, li], [li, ls], [ls, fu], [fu, base[medida].max()]]
            fig = px.bar(
                base.sort_values(by=medida, ascending=False),
                x=dimensao,
                y=medida
            )
            for i in range(0, 5):
                if intervals[i][0] < intervals[i][1]:
                    fig.add_hrect(
                        y0=intervals[i][0],
                        y1=intervals[i][1],
                        line_width=1,
                        fillcolor=st.secrets.colors[i],
                        opacity=0.5,
                        layer='below',
                        annotation_text=st.secrets.tags[i],
                        annotation_position="bottom right",
                        annotation_font_size=14,
                        annotation_font_color="black"
                    )
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=False)
            st.plotly_chart(fig)
        if opcao == 'Mapa':
            if dimensao in st.secrets.geografico:
                geografico = st.session_state['dw'].groupby(dimensao)[['latitude', 'longitude']].mean().reset_index()
                base = pd.merge(base, geografico, on=dimensao)
                mapregiao = folium.Map(
                    location=[base['latitude'].mean(), base['longitude'].mean()],
                    tiles="cartodbpositron",
                    zoom_start=4,
                )
                for index, row in base.iterrows():
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        popup=f"Dimensão: {row[dimensao]} Medida: {row[medida]}",
                        radius=row['estrelas']+2,
                        color=cores_posicao_final[row['posicao_final']]
                    ).add_to(mapregiao)
                st_folium(mapregiao, height=600, width=800)
            else:
                st.error('Dimensão não é geográfica')
    else:
        st.badge("Curva ABC", icon=":material/query_stats:", color="primary")
        abc = curvaabc(base, medida, dimensao)
        opcao = st.pills(
            label='Visualização',
            options=['Tabela', 'Pizza', 'Árvore', 'Percentual Acumulado', 'Mapa'],
            selection_mode='single',
            default='Tabela'
        )
        if opcao == 'Tabela':
            st.dataframe(
                abc.style.apply(cor_abc, axis=1),
                hide_index=True
            )
        elif opcao == 'Pizza':
            st.plotly_chart(
                px.sunburst(
                    abc,
                    values=medida,
                    path=['posicao', dimensao]
                )
            )
        elif opcao == 'Árvore':
            st.plotly_chart(
                px.treemap(
                    abc,
                    values=medida,
                    path=['posicao', dimensao]
                )
            )
        elif opcao == 'Percentual Acumulado':
            st.plotly_chart(
                px.line(
                    abc,
                    x=dimensao,
                    y='percentual_acumulado',
                    color='posicao',
                    height=600
                )
            )
        if opcao == 'Mapa':
            if dimensao in st.secrets.geografico:
                geografico = st.session_state['dw'].groupby(dimensao)[['latitude', 'longitude']].mean().reset_index()
                abc = pd.merge(abc, geografico, on=dimensao)
                mapregiao = folium.Map(
                    location=[abc['latitude'].mean(), abc['longitude'].mean()],
                    tiles="cartodbpositron",
                    zoom_start=4,
                )
                for index, row in abc.iterrows():
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        popup=f"Dimensão: {row[dimensao]} Medida: {row[medida]}",
                        radius=row['percentual']*100,
                        color=cores_abc[row['posicao']]
                    ).add_to(mapregiao)
                st_folium(mapregiao, height=600, width=800)
            else:
                st.error('Dimensão não é geográfica')