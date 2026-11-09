import pandas as pd
import streamlit as st
from sqlalchemy import create_engine

with st.container(horizontal=True):
    st.badge(label="Data Engineering", color="orange", icon=":material/pivot_table_chart:")
    st.badge(label="ETL - Extract, Transform and Load", color="green", icon=":material/pivot_table_chart:")
if 'carga' not in st.session_state:
    st.session_state['carga'] = 0
if st.session_state['carga'] == 0:
    arquivo = st.file_uploader(
        label="Upload data",
        accept_multiple_files=False,
        type="xlsx"
    )
    if arquivo is not None:
        if st.button('Carregar a base'):
            st.session_state['base'] = pd.read_excel(arquivo)
            st.session_state['carga'] = 1
            st.rerun()
if st.session_state['carga'] == 1:
    st.subheader('Base de Dados')
    with st.expander('Visualizar Dados Originais'):
        st.dataframe(st.session_state['base'])
    colunas = pd.DataFrame()
    colunas['base'] = st.session_state['base'].columns
    colunas['banco'] = ''
    colunas['dimensao'] = ''
    with st.container(horizontal=True):
        st.badge(label="Tabela das Colunas por dimensão", color="orange", icon=":material/pivot_table_chart:")
        st.badge(label="Informe as colunas do banco e classifique a dimensão", color="green", icon=":material/pivot_table_chart:")
    dados = st.data_editor(
        colunas,
        hide_index=True,
        column_config={
            "base": st.column_config.TextColumn(label="Coluna na Base de Dados"),
            "banco": st.column_config.TextColumn(label="Coluna no Banco de Dados", max_chars=20),
            "dimensao": st.column_config.SelectboxColumn(label="dimensão", options=st.secrets['dimensao'] + st.secrets['outros'])
        },
        disabled=["base"]
    )
    if st.toggle('Processar Dimensões'):
        with st.spinner(text="Aguarde...", show_time=True):
            st.session_state['dimensoes'] = []
            for index, dim in enumerate(st.secrets['dimensao']):
                lista_base = list(dados[dados['dimensao'] == dim]['base'])
                lista_banco = list(dados[dados['dimensao'] == dim]['banco'])
                temporario = st.session_state['base'][lista_base].drop_duplicates()
                temporario.columns = lista_banco
                temporario = temporario.reset_index()
                temporario = temporario.reset_index()
                temporario = temporario.drop(columns=['index'])
                temporario = temporario.rename(columns={'level_0': 'id'})
                temporario['id'] = temporario['id'] + 1
                st.session_state['dimensoes'].insert(len(st.session_state['dimensoes']), temporario)
        with st.expander('Dados das Dimensões'):
            tabs = st.tabs(st.secrets['dimensao'])
            for index, dim in enumerate(st.secrets['dimensao']):
                with tabs[index]:
                    st.write(dim)
                    st.dataframe(st.session_state['dimensoes'][index])
        chaves = pd.DataFrame()
        chaves['dimensao'] = st.secrets['dimensao']
        chaves['base'] = ''
        chaves['banco'] = ''
        with st.container(horizontal=True):
            st.badge(label="Tabela das Chaves por dimensão", color="orange", icon=":material/pivot_table_chart:")
            st.badge(label="Informe as colunas das Chaves da Base e do Banco", color="green",
                     icon=":material/pivot_table_chart:")
        dados_chaves = st.data_editor(
            chaves,
            hide_index=True,
            column_config={
                "dimensao": st.column_config.TextColumn(label="Dimensao"),
                "base": st.column_config.TextColumn(label="Chave na Base de Dados", max_chars=20),
                "banco": st.column_config.TextColumn(label="Chave no Banco de Dados", max_chars=20),
            },
            disabled=["dimensao"]
        )
        if st.toggle('Processar Fatos'):
            lista_medida_base = list(dados[dados['dimensao'] == 'medida']['base'])
            lista_medida_banco = list(dados[dados['dimensao'] == 'medida']['banco'])
            lista_id_banco = [f"{i}_id" for i in st.secrets['dimensao']]
            st.session_state['fato'] = pd.DataFrame(columns=lista_medida_banco + lista_id_banco)
            with st.spinner(text="Aguarde...", show_time=True):
                for index, row in st.session_state['base'].iterrows():
                    linha = {}
                    for base, banco in zip(lista_medida_base, lista_medida_banco):
                        linha[banco] = row[base]
                    for idx, rw in dados_chaves.iterrows():
                        conteudo = st.session_state['dimensoes'][idx].loc[
                            st.session_state['dimensoes'][idx][rw['banco']] == row[rw['base']]
                        ].reset_index()
                        linha[f"{rw['dimensao']}_id"] = conteudo['id'].iloc[0]
                    st.session_state['fato'] = pd.concat([st.session_state['fato'], pd.DataFrame([linha])])
            with st.expander('Visualizar Fatos'):
                st.dataframe(st.session_state['fato'])
            with st.spinner(text="Salvando...", show_time=True):
                conexao = create_engine(st.secrets['url'])
                st.session_state['fato'].to_sql(
                    name='fato',
                    con=conexao,
                    if_exists='replace',
                    index=False
                )
                for indice, dim in enumerate(st.secrets['dimensao']):
                    st.session_state['dimensoes'][indice].to_sql(
                        name=f"dim_{dim}",
                        con=conexao,
                        if_exists='replace',
                        index=False
                    )
            st.success('Banco de Dados Salvo!')