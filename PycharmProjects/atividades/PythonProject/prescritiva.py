import streamlit as st
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

from function import load_base
from function import serietemporal

import logging
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

st.session_state['dw'], st.session_state['dims'], st.session_state['meds'] = load_base()
with st.container(horizontal=True):
    st.badge(
        label="Data Analysis",
        color="orange",
        icon=":material/simulation:"
    )
    st.badge(
        label="Preditive Analysis",
        color="green",
        icon=":material/simulation:"
    )
dimensao = st.segmented_control(
    label='Dimensão',
    options=st.session_state['dims'],
    selection_mode='single'
)
percentual = st.segmented_control(
    label='Quantidade %',
    options=[30, 40, 50, 60, 70, 80, 90],
    selection_mode='single'
)
st.divider()
if (dimensao is not None) & (percentual is not None):
    st.badge(
        label="Resultados",
        icon=":material/troubleshoot:",
        color="primary")
    with st.spinner('Projeção de dados futuros (simulação)'):
        projecao = pd.DataFrame()
        for medida in st.session_state['meds']:
            ts = serietemporal(
                dataframe=st.session_state['dw'],
                dimensao=dimensao,
                medida=medida,
                periodos=1
            )
            ts['medida'] = medida
            if len(ts) > 0:
                projecao = pd.concat(
                    [
                        projecao,
                        ts[ts['historico'].isnull()]
                    ]
                )
    if len(projecao) > 0:
        st.badge(
            label="Valores Projetados",
            icon=":material/troubleshoot:",
            color="primary"
        )
        projecao_pivot = projecao.pivot_table(
            index=dimensao,
            columns='medida',
            values='previsto',
            aggfunc='sum'
        ).reset_index()
        st.dataframe(
            projecao_pivot,
            hide_index=True
        )
        cols = st.columns(len(st.session_state['meds']))
        c = 0
        for medida in st.session_state['meds']:
            cols[c].metric(
                label=f'Valor Projetado para {medida}',
                value=round(projecao_pivot[medida].sum(), 2),
                border=True
            )
            c += 1
        with st.spinner('Calculando valores otimizados!'):
            valor_total = projecao_pivot['venda'].sum() - projecao_pivot['lucro'].sum()
            unidades = projecao_pivot['quantidade'].sum() * percentual / 100
            indice = list(projecao_pivot.index.values)
            quantidade = projecao_pivot['quantidade'].values
            lucro = projecao_pivot['lucro'].values
            valor = projecao_pivot['venda'].values
            model = pyo.ConcreteModel()
            model.x = pyo.Var(indice, within=pyo.Binary)
            x = model.x
            model.valores_constraint = pyo.Constraint(
                expr=sum(
                    [x[p] * valor[p] for p in indice]
                ) <= valor_total
            )
            model.quantidade_constraint = pyo.Constraint(
                expr=sum(
                    [x[p] * quantidade[p] for p in indice]
                ) <= unidades)
            model.objective = pyo.Objective(
                expr=sum(
                    [x[p] * lucro[p] for p in indice]
                ),
                sense=pyo.maximize
            )
            opt = SolverFactory(
                'glpk',
                executable=st.secrets.solver
            )
            results = opt.solve(model)
            solution = [int(pyo.value(model.x[p])) for p in indice]
        projecao_pivot['manter'] = solution
        st.badge(
            label=f"Valores Otimizados para {percentual}% em {dimensao}",
            icon=":material/troubleshoot:",
            color="primary"
        )
        cols = st.columns(len(st.session_state['meds']))
        c = 0
        for medida in st.session_state['meds']:
            cols[c].metric(
                label=f'Valor Projetado para {medida}',
                value=round(projecao_pivot[projecao_pivot['manter'] == 1][medida].sum(), 2),
                border=True
            )
            c += 1
        st.badge(
            label="Sugestão de Permanência",
            icon=":material/troubleshoot:",
            color="primary"
        )
        st.dataframe(
            projecao_pivot[projecao_pivot['manter'] == 1][[dimensao] + st.session_state['meds']],
            hide_index=True
        )
        st.badge(
            label="Sugtestão de Corte",
            icon=":material/troubleshoot:",
            color="primary"
        )
        st.dataframe(
            projecao_pivot[projecao_pivot['manter'] == 0][[dimensao] + st.session_state['meds']],
            hide_index=True
        )
    else:
        st.error('Não foi possível calcular!')