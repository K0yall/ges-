from itertools import combinations

import branca.colormap as cm
import numpy as np
import pandas as pd
import streamlit as st

from pyod.models.knn import KNN
from sqlalchemy import create_engine
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from yellowbrick.cluster import KElbowVisualizer
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from prophet import Prophet

# ============================================================
# CARREGAMENTO DA BASE DE DADOS
# ============================================================

@st.cache_data
def load_base():
    conexao = create_engine(st.secrets["url"])

    try:
        with conexao.connect() as cnx:
            df = pd.read_sql(
                sql="SELECT * FROM dw_europa",
                con=cnx
            )
    finally:
        conexao.dispose()

    dims = [
        "modal", "segmento", "cidade", "estado", "pais", "regiao",
        "sigla", "categoria", "subcategoria", "produto", "marca",
        "tipo", "ano", "mes", "dia"
    ]

    meds = ["venda", "quantidade", "lucro"]

    return df, dims, meds


# ============================================================
# DETECÇÃO DE ANOMALIAS
# ============================================================

def deteccao_anomalia(dataframe):
    if dataframe.empty:
        return np.array([], dtype=int)

    dados = dataframe.select_dtypes(include=np.number).copy()

    if dados.shape[1] == 0:
        raise ValueError(
            "A detecção de anomalias exige pelo menos uma coluna numérica."
        )

    dados = dados.replace([np.inf, -np.inf], np.nan)

    # Remove linhas com valores ausentes para evitar erros no modelo.
    mascara = dados.notna().all(axis=1)
    resultado = np.zeros(len(dataframe), dtype=int)

    dados_validos = dados.loc[mascara]

    if len(dados_validos) < 2:
        return resultado

    numero_vizinhos = min(5, len(dados_validos) - 1)

    clf = KNN(n_neighbors=numero_vizinhos)
    clf.fit(dados_validos)

    previsoes = clf.predict(dados_validos)

    # 0 = normal; 1 = anomalia.
    resultado[mascara.to_numpy()] = previsoes

    return resultado


# ============================================================
# ANÁLISE RFM
# ============================================================

def rfm(dataframe, dimensao):
    colunas_necessarias = {
        "data", "quantidade", "lucro", "venda", dimensao
    }

    faltantes = colunas_necessarias - set(dataframe.columns)

    if faltantes:
        raise ValueError(
            f"Colunas ausentes na análise RFM: {sorted(faltantes)}"
        )

    df = dataframe.copy()

    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df = df.dropna(subset=["data", dimensao])

    if df.empty:
        return pd.DataFrame(
            columns=[dimensao, "data", "quantidade", "lucro", "venda"]
        )

    ultima_data = df["data"].max()

    resultado = df.groupby(dimensao).agg(
        data=("data", "max"),
        quantidade=("quantidade", "count"),
        lucro=("lucro", "sum"),
        venda=("venda", "sum")
    ).reset_index()

    # Recência: número de dias desde a última compra.
    resultado["data"] = (
        ultima_data - resultado["data"]
    ).dt.days

    return resultado


# ============================================================
# ESCOLHA DO NÚMERO DE CLUSTERS
# ============================================================

def numero_cluster(dataframe):
    dados = dataframe.select_dtypes(include=np.number).copy()

    dados = dados.replace([np.inf, -np.inf], np.nan)
    dados = dados.dropna()

    if len(dados) < 3:
        return 1

    if dados.shape[1] == 0:
        raise ValueError(
            "O agrupamento exige pelo menos uma coluna numérica."
        )

    # Não é possível testar mais clusters do que amostras.
    k_max = min(12, len(dados) - 1)

    if k_max < 2:
        return 1

    modelo = KMeans(
        random_state=0,
        n_init=10
    )

    visualizer = KElbowVisualizer(
        modelo,
        k=(2, k_max),
        force_model=True
    )

    visualizer.fit(dados)

    if visualizer.elbow_value_ is None:
        return 2

    return visualizer.elbow_value_


# ============================================================
# AGRUPAMENTO COM K-MEANS
# ============================================================

def clusterizacao(dataframe):
    dados = dataframe.select_dtypes(include=np.number).copy()

    dados = dados.replace([np.inf, -np.inf], np.nan)

    if dados.shape[1] == 0:
        raise ValueError(
            "O agrupamento exige pelo menos uma coluna numérica."
        )

    if dados.isna().any().any():
        raise ValueError(
            "Existem valores ausentes. Trate-os antes de agrupar os dados."
        )

    if dados.empty:
        return np.array([], dtype=int)

    k = numero_cluster(dados)

    modelo = KMeans(
        n_clusters=k,
        random_state=0,
        n_init=10
    )

    modelo.fit(dados)

    return modelo.labels_


# ============================================================
# ANÁLISE DE COMPONENTES PRINCIPAIS (PCA)
# ============================================================

def principalcomponenteanalysis(dataframe):
    dados = dataframe.select_dtypes(include=np.number).copy()

    dados = dados.replace([np.inf, -np.inf], np.nan)

    if dados.isna().any().any():
        raise ValueError(
            "Existem valores ausentes. Trate-os antes de executar o PCA."
        )

    if len(dados) < 2 or dados.shape[1] < 2:
        raise ValueError(
            "O PCA com dois componentes exige pelo menos duas amostras "
            "e duas colunas numéricas."
        )

    pca = PCA(n_components=2)

    componentes = pca.fit_transform(dados)

    return componentes[:, 0], componentes[:, 1]


# ============================================================
# CENTROIDES DOS CLUSTERS
# ============================================================

def centroides(dataframe):
    dados = dataframe.select_dtypes(include=np.number).copy()

    dados = dados.replace([np.inf, -np.inf], np.nan)

    if dados.empty:
        raise ValueError(
            "Não existem dados suficientes para calcular os centroides."
        )

    if dados.isna().any().any():
        raise ValueError(
            "Existem valores ausentes. Trate-os antes de calcular os centroides."
        )

    k = numero_cluster(dados)

    modelo = KMeans(
        n_clusters=k,
        random_state=0,
        n_init=10
    )

    modelo.fit(dados)

    resultado = pd.DataFrame(
        modelo.cluster_centers_,
        columns=dados.columns
    )

    resultado.insert(
        0,
        "cluster",
        range(len(resultado))
    )

    return resultado


# ============================================================
# TREINAMENTO DE RECOMENDAÇÕES POR SIMILARIDADE
# ============================================================

def treino_similares(dataframe, numero):
    if numero < 1:
        raise ValueError(
            "O número de vizinhos deve ser maior que zero."
        )

    dados = dataframe.select_dtypes(include=np.number).copy()

    dados = dados.replace([np.inf, -np.inf], np.nan)

    if dados.isna().any().any():
        raise ValueError(
            "Existem valores ausentes. Trate-os antes de treinar o modelo."
        )

    if dados.empty:
        raise ValueError(
            "Não existem dados suficientes para treinar o modelo."
        )

    vizinhos = min(numero, len(dados))

    modelo = NearestNeighbors(
        n_neighbors=vizinhos
    )

    return modelo.fit(dados)


# ============================================================
# CORES DOS CLUSTERS
# ============================================================

def cores_cluster(s):
    colormap = cm.linear.Paired_11.scale(0, 12).to_step(12)

    cor = colormap(s["cluster"])

    return [
        f"background-color: {cor}"
    ] * len(s)


# ============================================================
# ANÁLISE DE VARIÂNCIA E TESTE DE TUKEY
# ============================================================

def analise_variancia(df, var, grp):
    colunas = [
        "grupo1",
        "grupo2",
        "diferente",
        "valor",
        "grupo"
    ]

    resultado_vazio = pd.DataFrame(columns=colunas)

    if var not in df.columns or grp not in df.columns:
        return resultado_vazio

    dados = df[[var, grp]].copy()

    dados[var] = pd.to_numeric(
        dados[var],
        errors="coerce"
    )

    dados = dados.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if dados[grp].nunique() < 2:
        return resultado_vazio

    try:
        tukey = pairwise_tukeyhsd(
            endog=dados[var],
            groups=dados[grp],
            alpha=0.05
        )

        tabela = pd.DataFrame(
            tukey._results_table.data[1:],
            columns=tukey._results_table.data[0]
        )

        resultado = pd.DataFrame({
            "grupo1": tabela["group1"],
            "grupo2": tabela["group2"],
            "diferente": tabela["reject"],
            "valor": tabela["meandiff"]
        })

        resultado["grupo"] = resultado.apply(
            lambda x: f"{x['grupo1']} {x['grupo2']}",
            axis=1
        )

        return resultado[colunas]

    except (ValueError, TypeError):
        return resultado_vazio


# ============================================================
# CLASSIFICAÇÃO ESTATÍSTICA
# ============================================================

def classes(base, variavel):
    if variavel not in base.columns:
        raise ValueError(
            f"A coluna '{variavel}' não existe na base."
        )

    base = base.copy()

    base[variavel] = pd.to_numeric(
        base[variavel],
        errors="coerce"
    )

    base[variavel] = base[variavel].replace(
        [np.inf, -np.inf],
        np.nan
    )

    base = base.dropna(subset=[variavel])

    if base.empty:
        raise ValueError(
            "Não existem valores válidos para realizar a classificação."
        )

    q1 = base[variavel].quantile(0.25)
    q3 = base[variavel].quantile(0.75)

    iqr = q3 - q1

    out_min = q1 - (1.5 * iqr)
    out_max = q3 + (1.5 * iqr)


    media = base[variavel].mean()
    desvio = base[variavel].std()

    erro_padrao = (
        1.96 * desvio / np.sqrt(len(base))
        if len(base) > 1
        else 0
    )

    limite_inferior = media - erro_padrao
    limite_superior = media + erro_padrao

    base["outlier_minimo"] = base[variavel] < out_min
    base["outlier_maximo"] = base[variavel] > out_max

    base["posicao"] = np.select(
        [
            base[variavel] < limite_inferior,
            base[variavel] > limite_superior
        ],
        [
            "abaixo",
            "acima"
        ],
        default="media"
    )

    # Evita divisão por zero quando todos os valores são iguais.
    if pd.isna(desvio) or desvio == 0:
        base["estrelas"] = 3
    else:
        base["estrelas"] = (
            ((base[variavel] - media) / desvio)
            .round()
            .add(3)
        )

    base["posicao_final"] = np.select(
        [
            base["outlier_maximo"],
            base["posicao"].eq("acima"),
            base["posicao"].eq("media"),
            base["outlier_minimo"]
        ],
        [
            "outlier maximo",
            "acima da media",
            "na media",
            "outlier minimo"
        ],
        default="abaixo da media"
    )

    base["relacao_media"] = base[variavel] - media

    base["positivo"] = base["relacao_media"] >= 0

    return (
        base,
        limite_inferior,
        limite_superior,
        out_min,
        out_max
    )


# ============================================================
# CORES DAS CLASSIFICAÇÕES ESTATÍSTICAS
# ============================================================

def cor_classe(s):
    if s["outlier_maximo"]:
        cor = "#686de0"

    elif s["posicao"] == "acima":
        cor = "#badc58"

    elif s["posicao"] == "media":
        cor = "#f6e58d"

    elif s["outlier_minimo"]:
        cor = "#ff7979"

    else:
        cor = "#ffbe76"

    return [
        f"background-color: {cor}"
    ] * len(s)


# ============================================================
# CORES DA CURVA ABC
# ============================================================

cores_abc = {
    "A": "#badc58",
    "B": "#f6e58d",
    "C": "#ffbe76"
}


# ============================================================
# CURVA ABC
# ============================================================

def curvaabc(abc, medida, dimensao):
    if medida not in abc.columns:
        raise ValueError(
            f"A medida '{medida}' não existe na base."
        )

    if dimensao not in abc.columns:
        raise ValueError(
            f"A dimensão '{dimensao}' não existe na base."
        )

    resultado = abc.copy()

    resultado[medida] = pd.to_numeric(
        resultado[medida],
        errors="coerce"
    )

    resultado[medida] = resultado[medida].replace(
        [np.inf, -np.inf],
        np.nan
    )

    resultado = resultado.dropna(subset=[medida])

    resultado = resultado.sort_values(
        by=medida,
        ascending=False
    ).copy()

    total = resultado[medida].sum()

    if resultado.empty:
        resultado["percentual"] = pd.Series(dtype=float)
        resultado["percentual_acumulado"] = pd.Series(dtype=float)
        resultado["acumulado"] = pd.Series(dtype=float)
        resultado["posicao"] = pd.Series(dtype=str)
        resultado["posicao_final"] = pd.Series(dtype=str)

        return resultado

    if total == 0:
        resultado["percentual"] = 0.0
        resultado["percentual_acumulado"] = 0.0
    else:
        resultado["percentual"] = resultado[medida] / total

        resultado["percentual_acumulado"] = (
            resultado["percentual"].cumsum()
        )

    resultado["acumulado"] = resultado[medida].cumsum()

    resultado["posicao"] = np.select(
        [
            resultado["percentual_acumulado"] <= 0.65,
            resultado["percentual_acumulado"] <= 0.90
        ],
        [
            "A",
            "B"
        ],
        default="C"
    )

    resultado["posicao_final"] = np.where(
        resultado["posicao"] == "A",
        "Outros",
        resultado[dimensao]
    )

    return resultado


# ============================================================
# ESTILIZAÇÃO DA CURVA ABC
# ============================================================

def cor_abc(s):
    posicao = s["posicao"]

    cor = cores_abc.get(
        posicao,
        "#ffbe76"
    )

    return [
        f"background-color: {cor}"
    ] * len(s)


# ============================================================
# CORES DAS POSIÇÕES ESTATÍSTICAS
# ============================================================

cores_posicao_final = {
    "outlier maximo": "#686de0",
    "acima da media": "#badc58",
    "na media": "#f6e58d",
    "abaixo da media": "#ffbe76",
    "outlier minimo": "#ff7979"
}


def serietemporal(dataframe, dimensao, medida, periodos=30):
    """
    Gera uma projeção de série temporal usando o Prophet para cada categoria da dimensão.
    """
    resultados = []

    # 1. Agrupa os dados por data e pela dimensão (ex: 'modal')
    df_agrupado = dataframe.groupby(["data", dimensao], as_index=False)[medida].sum()

    # 2. Faz a previsão individual para cada categoria (ex: 'First Class', 'Standard Class')
    for categoria in df_agrupado[dimensao].unique():
        df_cat = df_agrupado[df_agrupado[dimensao] == categoria].copy()

        # Prepara os dados
        df_prophet = df_cat.rename(columns={"data": "ds", medida: "y"})
        df_prophet["ds"] = pd.to_datetime(df_prophet["ds"], errors="coerce")
        df_prophet = df_prophet.dropna(subset=["ds", "y"])

        if len(df_prophet) < 2:
            continue # Salta se não houver histórico suficiente

        # Treina o modelo
        modelo = Prophet(daily_seasonality=True)
        modelo.fit(df_prophet)

        # Prevê o futuro
        futuro = modelo.make_future_dataframe(periods=periodos)
        previsao = modelo.predict(futuro)

        # Junta o histórico
        previsao = previsao.merge(df_prophet[['ds', 'y']], on='ds', how='left')

        # Renomeia as colunas
        previsao = previsao.rename(columns={
            'y': 'historico',
            'yhat': 'previsto',
            'ds': 'data'
        })

        # 3. Readiciona a coluna da dimensão com o nome da categoria para o pivot_table funcionar
        previsao[dimensao] = categoria
        resultados.append(previsao)

    # 4. Junta todas as categorias numa única tabela final
    if resultados:
        tabela_final = pd.concat(resultados, ignore_index=True)
    else:
        tabela_final = pd.DataFrame()

    return tabela_final