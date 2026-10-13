# -*- coding: utf-8 -*-

"""
============================================================
DOWNLOAD DE VAZÃO - GEOGLOWS RFS V2
============================================================

Curso: Inteligência Artificial aplicada a Recursos Hídricos

Objetivo:
    Baixar séries retrospectivas de vazão do GEOGLOWS RFS V2.

O usuário deve informar manualmente o River ID (LINKNO)
correspondente ao trecho de rio de interesse.

Resoluções disponíveis:
    "daily"  -> vazão diária
    "hourly" -> vazão horária

Unidade da vazão:
    m³/s

============================================================
"""

import requests
import pandas as pd
from pathlib import Path
from io import StringIO


# ============================================================
# 1. CONFIGURAÇÕES DO USUÁRIO
# ============================================================

# ------------------------------------------------------------
# RESOLUÇÃO TEMPORAL
# ------------------------------------------------------------
#
# Opções:
#   "daily"  = série diária
#   "hourly" = série horária
#

RESOLUTION = "daily"


# ------------------------------------------------------------
# ESTAÇÕES / TRECHOS
# ------------------------------------------------------------
#
# Para cada estação informar:
#
# nome      = nome ou código da estação
# river_id  = River ID / LINKNO do GEOGLOWS
#
# Adicione quantas estações forem necessárias.
#
# Exemplo:
#
# ESTACOES = [
#     {
#         "nome": "Estacao_1",
#         "river_id": 640377423
#     },
#     {
#         "nome": "Estacao_2",
#         "river_id": 123456789
#     }
# ]
#

ESTACOES = [

    {
        "nome": "Estacao_1",
        "river_id": 640447698
    }

]


# ------------------------------------------------------------
# PASTA DE SAÍDA
# ------------------------------------------------------------

PASTA_SAIDA = Path("dados_geoglows")


# ------------------------------------------------------------
# API GEOGLOWS
# ------------------------------------------------------------

BASE_URL = "https://geoglows.ecmwf.int/api/v2"


# ============================================================
# 2. VERIFICAÇÕES INICIAIS
# ============================================================

if RESOLUTION not in ["daily", "hourly"]:

    raise ValueError(
        "RESOLUTION deve ser 'daily' ou 'hourly'."
    )


if len(ESTACOES) == 0:

    raise ValueError(
        "Nenhuma estação foi informada."
    )


for estacao in ESTACOES:

    if "nome" not in estacao:

        raise ValueError(
            "Todas as estações devem possuir um 'nome'."
        )

    if "river_id" not in estacao:

        raise ValueError(
            f"A estação {estacao['nome']} "
            "não possui 'river_id'."
        )


# Criar pasta de saída

PASTA_SAIDA.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. FUNÇÃO PARA BAIXAR A SÉRIE
# ============================================================

def baixar_serie_geoglows(river_id, resolution):

    """
    Baixa uma série retrospectiva de vazão do GEOGLOWS.

    Parâmetros
    ----------
    river_id : int
        River ID / LINKNO do trecho GEOGLOWS.

    resolution : str
        "daily" ou "hourly".

    Retorno
    -------
    pandas.DataFrame
        Dados retornados pela API.
    """

    # --------------------------------------------------------
    # Selecionar endpoint
    # --------------------------------------------------------

    if resolution == "daily":

        endpoint = "retrospectivedaily"

    elif resolution == "hourly":

        endpoint = "retrospectivehourly"

    else:

        raise ValueError(
            "Resolução deve ser 'daily' ou 'hourly'."
        )


    # --------------------------------------------------------
    # Construir URL
    # --------------------------------------------------------

    url = (
        f"{BASE_URL}/"
        f"{endpoint}/"
        f"{river_id}"
    )


    print("\nURL consultada:")

    print(url)


    # --------------------------------------------------------
    # Requisição
    # --------------------------------------------------------

    response = requests.get(
        url,
        params={"format": "csv"},
        timeout=180
    )


    # Verificar erro HTTP

    response.raise_for_status()


    # --------------------------------------------------------
    # Verificar resposta
    # --------------------------------------------------------

    if not response.text.strip():

        raise ValueError(
            "A API GEOGLOWS retornou uma resposta vazia."
        )


    # --------------------------------------------------------
    # Converter CSV para DataFrame
    # --------------------------------------------------------

    df = pd.read_csv(
        StringIO(response.text)
    )


    if df.empty:

        raise ValueError(
            "Nenhum dado foi retornado pelo GEOGLOWS."
        )


    return df


# ============================================================
# 4. FUNÇÃO PARA IDENTIFICAR AS COLUNAS
# ============================================================

def identificar_colunas(df):

    """
    Identifica automaticamente as colunas de data e vazão.
    """

    # --------------------------------------------------------
    # Coluna de data
    # --------------------------------------------------------

    date_candidates = [

        coluna

        for coluna in df.columns

        if (
            "date" in str(coluna).lower()
            or
            "time" in str(coluna).lower()
        )

    ]


    if date_candidates:

        date_col = date_candidates[0]

    else:

        # Caso não encontre pelo nome,
        # assume a primeira coluna.

        date_col = df.columns[0]


    # --------------------------------------------------------
    # Coluna de vazão
    # --------------------------------------------------------

    flow_candidates = [

        coluna

        for coluna in df.columns

        if (
            coluna != date_col

            and (

                "flow" in str(coluna).lower()

                or

                "streamflow" in str(coluna).lower()

                or

                "discharge" in str(coluna).lower()

            )

        )

    ]


    if flow_candidates:

        flow_col = flow_candidates[0]

    else:

        # Caso o nome seja diferente,
        # procura qualquer outra coluna.

        candidatos = [

            coluna

            for coluna in df.columns

            if coluna != date_col

        ]


        if not candidatos:

            raise ValueError(
                "Não foi possível identificar "
                "a coluna de vazão."
            )


        flow_col = candidatos[0]


    return date_col, flow_col


# ============================================================
# 5. PROCESSAR ESTAÇÕES
# ============================================================

print("\n")

print("=" * 65)

print("GEOGLOWS RFS V2")

print("DOWNLOAD DE VAZÃO RETROSPECTIVA")

print("=" * 65)


print(
    f"\nResolução selecionada: {RESOLUTION}"
)


print(
    f"Número de estações: {len(ESTACOES)}"
)


resumo = []


# ============================================================
# LOOP DAS ESTAÇÕES
# ============================================================

for i, estacao in enumerate(
    ESTACOES,
    start=1
):

    nome = str(
        estacao["nome"]
    )


    river_id = int(
        estacao["river_id"]
    )


    print("\n")

    print("=" * 65)

    print(
        f"ESTAÇÃO {i}/{len(ESTACOES)}"
    )

    print("=" * 65)


    print(
        f"\nNome:       {nome}"
    )


    print(
        f"River ID:   {river_id}"
    )


    print(
        f"Resolução:  {RESOLUTION}"
    )


    try:

        # ====================================================
        # 5.1 DOWNLOAD
        # ====================================================

        print(
            "\nBaixando série..."
        )


        df = baixar_serie_geoglows(
            river_id,
            RESOLUTION
        )


        print(
            f"\nRegistros recebidos: {len(df):,}"
        )


        print(
            "Colunas recebidas:"
        )

        print(
            df.columns.tolist()
        )


        # ====================================================
        # 5.2 IDENTIFICAR COLUNAS
        # ====================================================

        date_col, flow_col = (
            identificar_colunas(df)
        )


        print(
            f"\nColuna de data:  {date_col}"
        )


        print(
            f"Coluna de vazão: {flow_col}"
        )


        # ====================================================
        # 5.3 SELECIONAR E PADRONIZAR
        # ====================================================

        df = df[
            [
                date_col,
                flow_col
            ]
        ].copy()


        df.columns = [
            "Data",
            "Vazao_GEOGLOWS"
        ]


        # ====================================================
        # 5.4 CONVERTER DATA
        # ====================================================

        df["Data"] = pd.to_datetime(
            df["Data"],
            errors="coerce"
        )


        # ====================================================
        # 5.5 CONVERTER VAZÃO
        # ====================================================

        df["Vazao_GEOGLOWS"] = (
            pd.to_numeric(
                df["Vazao_GEOGLOWS"],
                errors="coerce"
            )
        )


        # ====================================================
        # 5.6 LIMPEZA
        # ====================================================

        # Remover registros sem data

        df = df.dropna(
            subset=["Data"]
        )


        # Ordenar cronologicamente

        df = df.sort_values(
            "Data"
        )


        # Remover datas duplicadas

        df = df.drop_duplicates(
            subset=["Data"],
            keep="first"
        )


        # Resetar índice

        df = df.reset_index(
            drop=True
        )


        # ====================================================
        # 5.7 ADICIONAR METADADOS
        # ====================================================

        df["Estacao"] = nome

        df["River_ID"] = river_id

        df["Resolucao"] = RESOLUTION


        # ====================================================
        # 5.8 ESTATÍSTICAS
        # ====================================================

        n_registros = len(df)


        n_nan = (
            df["Vazao_GEOGLOWS"]
            .isna()
            .sum()
        )


        data_inicio = (
            df["Data"].min()
        )


        data_fim = (
            df["Data"].max()
        )


        q_min = (
            df["Vazao_GEOGLOWS"]
            .min()
        )


        q_media = (
            df["Vazao_GEOGLOWS"]
            .mean()
        )


        q_max = (
            df["Vazao_GEOGLOWS"]
            .max()
        )


        # ====================================================
        # 5.9 MOSTRAR RESUMO
        # ====================================================

        print("\n")

        print("-" * 65)

        print("RESUMO DA SÉRIE")

        print("-" * 65)


        print(
            f"Período: "
            f"{data_inicio} até {data_fim}"
        )


        print(
            f"Número de registros: "
            f"{n_registros:,}"
        )


        print(
            f"Valores ausentes: "
            f"{n_nan:,}"
        )


        print(
            f"Vazão mínima: "
            f"{q_min:.3f} m³/s"
        )


        print(
            f"Vazão média: "
            f"{q_media:.3f} m³/s"
        )


        print(
            f"Vazão máxima: "
            f"{q_max:.3f} m³/s"
        )


        # ====================================================
        # 5.10 NOME DO ARQUIVO
        # ====================================================

        nome_seguro = (
            nome
            .strip()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
        )


        nome_arquivo = (

            f"GEOGLOWS_"
            f"{nome_seguro}_"
            f"{river_id}_"
            f"{RESOLUTION}.csv"

        )


        arquivo_saida = (

            PASTA_SAIDA
            /
            nome_arquivo

        )


        # ====================================================
        # 5.11 SALVAR CSV
        # ====================================================

        df.to_csv(

            arquivo_saida,

            index=False,

            sep=";",

            decimal=",",

            date_format="%Y-%m-%d %H:%M:%S",

            encoding="utf-8-sig"

        )


        print(
            f"\nArquivo salvo:"
        )

        print(
            arquivo_saida
        )


        # ====================================================
        # 5.12 ADICIONAR AO RESUMO
        # ====================================================

        resumo.append(
            {

                "Estacao": nome,

                "River_ID": river_id,

                "Resolucao": RESOLUTION,

                "Data_inicial": data_inicio,

                "Data_final": data_fim,

                "Numero_registros": n_registros,

                "Vazao_minima_m3s": q_min,

                "Vazao_media_m3s": q_media,

                "Vazao_maxima_m3s": q_max,

                "Valores_ausentes": n_nan,

                "Status": "OK"

            }
        )


    # ========================================================
    # ERRO
    # ========================================================

    except Exception as erro:

        print("\n")

        print(
            f"ERRO ao processar {nome}:"
        )

        print(
            erro
        )


        resumo.append(
            {

                "Estacao": nome,

                "River_ID": river_id,

                "Resolucao": RESOLUTION,

                "Data_inicial": None,

                "Data_final": None,

                "Numero_registros": 0,

                "Vazao_minima_m3s": None,

                "Vazao_media_m3s": None,

                "Vazao_maxima_m3s": None,

                "Valores_ausentes": None,

                "Status": f"ERRO: {erro}"

            }
        )


# ============================================================
# 6. GERAR TABELA RESUMO
# ============================================================

df_resumo = pd.DataFrame(
    resumo
)


arquivo_resumo = (

    PASTA_SAIDA
    /
    "resumo_estacoes_GEOGLOWS.csv"

)


df_resumo.to_csv(

    arquivo_resumo,

    index=False,

    sep=";",

    decimal=",",

    encoding="utf-8-sig"

)


# ============================================================
# 7. MOSTRAR RESUMO FINAL
# ============================================================

print("\n")

print("=" * 65)

print("RESUMO FINAL")

print("=" * 65)


print(
    df_resumo.to_string(
        index=False
    )
)


# ============================================================
# 8. FINALIZAÇÃO
# ============================================================

print("\n")

print("=" * 65)

print("PROCESSAMENTO CONCLUÍDO")

print("=" * 65)


print(
    f"\nArquivos salvos em:"
)

print(
    PASTA_SAIDA.resolve()
)


print(
    f"\nTabela resumo:"
)

print(
    arquivo_resumo.resolve()
)