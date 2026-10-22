# -*- coding: utf-8 -*-
"""Regionalização de mínimas de referência — Q90/Q98.
Versão dividida a partir de Regionalizacao_CAMELS_Filtro_UF.py.
Arquivo independente, organizado em células # %% para o Spyder.
Não importa o outro script nem módulos próprios do projeto.
Qp é uma vazão de permanência diária. Não implementa Q7,10 ou mínimas anuais.
Resultados: aula 7_index_flood/resultados/minimas/<execução>/.
Regressões por cluster, com preditores configuráveis; os estados filtram a UF do posto.
"""
from __future__ import annotations

# %% 00 | CONFIGURAÇÕES E BIBLIOTECAS — execute primeiro
from pathlib import Path
from datetime import datetime
from dataclasses import asdict, dataclass
from typing import Optional, Sequence
import os
import json
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path as Caminho
from matplotlib.patches import PathPatch
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy.cluster.hierarchy import linkage, cut_tree, dendrogram
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score, adjusted_rand_score

# Cada script possui sua própria configuração: não lê resultados do outro.
PASTA = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
# O ZIP contém uma única pasta "aula 7_index_flood". Ao copiar apenas o .py
# para outra pasta, a subpasta da aula será criada ao lado do script.
PASTA_AULA = PASTA if PASTA.name == 'aula 7_index_flood' else PASTA / 'aula 7_index_flood'
# Para fixar outro endereço, altere SOMENTE PASTA_AULA, por exemplo:
# PASTA_AULA = Path(r'C:\\aula 7_index_flood')

PASTA_CAMELS = Path(
    r'C:\Users\afrod\OneDrive\Documents\HydroUAI_git'
    r'\hydrouai-epagri-2026\01_Aulas\07_index_flood_transfer_learning'
    r'\epagri\epagri\Info Bacias - CAMELS'
)
USAR_AMOSTRA_ANEXADA = False
if USAR_AMOSTRA_ANEXADA:
    PASTA_CAMELS = PASTA / 'amostra_camels'
ARQUIVO_ATRIBUTOS = PASTA_CAMELS / 'Caracteristicas_bacias_CAMELS.xlsx'
PASTA_VAZAO = PASTA_CAMELS / 'Vazão'
PASTA_PRECIPITACAO = PASTA_CAMELS / 'Precipitação'
BUSCAR_EM_SUBPASTAS = True

# Filtros preservados da última versão do script.
ANO_INICIAL_ANALISE = 1980
ANO_FINAL_ANALISE = 2024
ANOS_MINIMOS = 20
MAX_FALHAS_PERIODO_PCT = 10.0
MAX_FALHAS_ANO_PCT = 10.0
MES_INICIO_ANO_HIDROLOGICO = 1
FLAGS_VAZAO_ACEITAS = None
IDS_INCLUIR = None
ESTADOS_INCLUIR = None  # Ex.: ['SC'] ou ['PR', 'SC', 'RS']; UF do POSTO.

FONTE_PRECIPITACAO = 'serie'
PRODUTO_PRECIPITACAO = 'p_chirps'
ANOS_MINIMOS_PRECIPITACAO = 30
# 'serie': média dos totais de anos civis completos. 'atributo': p_mean * 365.2425.

# Definir os atributos adequados a ESTA análise. Não se impõe que mínimas
# e máximas tenham os mesmos grupos. Defaults preservam a última configuração.
ATRIBUTOS_CLUSTER = [
    'precip_mm_ano',
    'pet_mean',
    'slope_mean',
    'gauge_lat',
    'gauge_lon',
    "elev_gauge",
]
K = 5
METODO_REGIONAL = 'kmeans'  # Ambos são comparados; este escolhe os grupos das aplicações.
MIN_BACIAS_POR_GRUPO = 5   # Limite operacional; não garante homogeneidade.
SEMENTE = 42
BACIA_EXEMPLO = '60010000'
BACIA_ALVO = None
RESERVAR_BACIA_ALVO = True
FORMA_REGRESSAO = 'potencial_simples'
# Primeira variável = regressão simples; todas = regressão múltipla.
# Aceita nomes originais da planilha CAMELS e derivados (p.ex. area_km2).
# Para modelos potenciais, as variáveis selecionadas devem ser > 0.
VARIAVEIS_REGRESSAO = [
    'area_km2',
        # 'pet_mean',
]

# linear_simples, linear_multipla, potencial_simples, potencial_multipla.
# Simples: primeira variável. Múltipla: todas. Potenciais: log + smearing.
EXECUTAR_ETAPAS_REGIONAIS = True
EXECUTAR_VALIDACAO_ESPACIAL = True

MOSTRAR_GRAFICOS = True
ANOS_HIDROGRAMA = 3
DPI_FIGURAS = 160
PLOTAR_MAPAS = True
ARQUIVO_MAPA_BRASIL = PASTA / 'mapas' / 'brasil_uf_ibge.geojson'
MAPA_REGIOES_ZOOM = True
ROTULAR_BACIAS_NO_MAPA = False

TIPO_ANALISE = 'minimas'
PERMANENCIAS = [98]
COMPARAR_EQUACAO_GLOBAL = True  # Referência adicional; não substitui as equações regionais.

# %% 00A | FUNÇÕES DE LEITURA E SELEÇÃO CAMELS
@dataclass(frozen=True)
class ConfiguracaoSelecao:
    ano_inicial: int = 1980
    ano_final: int = 2024
    anos_minimos: int = 10
    max_falhas_periodo_pct: float = 10.0
    max_falhas_ano_pct: float = 0.0
    mes_inicio_ano_hidrologico: int = 1
    fonte_precip: str = 'serie'       # 'serie' ou 'atributo'; sem fallback
    coluna_precip: str = 'p_brdwgd'   # produto único para todas as bacias
    anos_minimos_precip: int = 10
    flags_vazao_aceitas: Optional[Sequence] = None
    ids_incluir: Optional[Sequence] = None
    recursivo: bool = True

    def validar(self):
        inteiros = ['ano_inicial', 'ano_final', 'anos_minimos',
                    'mes_inicio_ano_hidrologico', 'anos_minimos_precip']
        for nome in inteiros:
            valor = getattr(self, nome)
            if isinstance(valor, bool) or not isinstance(valor, (int, np.integer)):
                raise ValueError(f'{nome} deve ser inteiro.')
        if self.ano_inicial > self.ano_final:
            raise ValueError('ano_inicial deve ser menor ou igual a ano_final.')
        if self.anos_minimos < 1 or self.anos_minimos_precip < 1:
            raise ValueError('Os mínimos de anos devem ser positivos.')
        if self.mes_inicio_ano_hidrologico not in range(1, 13):
            raise ValueError('O mês de início deve estar entre 1 e 12.')
        for nome in ['max_falhas_periodo_pct', 'max_falhas_ano_pct']:
            valor = getattr(self, nome)
            if not np.isfinite(valor) or not 0 <= valor <= 100:
                raise ValueError(f'{nome} deve estar entre 0 e 100.')
        if self.max_falhas_ano_pct >= 100:
            raise ValueError('max_falhas_ano_pct deve ser menor que 100: um ano aceito precisa de observações.')
        if self.fonte_precip not in ['serie', 'atributo']:
            raise ValueError("fonte_precip deve ser 'serie' ou 'atributo'.")
        if not isinstance(self.coluna_precip, str) or not self.coluna_precip.strip():
            raise ValueError('Informe a coluna do produto de precipitação.')
        if self.ids_incluir is not None:
            if isinstance(self.ids_incluir, (str, bytes)):
                raise ValueError('ids_incluir deve ser uma lista de códigos, não texto único.')
            try:
                quantidade_ids = len(self.ids_incluir)
            except TypeError as exc:
                raise ValueError('ids_incluir deve ser uma lista de códigos ou None.') from exc
            if quantidade_ids == 0:
                raise ValueError('ids_incluir está vazia. Use None para avaliar todas as estações ou informe uma lista de códigos.')
        if isinstance(self.flags_vazao_aceitas, (str, bytes)):
            raise ValueError('flags_vazao_aceitas deve ser uma lista de códigos ou None, não texto único.')
        # Também verifica os limites de datas suportados pelo pandas.
        self.calendario()
        return self

    def calendario(self):
        datas = pd.date_range(f'{self.ano_inicial:04d}-01-01',
                              f'{self.ano_final:04d}-12-31', freq='D')
        return datas.rename('data')


def normalizar_id(valor):
    """Conservar códigos textuais e remover apenas a terminação numérica .0."""
    if pd.isna(valor):
        raise ValueError('Código de estação ausente.')
    if isinstance(valor, (int, np.integer)):
        return str(int(valor))
    if isinstance(valor, (float, np.floating)):
        if not np.isfinite(valor) or not float(valor).is_integer():
            raise ValueError(f'Código numérico inválido: {valor!r}')
        return str(int(valor))
    codigo = str(valor).strip()
    if not codigo:
        raise ValueError('Código de estação vazio.')
    return re.sub(r'^(\d+)\.0+$', r'\1', codigo)


def ler_atributos_camels(arquivo):
    """Unir as abas pela chave gauge_id; preservar atributos e suas fontes.

    Escolhas explícitas para o arquivo fornecido:
      area (topografia, km²) -> area_km2;
      slope_mean (m/km) / 10 -> declividade_pct;
      gauge_lat/gauge_lon -> latitude/longitude.
    area_ana e area_gsim são preservadas, mas não substituem area.
    """
    arquivo = Path(arquivo)
    if not arquivo.is_file():
        raise FileNotFoundError(f'Planilha de atributos não encontrada: {arquivo}')
    abas = pd.read_excel(arquivo, sheet_name=None, dtype=object, engine='openpyxl')
    quadros, fontes = [], {}
    for nome, quadro in abas.items():
        quadro = quadro.dropna(how='all').copy()
        quadro.columns = [str(c).strip() for c in quadro.columns]
        if 'gauge_id' not in quadro.columns:
            continue
        quadro['gauge_id'] = quadro['gauge_id'].map(normalizar_id)
        if quadro['gauge_id'].duplicated().any():
            repetidos = quadro.loc[quadro.gauge_id.duplicated(False), 'gauge_id'].unique()
            raise ValueError(f'IDs repetidos na aba {nome}: {list(repetidos)}')
        quadro = quadro.set_index('gauge_id')
        for coluna in quadro.columns:
            if coluna in fontes:
                raise ValueError(f'Coluna {coluna!r} em mais de uma aba; explicite a fonte antes de unir.')
            fontes[coluna] = nome
        quadros.append(quadro)
    if not quadros:
        raise ValueError('Nenhuma aba contém a coluna gauge_id.')
    atributos = pd.concat(quadros, axis=1, join='outer').sort_index()
    atributos.index.name = 'id_bacia'
    obrigatorias = ['area', 'slope_mean', 'p_mean', 'gauge_lat', 'gauge_lon']
    faltantes = [c for c in obrigatorias if c not in atributos]
    if faltantes:
        raise ValueError(f'Atributos CAMELS ausentes: {faltantes}')
    atributos['area_km2'] = pd.to_numeric(atributos['area'], errors='coerce')
    atributos['declividade_pct'] = pd.to_numeric(atributos['slope_mean'], errors='coerce') / 10
    atributos['latitude'] = pd.to_numeric(atributos['gauge_lat'], errors='coerce')
    atributos['longitude'] = pd.to_numeric(atributos['gauge_lon'], errors='coerce')
    atributos['precip_climatologica_mm_ano'] = pd.to_numeric(atributos['p_mean'], errors='coerce') * 365.2425
    atributos['precip_mm_ano'] = np.nan  # preenchida pela fonte selecionada
    meta = {
        'arquivo_atributos': str(arquivo),
        'abas_com_atributos': [nome for nome, q in abas.items() if 'gauge_id' in q.columns],
        'fontes_colunas': fontes,
        'coluna_area': 'area', 'unidade_area': 'km²',
        'conversao_declividade': 'slope_mean [m/km] / 10 = declividade_pct [%]',
        'conversao_precip_atributo': 'p_mean [mm/dia] * 365.2425 = mm/ano',
        'precip_atributo_observacao': 'Climatologia CHIRPS da planilha; não recalculada na janela escolhida.',
    }
    return atributos, meta


def _inventariar_arquivos(pasta, sufixo, recursivo=True):
    encontrados = {}
    if pasta is None or not Path(pasta).is_dir():
        return encontrados
    pasta = Path(pasta)
    caminhos = pasta.rglob('*') if recursivo else pasta.glob('*')
    padrao = re.compile(r'^(.+)' + re.escape(sufixo) + r'$', flags=re.I)
    for arquivo in caminhos:
        if not arquivo.is_file():
            continue
        match = padrao.fullmatch(arquivo.name)
        if match:
            codigo = normalizar_id(match.group(1))
            encontrados.setdefault(codigo, []).append(arquivo)
    return {codigo: sorted(lista) for codigo, lista in encontrados.items()}


def _codigo_flag(valor):
    if pd.isna(valor):
        return 'ausente'
    return normalizar_id(valor)


def ler_serie_diaria_camels(arquivo, coluna, flags_aceitas=None):
    """Ler uma coluna diária e devolver (série, diagnóstico da leitura).

    Datas inválidas/duplicadas invalidam o arquivo. Valores negativos,
    não numéricos e infinitos tornam-se NaN; zeros são mantidos. O filtro
    opcional de qualidade usa qual_flag, sem interpretar seus códigos.
    """
    arquivo = Path(arquivo)
    dados = pd.read_csv(arquivo, sep=r'\s+', encoding='utf-8-sig')
    exigidas = ['year', 'month', 'day', coluna]
    if any(c not in dados for c in exigidas):
        raise ValueError(f'Colunas necessárias: {exigidas}; encontradas: {list(dados.columns)}')
    if dados.empty:
        raise ValueError('Arquivo diário vazio.')
    partes = dados[['year', 'month', 'day']].apply(pd.to_numeric, errors='coerce')
    if partes.isna().any().any() or ((partes % 1) != 0).any().any():
        raise ValueError('Há componentes de data inválidos ou não inteiros.')
    datas = pd.to_datetime(partes, errors='coerce')
    if datas.isna().any():
        raise ValueError(f'Há {int(datas.isna().sum())} datas inválidas.')
    if datas.duplicated().any():
        exemplo = datas[datas.duplicated(False)].dt.strftime('%Y-%m-%d').unique()[:5]
        raise ValueError(f'Datas duplicadas; nenhuma linha foi escolhida: {list(exemplo)}')
    valores = pd.to_numeric(dados[coluna], errors='coerce').astype(float)
    finitos = np.isfinite(valores)
    negativos = finitos & (valores < 0)
    valores = valores.where(finitos & (valores >= 0))
    flags = dados['qual_flag'].map(_codigo_flag) if 'qual_flag' in dados else None
    descartadas = 0
    if flags_aceitas is not None:
        if flags is None:
            raise ValueError('Filtro solicitado, mas qual_flag não existe no arquivo.')
        permitidas = {_codigo_flag(x) for x in flags_aceitas}
        aceitas = flags.isin(permitidas)
        descartadas = int((valores.notna() & ~aceitas).sum())
        valores = valores.where(aceitas)
    serie = pd.Series(valores.to_numpy(), index=pd.DatetimeIndex(datas), name=coluna).sort_index()
    serie.index.name = 'data'
    diagnostico = {
        'linhas_arquivo': len(dados),
        'inicio_arquivo': serie.index.min().strftime('%Y-%m-%d'),
        'fim_arquivo': serie.index.max().strftime('%Y-%m-%d'),
        'n_negativos_arquivo': int(negativos.sum()),
        'n_nao_finitos_arquivo': int((~finitos).sum()),
        'n_descartadas_flag_arquivo': descartadas,
        'flags_no_arquivo': json.dumps(flags.value_counts().to_dict(), ensure_ascii=False) if flags is not None else '{}',
    }
    return serie, diagnostico


def cobertura_por_ano(serie, mes_inicio=1):
    """Frações observadas por ano hidrológico; denominador sempre 365/366.

    O rótulo é o ano de término quando o ano não começa em janeiro.
    Por exemplo, out/1999–set/2000 recebe o rótulo 2000.
    """
    if mes_inicio not in range(1, 13):
        raise ValueError('Mês de início inválido.')
    if not isinstance(serie.index, pd.DatetimeIndex) or serie.index.has_duplicates:
        raise ValueError('A série deve ter datas diárias únicas.')
    d = serie.index
    rotulos = d.year if mes_inicio == 1 else d.year + (d.month >= mes_inicio).astype(int)
    contagens = serie.groupby(rotulos).count()
    esperados = {}
    for ano in contagens.index:
        inicio = pd.Timestamp(int(ano) if mes_inicio == 1 else int(ano)-1, mes_inicio, 1)
        esperados[int(ano)] = int(((inicio + pd.DateOffset(years=1)) - inicio).days)
    cobertura = contagens / pd.Series(esperados)
    cobertura.index.name = 'ano_hidrologico'
    cobertura.name = serie.name
    return cobertura


def avaliar_serie_vazao(serie, config=None):
    """Devolver (resumo, série no calendário completo, cobertura anual).

    Um ano atende ao mínimo quando sua cobertura cumpre max_falhas_ano_pct.
    Os anos aceitos não precisam ser consecutivos. Percentuais nos limites
    são aceitos; não há preenchimento de falhas ou uso da Q da planilha.
    """
    cfg = (config or ConfiguracaoSelecao()).validar()
    if not isinstance(serie.index, pd.DatetimeIndex) or serie.index.has_duplicates:
        raise ValueError('A série deve possuir DatetimeIndex com datas únicas.')
    dados = pd.to_numeric(serie, errors='coerce').astype(float)
    dados = dados.where(np.isfinite(dados) & (dados >= 0))
    q = dados.reindex(cfg.calendario())
    cobertura = cobertura_por_ano(q, cfg.mes_inicio_ano_hidrologico)
    anos_aceitos = (cobertura > 0) & (100*(1-cobertura) <= cfg.max_falhas_ano_pct + 1e-10)
    n = len(q)
    validos = int(q.notna().sum())
    n_anos = int(anos_aceitos.sum())
    falhas = n-validos
    percentual = 100*falhas/n
    motivos = []
    if percentual > cfg.max_falhas_periodo_pct + 1e-10:
        motivos.append(f'{percentual:.4f}% de falhas no período > {cfg.max_falhas_periodo_pct:g}%')
    if n_anos < cfg.anos_minimos:
        motivos.append(f'{n_anos} anos aceitos < mínimo de {cfg.anos_minimos}')
    datas_validas = q.index[q.notna()]
    resumo = {
        'primeiro_dado_valido': datas_validas.min().strftime('%Y-%m-%d') if validos else '',
        'ultimo_dado_valido': datas_validas.max().strftime('%Y-%m-%d') if validos else '',
        'dias_periodo': n, 'dias_validos': validos, 'dias_falha': falhas,
        'falhas_periodo_pct': percentual,
        'anos_equivalentes_observados': validos/365.2425,
        'anos_validos': n_anos,
        'anos_aceitos': ';'.join(str(a) for a in cobertura.index[anos_aceitos]),
        'n_dias_sem_linha_no_periodo': len(q.index.difference(serie.index)),
        'selecionada_vazao': not motivos,
        'motivo_vazao': '; '.join(motivos) if motivos else 'Aprovada nos dois critérios de vazão',
    }
    return resumo, q, cobertura


def resumir_precipitacao(serie, config=None):
    """Média dos totais de anos CIVIS completos na janela de análise.

    Um ano com qualquer dia faltante não entra na média. A soma parcial
    não é tratada como total anual e não há extrapolação de dias faltantes.
    Retorna (resumo, totais anuais, cobertura anual).
    """
    cfg = (config or ConfiguracaoSelecao()).validar()
    p = pd.to_numeric(serie, errors='coerce').astype(float)
    p = p.where(np.isfinite(p) & (p >= 0)).reindex(cfg.calendario())
    cobertura = cobertura_por_ano(p, 1)
    totais = p.groupby(p.index.year).sum(min_count=1).where(cobertura == 1)
    totais.index.name = 'ano_civil'
    totais.name = 'precip_total_mm'
    n_anos = int(totais.notna().sum())
    media = float(totais.mean()) if n_anos else np.nan
    aprovada = n_anos >= cfg.anos_minimos_precip and np.isfinite(media) and media > 0
    return {
        'produto': cfg.coluna_precip,
        'anos_precip_completos': n_anos,
        'media_totais_anuais_mm': media,
        'falhas_precip_periodo_pct': float(100*p.isna().mean()),
        'precip_serie_aprovada': bool(aprovada),
        'motivo_precip_serie': ('Aprovada' if aprovada else
             f'Exige >= {cfg.anos_minimos_precip} anos civis completos e média anual positiva; disponíveis {n_anos}'),
        'anos_precip_usados': ';'.join(str(a) for a in totais.dropna().index),
    }, totais, cobertura


def preparar_base_camels(arquivo_atributos, pasta_vazao, pasta_precip, config=None,
                        estados_incluir=None, arquivo_malha=None):
    """Preparar entradas e relatórios sem gravar/modificar os arquivos originais.

    `vazoes_selecionadas` contém todos os postos aprovados pela vazão.
    `vazoes` e `atributos` contêm os postos também aptos em área, declividade
    e precipitação. Isso evita confundir ausência de atributo com falha na Q.
    """
    cfg = (config or ConfiguracaoSelecao()).validar()
    if not Path(pasta_vazao).is_dir():
        raise FileNotFoundError(f'Pasta de vazão não encontrada: {pasta_vazao}')
    atributos, metadados = ler_atributos_camels(arquivo_atributos)
    arquivos_q = _inventariar_arquivos(pasta_vazao, '_streamflow.txt', cfg.recursivo)
    arquivos_p = _inventariar_arquivos(pasta_precip, '_precipitation.txt', cfg.recursivo)
    universo = sorted(set(atributos.index) | set(arquivos_q) | set(arquivos_p))
    if cfg.ids_incluir is not None:
        universo = sorted({normalizar_id(x) for x in cfg.ids_incluir})

    # Recorte por UF ANTES de ler/selecionar séries e antes dos agrupamentos.
    # Nenhuma estação externa ao recorte participa dos modelos regionais.
    ufs_solicitadas = normalizar_estados_incluir(estados_incluir)
    diagnostico_estados = pd.DataFrame()
    n_antes_filtro_uf = len(universo)
    if ufs_solicitadas is not None:
        universo, diagnostico_estados = selecionar_ids_por_estado(
            atributos, ufs_solicitadas, arquivo_malha, ids_incluir=universo)
        atributos['uf_posto'] = diagnostico_estados['uf_posto'].reindex(atributos.index)
        print('Filtro geográfico: UF do posto em ' + ', '.join(ufs_solicitadas))
        print(f'{len(universo)} de {n_antes_filtro_uf} postos permanecem antes da seleção por dados.')
        indefinidos = int(diagnostico_estados.uf_posto.eq('').sum())
        if indefinidos:
            print(f'{indefinidos} posto(s) sem UF inequívoca na malha; não incluídos. '
                  'Verifique o relatório 01_filtro_estados.csv e a simplificação cartográfica.')
        if not universo:
            disponiveis = sorted(set(diagnostico_estados.uf_posto) - {''})
            raise ValueError(
                f'Nenhum posto localizado nas UFs {ufs_solicitadas}. '
                f'UFs identificadas no inventário atual: {disponiveis}. '
                'O filtro NÃO foi desativado automaticamente. '
                'Confira as UFs, IDS_INCLUIR, as coordenadas e a malha.')
    registros, resumos_p, series_q, coberturas, totais_p = [], [], {}, {}, {}
    calendario = cfg.calendario()
    erros_lidos = (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError)
    for codigo in universo:
        caminhos_q = arquivos_q.get(codigo, [])
        registro = {
            'id_bacia': codigo, 'arquivo_encontrado': bool(caminhos_q),
            'arquivo_vazao': '; '.join(str(p) for p in caminhos_q),
            'primeiro_dado_valido': '', 'ultimo_dado_valido': '',
            'dias_periodo': len(calendario), 'dias_validos': 0,
            'dias_falha': len(calendario), 'falhas_periodo_pct': 100.,
            'anos_validos': 0, 'selecionada_vazao': False,
            'motivo_vazao': 'Arquivo de vazão não encontrado',
            'apta_regionalizacao': False, 'motivo_regionalizacao': '',
        }
        if len(caminhos_q) > 1:
            registro['motivo_vazao'] = 'Mais de um arquivo de vazão para o ID; nenhuma versão escolhida'
        elif len(caminhos_q) == 1:
            try:
                serie, leitura = ler_serie_diaria_camels(caminhos_q[0], 'streamflow_m3s', cfg.flags_vazao_aceitas)
                resumo, q, cobertura = avaliar_serie_vazao(serie, cfg)
                registro.update(leitura)
                registro.update(resumo)
                coberturas[codigo] = cobertura
                if registro['selecionada_vazao']:
                    series_q[codigo] = q
            except erros_lidos as exc:
                registro['motivo_vazao'] = f'Erro na leitura: {exc}'

        caminhos_p = arquivos_p.get(codigo, [])
        resumo_p = {
            'id_bacia': codigo, 'arquivo_precip': '; '.join(str(p) for p in caminhos_p),
            'produto': cfg.coluna_precip, 'anos_precip_completos': 0,
            'media_totais_anuais_mm': np.nan, 'precip_serie_aprovada': False,
            'motivo_precip_serie': 'Arquivo de precipitação não encontrado',
        }
        if len(caminhos_p) > 1:
            resumo_p['motivo_precip_serie'] = 'Mais de um arquivo de precipitação para o ID; nenhuma versão escolhida'
        elif len(caminhos_p) == 1:
            try:
                serie_p, _ = ler_serie_diaria_camels(caminhos_p[0], cfg.coluna_precip)
                diag_p, totais, _ = resumir_precipitacao(serie_p, cfg)
                resumo_p.update(diag_p)
                totais_p[codigo] = totais
            except erros_lidos as exc:
                resumo_p['motivo_precip_serie'] = f'Erro na leitura: {exc}'

        motivos_atributos = []
        if codigo not in atributos.index:
            motivos_atributos.append('ID ausente da planilha de atributos')
            a, s, p = np.nan, np.nan, np.nan
        else:
            a = atributos.at[codigo, 'area_km2']
            s = atributos.at[codigo, 'declividade_pct']
            if cfg.fonte_precip == 'serie':
                p = resumo_p['media_totais_anuais_mm'] if resumo_p['precip_serie_aprovada'] else np.nan
                if not resumo_p['precip_serie_aprovada']:
                    motivos_atributos.append('Precipitação: ' + resumo_p['motivo_precip_serie'])
            else:
                p = atributos.at[codigo, 'precip_climatologica_mm_ano']
            atributos.at[codigo, 'precip_mm_ano'] = p
            if not np.isfinite(a) or a <= 0:
                motivos_atributos.append('Área ausente ou não positiva')
            if not np.isfinite(s) or s < 0:
                motivos_atributos.append('Declividade ausente ou negativa')
            if not np.isfinite(p) or p <= 0:
                if not any(m.startswith('Precipitação:') for m in motivos_atributos):
                    motivos_atributos.append('Precipitação ausente ou não positiva')
        registro.update({'area_km2': a, 'declividade_pct': s, 'precip_mm_ano': p,
                         'fonte_precip': cfg.fonte_precip,
                         'anos_precip_completos': resumo_p['anos_precip_completos']})
        registro['apta_regionalizacao'] = bool(registro['selecionada_vazao'] and not motivos_atributos)
        motivos_regionais = ([] if registro['selecionada_vazao'] else ['Vazão não aprovada']) + motivos_atributos
        registro['motivo_regionalizacao'] = '; '.join(motivos_regionais) if motivos_regionais else 'Apta'
        resumo_p.update({'fonte_usada': cfg.fonte_precip, 'precip_mm_ano_usada': p})
        registros.append(registro)
        resumos_p.append(resumo_p)

    # Tratamento defensivo para uma planilha sem estações e sem arquivos.
    if registros:
        selecao = pd.DataFrame(registros).set_index('id_bacia')
        prec = pd.DataFrame(resumos_p).set_index('id_bacia')
    else:
        selecao = pd.DataFrame(columns=['selecionada_vazao', 'apta_regionalizacao'])
        selecao.index.name = 'id_bacia'
        prec = pd.DataFrame(index=selecao.index)
    vazoes_selecionadas = pd.DataFrame(series_q, index=calendario)
    vazoes_selecionadas.columns.name = 'id_bacia'
    ids_aptos = selecao.index[selecao['apta_regionalizacao'].astype(bool)].tolist()
    cobertura_anual = pd.DataFrame(coberturas)
    cobertura_anual.index.name = 'ano_hidrologico'
    metadados.update({
        'configuracao': asdict(cfg),
        'inicio_avaliacao': calendario.min().strftime('%Y-%m-%d'),
        'fim_avaliacao': calendario.max().strftime('%Y-%m-%d'),
        'dias_avaliados': len(calendario),
        'n_bacias_planilha': len(atributos),
        'n_estacoes_avaliadas': len(selecao),
        'estados_incluir': ufs_solicitadas,
        'n_postos_antes_filtro_uf': n_antes_filtro_uf,
        'n_postos_apos_filtro_uf': len(universo),
        'criterio_filtro_uf': 'Estado do posto obtido por ponto-no-polígono; não é contenção da bacia.',
        'arquivo_malha_filtro_uf': str(arquivo_malha) if ufs_solicitadas is not None else None,
        'precaucao_filtro_uf': 'Malha simplificada; conferir postos próximos às divisas; não usa UF mais próxima.',

        'n_ids_com_arquivo_vazao': len(arquivos_q),
        'n_ids_com_arquivo_precip': len(arquivos_p),
        'n_selecionadas_vazao': len(vazoes_selecionadas.columns),
        'n_aptas_regionalizacao': len(ids_aptos),
        'pasta_vazao': str(pasta_vazao), 'pasta_precip': str(pasta_precip),
        'pasta_precip_existe': pasta_precip is not None and Path(pasta_precip).is_dir(),
        'criterio_falhas': 'Calendário completo inclusive; ausências e valores inválidos contam como falha.',
        'criterio_anos': 'Número de anos hidrológicos aceitos pelo limite anual; não precisam ser consecutivos.',
        'criterio_precip_serie': 'Média dos totais de anos civis completos, sem extrapolação e sem preenchimento.',
    })
    return {
        'atributos': atributos.loc[ids_aptos].copy(),
        'vazoes': vazoes_selecionadas.loc[:, ids_aptos].copy(),
        'vazoes_selecionadas': vazoes_selecionadas,
        'atributos_completos': atributos,
        'selecao': selecao,
        'cobertura_anual': cobertura_anual,
        'precipitacao_resumo': prec,
        'precipitacao_anual': pd.DataFrame(totais_p),
        'metadados': metadados,
        'filtro_estados': diagnostico_estados,
    }

# %% 00B | FUNÇÕES DE FILTRO POR UF E CARTOGRAFIA

NAVY = '#123247'
EXTENSAO_BRASIL = (-76., -30., -36., 8.)
SIGLAS_UF = dict(zip(
    ['11','12','13','14','15','16','17','21','22','23','24','25','26','27',
     '28','29','31','32','33','35','41','42','43','50','51','52','53'],
    ['RO','AC','AM','RR','PA','AP','TO','MA','PI','CE','RN','PB','PE','AL',
     'SE','BA','MG','ES','RJ','SP','PR','SC','RS','MS','MT','GO','DF']))
def normalizar_estados_incluir(estados):
    """Valide siglas de UF. None desativa; uma lista vazia é erro, não 'todos'."""
    if estados is None:
        return None
    if isinstance(estados, str):
        estados = [estados]
    try:
        valores = list(estados)
    except TypeError as exc:
        raise ValueError("ESTADOS_INCLUIR deve ser None ou uma lista de siglas, ex.: ['SC', 'RS'].") from exc
    if not valores:
        raise ValueError('ESTADOS_INCLUIR está vazio. Use None para não filtrar ou informe siglas.')
    siglas = sorted({str(uf).strip().upper() for uf in valores})
    invalidas = sorted(set(siglas) - set(SIGLAS_UF.values()))
    if invalidas:
        raise ValueError(f'Siglas inválidas em ESTADOS_INCLUIR: {invalidas}. Use siglas como MG, SP, PR, SC e RS.')
    return siglas


def _pontos_no_anel_uf(pontos, anel, tolerancia=1e-8):
    """Interior e fronteira numérica de um anel em graus (não distância em km).

    A tolerância identifica apenas pontos praticamente sobre um segmento da malha.
    Não corrige a simplificação cartográfica nem ajusta coordenadas ao estado vizinho.
    """
    xy = np.asarray(anel, dtype=float)
    if xy.ndim != 2 or xy.shape[0] < 3 or xy.shape[1] < 2:
        raise ValueError('Anel inválido na malha das UFs.')
    xy = xy[:, :2]
    if not np.isfinite(xy).all():
        raise ValueError('Há coordenadas não finitas na malha das UFs.')
    if not np.array_equal(xy[0], xy[-1]):
        xy = np.vstack([xy, xy[0]])
    dentro = np.zeros(len(pontos), dtype=bool)
    fronteira = np.zeros(len(pontos), dtype=bool)
    na_caixa = np.all((pontos >= xy.min(axis=0)-tolerancia) &
                     (pontos <= xy.max(axis=0)+tolerancia), axis=1)
    indices = np.flatnonzero(na_caixa)
    if not len(indices):
        return dentro, fronteira
    p = pontos[indices]
    dentro[indices] = Caminho(xy).contains_points(p)
    a = xy[:-1]
    segmentos = xy[1:] - a
    comprimento2 = np.sum(segmentos*segmentos, axis=1)
    validos = comprimento2 > 0
    a, segmentos, comprimento2 = a[validos], segmentos[validos], comprimento2[validos]
    if len(a):
        # Blocos evitam matrizes grandes em malhas mais detalhadas.
        for inicio in range(0, len(p), 128):
            bloco = p[inicio:inicio+128]
            diferenca = bloco[:, None, :] - a[None, :, :]
            u = np.sum(diferenca*segmentos[None, :, :], axis=2)/comprimento2[None, :]
            u = np.clip(u, 0, 1)
            distancia2 = np.sum((diferenca-u[:, :, None]*segmentos[None, :, :])**2, axis=2)
            fronteira[indices[inicio:inicio+len(bloco)]] = np.any(distancia2 <= tolerancia**2, axis=1)
    return dentro, fronteira


def identificar_uf_postos(atributos, arquivo_malha):
    """Associe coordenadas de postos às UFs da malha local.

    O resultado não informa se a bacia está inteiramente no estado. Respeita
    polígonos múltiplos e seus buracos. Fronteiras, sobreposições e postos fora
    da malha ficam sem UF definida, sem atribuição arbitrária ao vizinho.
    """
    if atributos.index.has_duplicates:
        raise ValueError('IDs duplicados na identificação de UFs.')
    if arquivo_malha is None:
        raise ValueError('Informe ARQUIVO_MAPA_BRASIL para aplicar ESTADOS_INCLUIR.')
    malha = _ler_malha(arquivo_malha)
    coordenadas = pd.DataFrame(index=atributos.index)
    for coluna in ('latitude', 'longitude'):
        coordenadas[coluna] = pd.to_numeric(
            atributos.get(coluna, pd.Series(index=atributos.index, dtype=float)), errors='coerce')
    validas = np.isfinite(coordenadas.to_numpy()).all(axis=1)
    validas &= coordenadas.latitude.between(-90, 90).to_numpy()
    validas &= coordenadas.longitude.between(-180, 180).to_numpy()
    indices = np.flatnonzero(validas)
    pontos = coordenadas.iloc[indices][['longitude', 'latitude']].to_numpy(dtype=float)
    internas = [set() for _ in range(len(pontos))]
    limites = [set() for _ in range(len(pontos))]
    for feature in malha['features']:
        codigo = str(feature.get('properties', {}).get('codarea', '')).strip()
        uf = SIGLAS_UF.get(codigo)
        if uf is None:
            raise ValueError(f'UF não reconhecida na malha: codarea={codigo!r}. Use a malha que acompanha o pacote.')
        geometria = feature['geometry']
        poligonos = ([geometria['coordinates']] if geometria['type'] == 'Polygon'
                     else geometria['coordinates'])
        for poligono in poligonos:
            dentro, fronteira = _pontos_no_anel_uf(pontos, poligono[0])
            for buraco in poligono[1:]:
                dentro_buraco, limite_buraco = _pontos_no_anel_uf(pontos, buraco)
                dentro &= ~dentro_buraco
                fronteira |= limite_buraco
            for j in np.flatnonzero(dentro & ~fronteira):
                internas[j].add(uf)
            for j in np.flatnonzero(fronteira):
                limites[j].add(uf)
    registros = []
    for linha, (codigo, coord) in enumerate(coordenadas.iterrows()):
        registros.append({'id_bacia': codigo, 'latitude': coord.latitude,
                          'longitude': coord.longitude, 'uf_posto': '',
                          'ufs_candidatas': '', 'situacao_uf': 'coordenadas_invalidas'})
    for j, linha in enumerate(indices):
        candidatas = internas[j] | limites[j]
        registro = registros[linha]
        registro['ufs_candidatas'] = ';'.join(sorted(candidatas))
        if limites[j]:
            registro['situacao_uf'] = 'sobre_fronteira_da_malha'
        elif len(internas[j]) == 1:
            registro['uf_posto'] = next(iter(internas[j]))
            registro['situacao_uf'] = 'identificada_na_malha'
        elif len(internas[j]) > 1:
            registro['situacao_uf'] = 'sobreposicao_de_ufs'
        else:
            registro['situacao_uf'] = 'sem_correspondencia_na_malha'
    return pd.DataFrame(registros, columns=[
        'id_bacia', 'latitude', 'longitude', 'uf_posto', 'ufs_candidatas', 'situacao_uf'
    ]).set_index('id_bacia')


def selecionar_ids_por_estado(atributos, estados, arquivo_malha, ids_incluir=None):
    """Retorne IDs selecionados e diagnóstico; não altere os atributos originais."""
    ufs = normalizar_estados_incluir(estados)
    if ufs is None:
        raise ValueError('Esta função requer pelo menos uma UF; None desativa o filtro na preparação da base.')
    if ids_incluir is None:
        candidatos = list(atributos.index)
    else:
        if isinstance(ids_incluir, str):
            raise ValueError('IDS_INCLUIR deve ser uma lista de códigos ou None.')
        candidatos = sorted({normalizar_id(x) for x in ids_incluir})
    diagnostico = identificar_uf_postos(atributos.reindex(candidatos), arquivo_malha)
    diagnostico['incluida_filtro_uf'] = diagnostico.uf_posto.isin(ufs)
    diagnostico['motivo_filtro_uf'] = np.where(
        diagnostico.incluida_filtro_uf, 'UF do posto incluída',
        np.where(diagnostico.uf_posto.ne(''), 'UF do posto fora do recorte',
                 'UF indeterminada na malha; verificar coordenadas/fronteira'))
    selecionadas = diagnostico.index[diagnostico.incluida_filtro_uf].tolist()
    return selecionadas, diagnostico


def conferir_coordenadas_mapa(atributos):
    """Diagnóstico gráfico; não modifica a seleção hidrológica das séries."""
    dados = pd.DataFrame(index=atributos.index)
    for c in ['latitude', 'longitude']:
        dados[c] = pd.to_numeric(atributos.get(c, pd.Series(index=atributos.index,
                                                          dtype=float)), errors='coerce')
    finitas = np.isfinite(dados[['latitude', 'longitude']]).all(axis=1)
    geograficas = dados.latitude.between(-90, 90) & dados.longitude.between(-180, 180)
    xmin, xmax, ymin, ymax = EXTENSAO_BRASIL
    enquadradas = dados.latitude.between(ymin, ymax) & dados.longitude.between(xmin, xmax)
    dados['plotavel'] = finitas & geograficas & enquadradas
    dados['motivo_mapa'] = np.where(dados.plotavel, 'Coordenadas disponíveis',
        np.where(finitas & geograficas, 'Fora do enquadramento do Brasil; conferir coordenadas',
                 'Coordenadas ausentes ou inválidas'))
    return dados


def _postos_validos(atributos):
    diagnostico = conferir_coordenadas_mapa(atributos)
    invalidos = diagnostico.index[~diagnostico.plotavel]
    if len(invalidos):
        warnings.warn(f'{len(invalidos)} posto(s) omitido(s) apenas do mapa por coordenadas: '
                      + ', '.join(map(str, invalidos)), stacklevel=2)
    dados = atributos.loc[diagnostico.plotavel].copy()
    dados[['latitude', 'longitude']] = diagnostico.loc[diagnostico.plotavel, ['latitude', 'longitude']]
    return dados


def _ler_malha(arquivo):
    caminho = Path(arquivo)
    if not caminho.is_file():
        raise FileNotFoundError(f'Malha do Brasil não encontrada: {caminho}. '
                                'Extraia também a pasta mapas do pacote.')
    malha = json.loads(caminho.read_text(encoding='utf-8'))
    if malha.get('type') != 'FeatureCollection' or not malha.get('features'):
        raise ValueError('A malha deve ser um GeoJSON FeatureCollection com polígonos.')
    if any(f['geometry']['type'] not in ['Polygon', 'MultiPolygon'] for f in malha['features']):
        raise ValueError('A malha do Brasil deve conter Polygon ou MultiPolygon.')
    return malha


def _caminho_poligono(aneis):
    """Preserva ilhas e buracos; exterior anti-horário, interiores horários."""
    vertices, codigos = [], []
    for i, anel in enumerate(aneis):
        xy = np.asarray(anel, dtype=float)[:, :2]
        if not np.array_equal(xy[0], xy[-1]):
            xy = np.vstack([xy, xy[0]])
        area2 = np.sum(xy[:-1, 0]*xy[1:, 1] - xy[1:, 0]*xy[:-1, 1])
        if (i == 0 and area2 < 0) or (i > 0 and area2 > 0):
            xy = xy[::-1]
        vertices.extend(xy)
        codigos.extend([Caminho.MOVETO] + [Caminho.LINETO]*(len(xy)-2) + [Caminho.CLOSEPOLY])
    return Caminho(vertices, codigos)


def _centro_anel(anel):
    xy = np.asarray(anel, dtype=float)[:, :2]
    x, y = xy[:, 0], xy[:, 1]
    cruz = x[:-1]*y[1:] - x[1:]*y[:-1]
    area2 = cruz.sum()
    if abs(area2) < 1e-10:
        return abs(area2), x.mean(), y.mean()
    return abs(area2), ((x[:-1]+x[1:])*cruz).sum()/(3*area2), ((y[:-1]+y[1:])*cruz).sum()/(3*area2)


def _fundo_brasil(ax, malha, extensao, rotular_ufs=False):
    # Cor neutra: a malha não representa países vizinhos nem separa terra/oceano.
    ax.set_facecolor('#EEF0F1')
    xmin, xmax, ymin, ymax = extensao
    for feature in malha['features']:
        geom = feature['geometry']
        poligonos = [geom['coordinates']] if geom['type'] == 'Polygon' else geom['coordinates']
        for poligono in poligonos:
            ax.add_patch(PathPatch(_caminho_poligono(poligono), facecolor='#F6F4EE',
                                   edgecolor='#9BA9AF', linewidth=.65, zorder=1))
        if rotular_ufs:
            _, x, y = max(_centro_anel(p[0]) for p in poligonos)
            sigla = SIGLAS_UF.get(str(feature.get('properties', {}).get('codarea')), '')
            if xmin < x < xmax and ymin < y < ymax:
                ax.text(x, y, sigla, color='#68777E', fontsize=8, ha='center',
                        va='center', zorder=2, clip_on=True)
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), xlabel='Longitude', ylabel='Latitude')
    ax.set_aspect('equal', adjustable='box')
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{abs(v):g}°' + ('O' if v < 0 else 'L' if v > 0 else '')))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{abs(v):g}°' + ('S' if v < 0 else 'N' if v > 0 else '')))
    ax.tick_params(labelsize=9)
    ax.grid(True, color='#B4C1C8', linewidth=.5, alpha=.45, zorder=0)


def _extensao_postos(postos):
    if postos.empty:
        return EXTENSAO_BRASIL
    xmin, xmax = postos.longitude.min(), postos.longitude.max()
    ymin, ymax = postos.latitude.min(), postos.latitude.max()
    margem_x, margem_y = max((xmax-xmin)*.14, 1.5), max((ymax-ymin)*.12, 1.5)
    return (max(xmin-margem_x, EXTENSAO_BRASIL[0]), min(xmax+margem_x, EXTENSAO_BRASIL[1]),
            max(ymin-margem_y, EXTENSAO_BRASIL[2]), min(ymax+margem_y, EXTENSAO_BRASIL[3]))


def _rotular_postos(ax, dados):
    for codigo, linha in dados.iterrows():
        ax.annotate(str(codigo), (linha.longitude, linha.latitude), xytext=(6, -9),
                    textcoords='offset points', fontsize=7, va='top', color=NAVY, zorder=8,
                    arrowprops=dict(arrowstyle='-', linewidth=.6, color=NAVY, shrinkA=2, shrinkB=4),
                    bbox=dict(facecolor='white', alpha=.8, edgecolor='none', pad=.8))

# %% 00C | FUNÇÕES DE CLUSTERIZAÇÃO E REGRESSÃO
def matriz_atributos(atributos, nomes):
    """Log da área, quando selecionada; demais atributos em escala original."""
    x = atributos.loc[:, nomes].astype(float).copy()
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError('Há atributo inválido.')
    if 'area_km2' in x:
        if (x.area_km2 <= 0).any():
            raise ValueError('Área precisa ser positiva para log.')
        x['area_km2'] = np.log(x['area_km2'])
    return x


def agrupar_bacias(atributos, nomes, k=3, metodo='kmeans', semente=42):
    """Ajustar transformação, padronização e grupos apenas no treinamento."""
    if not 2 <= k < len(atributos):
        raise ValueError('K deve estar entre 2 e N−1.')
    x = matriz_atributos(atributos, nomes)
    scaler = StandardScaler().fit(x)
    z = scaler.transform(x)
    if metodo == 'kmeans':
        modelo = KMeans(n_clusters=k, n_init=30, random_state=semente).fit(z)
        labels = modelo.labels_
        arvore = None
    elif metodo == 'ward':
        arvore = linkage(z, method='ward', metric='euclidean')
        labels = cut_tree(arvore, n_clusters=k).ravel()
        modelo = None
    else:
        raise ValueError("Método deve ser 'kmeans' ou 'ward'.")
    # Só a numeração/legenda muda: grupos ordenados pela precipitação média.
    ordem = sorted(np.unique(labels), key=lambda g: atributos.loc[labels == g, 'precip_mm_ano'].mean())
    mapa = {g: i+1 for i, g in enumerate(ordem)}
    labels = pd.Series([mapa[g] for g in labels], index=atributos.index, name='regiao')
    centros = {int(g): z[labels.to_numpy() == g].mean(axis=0) for g in sorted(labels.unique())}
    tamanhos = labels.value_counts().to_dict()
    return {'metodo': metodo, 'nomes': list(nomes), 'scaler': scaler, 'z': z,
            'labels': labels, 'centros': centros, 'tamanhos': tamanhos,
            'arvore': arvore, 'modelo': modelo,
            'silhueta': float(silhouette_score(z, labels))}


def atribuir_bacia(ajuste, atributos_alvo):
    """Atribuir novos atributos aos grupos já formados.

    K-means: centroide mais próximo, equivalente à regra de predict.
    Ward: regra adicional explicitada para a aula, centroide mais próximo.
    Ela NÃO é uma previsão nativa nem uma reexecução da hierarquia de Ward.
    A regra também é usada na validação com uma bacia fora por vez.
    """
    z = ajuste['scaler'].transform(matriz_atributos(atributos_alvo, ajuste['nomes']))
    grupos = sorted(ajuste['centros'])
    centros = np.array([ajuste['centros'][g] for g in grupos])
    dist2 = ((z[:, None, :]-centros[None, :, :])**2).sum(axis=2)
    return pd.Series(np.asarray(grupos)[dist2.argmin(axis=1)], index=atributos_alvo.index, name='regiao')


def comparar_grupos(kmeans, ward):
    tabela = pd.crosstab(kmeans['labels'], ward['labels'])
    return tabela, float(adjusted_rand_score(kmeans['labels'], ward['labels']))


def nomes_regressao(forma):
    """Escolhe uma variável para simples, todas para múltipla."""
    return (VARIAVEIS_REGRESSAO[:1] if forma.endswith('simples')
            else list(VARIAVEIS_REGRESSAO))


def matriz_regressao(atributos, forma, nomes=None):
    nomes = nomes_regressao(forma) if nomes is None else list(nomes)
    faltantes = [nome for nome in nomes if nome not in atributos.columns]
    if faltantes:
        raise ValueError(f'Variáveis de regressão ausentes: {faltantes}')
    x = atributos.loc[:, nomes].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError(f'Valores faltantes/não numéricos nas variáveis de regressão: {nomes}')
    if forma.startswith('potencial'):
        if (x <= 0).any():
            raise ValueError(f'Regressão potencial exige entradas positivas: {nomes}')
        x = np.log(x)
    return x, nomes


def ajustar_regressao(atributos, alvo, forma='potencial_multipla'):
    """OLS na escala original ou log; retransposição por smearing nos logs."""
    formas = ['linear_simples', 'linear_multipla', 'potencial_simples', 'potencial_multipla']
    if forma not in formas:
        raise ValueError(f'Forma inválida: {forma}.')
    x, nomes = matriz_regressao(atributos, forma)
    y = np.asarray(alvo, dtype=float)
    if not np.isfinite(y).all():
        raise ValueError('Alvo ausente/infinito.')
    if len(y) <= x.shape[1]+1:
        raise ValueError('Poucas bacias para os parâmetros da regressão.')
    log = forma.startswith('potencial')
    if log and (y <= 0).any():
        raise ValueError('Modelo potencial exige alvo >0; não somar epsilon a Q90=0 silenciosamente.')
    yt = np.log(y) if log else y
    modelo = LinearRegression().fit(x, yt)
    residuos = yt-modelo.predict(x)
    fator = float(np.exp(residuos).mean()) if log else 1.0
    return {'modelo': modelo, 'forma': forma, 'nomes': nomes,
            'smearing': fator, 'n_bacias': len(y), 'residuos_ajuste': residuos}


def prever_regressao(ajuste, atributos):
    x, _ = matriz_regressao(atributos, ajuste['forma'], ajuste['nomes'])
    estimativa = ajuste['modelo'].predict(x)
    if ajuste['forma'].startswith('potencial'):
        estimativa = np.exp(estimativa)*ajuste['smearing']
    # Previsões lineares negativas são mantidas e sinalizadas, nunca truncadas.
    return pd.Series(estimativa, index=atributos.index, name='previsto')


def texto_equacao(ajuste, alvo='μ'):
    """Exibe todos os coeficientes com os nomes de fato usados no modelo."""
    b0 = float(ajuste['modelo'].intercept_)
    betas = ajuste['modelo'].coef_
    nomes = ajuste['nomes']
    if ajuste['forma'].startswith('potencial'):
        a = np.exp(b0) * ajuste['smearing']
        termos = ''.join(f' × {nome}^({float(beta):.5g})'
                        for nome, beta in zip(nomes, betas))
        return f'{alvo} = {a:.6g}{termos}  [com smearing]'
    termos = ''.join(f' {float(beta):+.6g} × {nome}'
                    for nome, beta in zip(nomes, betas))
    return f'{alvo} = {b0:.6g}{termos}'


def metricas_regressao(observado, previsto):
    o, p = np.asarray(observado, float), np.asarray(previsto, float)
    valido = np.isfinite(o) & np.isfinite(p)
    o, p = o[valido], p[valido]
    if not len(o):
        return {'n': 0, 'RMSE_m3s': np.nan, 'erro_rel_abs_mediano_pct': np.nan, 'vies_pct': np.nan}
    return {'n': len(o), 'RMSE_m3s': float(np.sqrt(np.mean((p-o)**2))),
            'erro_rel_abs_mediano_pct': float(np.median(np.abs((p[o != 0]-o[o != 0])/o[o != 0]))*100) if np.any(o != 0) else np.nan,
            'vies_pct': float(100*np.sum(p-o)/np.sum(o)) if np.sum(o) else np.nan}

# %% 00D | FUNÇÕES DE ORGANIZAÇÃO, FIGURAS E SAÍDAS
def caminho_io(caminho):
    """Caminho absoluto; prefixo estendido no Windows, sem mudar a pasta escolhida."""
    p = Path(caminho).expanduser().absolute()
    texto = str(p)
    if os.name == 'nt' and not texto.startswith('\\\\?\\'):
        if texto.startswith('\\\\'):
            texto = '\\\\?\\UNC\\' + texto[2:]
        else:
            texto = '\\\\?\\' + texto
    return Path(texto)


def iniciar_saidas(tipo):
    """Cria uma execução própria em aula 7_index_flood/resultados/<tipo>."""
    raiz = caminho_io(PASTA_AULA / 'resultados' / tipo)
    raiz.mkdir(parents=True, exist_ok=True)
    pasta = raiz / datetime.now().strftime('%y%m%d_%H%M%S_%f')
    pasta.mkdir(exist_ok=False)
    (pasta / 'tab').mkdir()
    (pasta / 'fig').mkdir()
    print(f'\nRESULTADOS ({tipo}): {pasta}')
    return pasta


def salvar_csv(quadro, pasta, nome, index=True):
    destino = caminho_io(Path(pasta) / nome)
    destino.parent.mkdir(parents=True, exist_ok=True)
    quadro.to_csv(destino, index=index, encoding='utf-8-sig')
    return destino


def salvar_json(conteudo, pasta, nome):
    destino = caminho_io(Path(pasta) / nome)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(conteudo, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def figura_aula(fig, nome, pasta=None):
    destino = caminho_io((PASTA_FIGURAS if pasta is None else Path(pasta)) / f'{nome}.png')
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destino, dpi=DPI_FIGURAS, bbox_inches='tight')
    if MOSTRAR_GRAFICOS:
        plt.show(block=False)
    else:
        plt.close(fig)


def pasta_cluster(g):
    pasta = PASTA_RESULTADOS / 'clusters' / f'C{int(g):02d}'
    (pasta / 'tab').mkdir(parents=True, exist_ok=True)
    (pasta / 'fig').mkdir(parents=True, exist_ok=True)
    return pasta


def anunciar(numero, titulo, pergunta=''):
    print('\n' + '=' * 72)
    print(f'{numero} | {titulo}')
    if pergunta:
        print('Pergunta: ' + pergunta)


def verificar_configuracao():
    if isinstance(K, bool) or not isinstance(K, (int, np.integer)) or K < 2:
        raise ValueError('K deve ser inteiro e >= 2.')
    if METODO_REGIONAL not in ('kmeans', 'ward'):
        raise ValueError("METODO_REGIONAL deve ser 'kmeans' ou 'ward'.")
    if MIN_BACIAS_POR_GRUPO < 2:
        raise ValueError('MIN_BACIAS_POR_GRUPO deve ser >= 2.')
    if len(ATRIBUTOS_CLUSTER) < 2 or len(set(ATRIBUTOS_CLUSTER)) != len(ATRIBUTOS_CLUSTER):
        raise ValueError('Informe pelo menos dois atributos diferentes em ATRIBUTOS_CLUSTER.')
    normalizar_estados_incluir(ESTADOS_INCLUIR)
    if TIPO_ANALISE == 'indexflood':
        if FAMILIA_APLICACAO not in ('GEV', 'PE3'):
            raise ValueError("FAMILIA_APLICACAO deve ser 'GEV' ou 'PE3'.")
        if not np.isfinite(TEMPOS_RETORNO).all() or np.any(TEMPOS_RETORNO <= 1):
            raise ValueError('Todos os tempos de retorno devem ser finitos e maiores que 1.')
    if not isinstance(VARIAVEIS_REGRESSAO, (list, tuple)) or not VARIAVEIS_REGRESSAO:
        raise ValueError('VARIAVEIS_REGRESSAO exige uma lista não vazia.')
    if any(not isinstance(c, str) or not c.strip() for c in VARIAVEIS_REGRESSAO):
        raise ValueError('Cada variável da regressão deve ser texto não vazio.')
    if len(set(VARIAVEIS_REGRESSAO)) != len(VARIAVEIS_REGRESSAO):
        raise ValueError('VARIAVEIS_REGRESSAO contém variáveis repetidas.')
    if FORMA_REGRESSAO.endswith('multipla') and len(VARIAVEIS_REGRESSAO) < 2:
        raise ValueError('Regressão múltipla exige pelo menos duas variáveis.')
    permitidas = ('linear_simples', 'linear_multipla', 'potencial_simples', 'potencial_multipla')
    if FORMA_REGRESSAO not in permitidas:
        raise ValueError(f'FORMA_REGRESSAO deve ser uma de {permitidas}.')
    if PLOTAR_MAPAS or ESTADOS_INCLUIR is not None:
        if not Path(ARQUIVO_MAPA_BRASIL).is_file():
            raise FileNotFoundError(f'Malha não encontrada: {ARQUIVO_MAPA_BRASIL}. Extraia também a pasta mapas.')


def criar_configuracao_selecao():
    return ConfiguracaoSelecao(
        ano_inicial=ANO_INICIAL_ANALISE, ano_final=ANO_FINAL_ANALISE,
        anos_minimos=ANOS_MINIMOS, max_falhas_periodo_pct=MAX_FALHAS_PERIODO_PCT,
        max_falhas_ano_pct=MAX_FALHAS_ANO_PCT,
        mes_inicio_ano_hidrologico=MES_INICIO_ANO_HIDROLOGICO,
        fonte_precip=FONTE_PRECIPITACAO, coluna_precip=PRODUTO_PRECIPITACAO,
        anos_minimos_precip=ANOS_MINIMOS_PRECIPITACAO,
        flags_vazao_aceitas=FLAGS_VAZAO_ACEITAS, ids_incluir=IDS_INCLUIR,
        recursivo=BUSCAR_EM_SUBPASTAS).validar()


def exportar_inventario(base):
    sel = base['selecao']
    for chave, nome in [
        ('selecao', '01_selecao.csv'), ('cobertura_anual', '01_cobertura.csv'),
        ('precipitacao_resumo', '01_precip.csv'), ('precipitacao_anual', '01_p_anuais.csv'),
        ('atributos', '01_atributos.csv'), ('filtro_estados', '01_filtro_uf.csv')]:
        salvar_csv(base[chave], PASTA_TABELAS, nome)
    salvar_csv(sel.loc[sel.selecionada_vazao], PASTA_TABELAS, '01_aprovadas.csv')
    salvar_csv(sel.loc[~sel.selecionada_vazao], PASTA_TABELAS, '01_excluidas.csv')
    meta = dict(base['metadados'])
    meta.update({'analise': TIPO_ANALISE, 'atributos_cluster': ATRIBUTOS_CLUSTER,
                 'K': K, 'metodo_regional': METODO_REGIONAL,
                 'min_bacias_por_grupo': MIN_BACIAS_POR_GRUPO,
                 'forma_regressao': FORMA_REGRESSAO, 'variaveis_regressao': VARIAVEIS_REGRESSAO,
                 'semente': SEMENTE, 'pasta_resultados': str(PASTA_RESULTADOS)})
    salvar_json(meta, PASTA_RESULTADOS, 'config.json')
    print(f'Inventário: {len(sel)} postos; {int(sel.arquivo_encontrado.sum())} com arquivo de vazão; '
          f'{int(sel.selecionada_vazao.sum())} aprovados; {int(sel.apta_regionalizacao.sum())} com atributos básicos.')
    print('Falhas globais: todos os dias da janela solicitada, inclusive suas bordas.')
    print('ANOS_MINIMOS: anos aceitos no critério anual; não precisam ser consecutivos.')
    print('Nenhuma falha é preenchida. Zeros de vazão são preservados.')
    parte = sel.loc[sel.arquivo_encontrado]
    if len(parte):
        print(parte[['anos_validos', 'falhas_periodo_pct', 'selecionada_vazao',
                     'apta_regionalizacao']].round(3).to_string())


def plotar_selecao(base):
    sel = base['selecao']
    parte = sel.loc[sel.arquivo_encontrado]
    fig, ax = plt.subplots(figsize=(8, 5))
    for aprovado, marcador, nome in [(True, 'o', 'Série aprovada'), (False, 'x', 'Série excluída')]:
        d = parte.loc[parte.selecionada_vazao == aprovado]
        if len(d):
            ax.scatter(d.anos_validos, d.falhas_periodo_pct, marker=marcador, label=nome)
    ax.axvline(ANOS_MINIMOS, ls='--', label='Mínimo de anos')
    ax.axhline(MAX_FALHAS_PERIODO_PCT, ls=':', label='Máximo de falhas')
    ax.set(xlabel='Anos aceitos', ylabel='Falhas na janela (%)', title='Seleção das séries')
    ax.grid(alpha=.25); ax.legend(fontsize=9)
    figura_aula(fig, '01_selecao')
    if PLOTAR_MAPAS:
        atributos = base['atributos_completos'].reindex(sel.index)
        validos = _postos_validos(atributos)
        for zoom, sufixo in [(False, 'brasil'), (True, 'area')]:
            fig, ax = plt.subplots(figsize=(8, 7))
            malha = _ler_malha(ARQUIVO_MAPA_BRASIL)
            _fundo_brasil(ax, malha, _extensao_postos(validos) if zoom else EXTENSAO_BRASIL,
                          rotular_ufs=zoom)
            for mascara, marcador, nome in [
                (~sel.arquivo_encontrado, '.', 'Sem arquivo'),
                (sel.arquivo_encontrado & ~sel.selecionada_vazao, 'x', 'Excluída'),
                (sel.selecionada_vazao & ~sel.apta_regionalizacao, '^', 'Faltam atributos'),
                (sel.apta_regionalizacao, 'o', 'Apta')]:
                ids = validos.index.intersection(sel.index[mascara])
                d = validos.loc[ids]
                if len(d):
                    ax.scatter(d.longitude, d.latitude, s=30, marker=marcador,
                               label=f'{nome} (n={len(d)})', zorder=4)
            if ROTULAR_BACIAS_NO_MAPA and zoom:
                _rotular_postos(ax, validos)
            ax.legend(fontsize=8)
            ax.set_title('Postos CAMELS — seleção das séries')
            fig.supxlabel('Pontos = postos, não polígonos de bacias. Fundo: IBGE / UFs.', fontsize=9)
            figura_aula(fig, f'01_mapa_{sufixo}')


def preparar_treino(base):
    """Separar a bacia-alvo ANTES de quaisquer ajustes ou agrupamentos."""
    atributos = base['atributos'].copy()
    exigidas = list(dict.fromkeys(ATRIBUTOS_CLUSTER + VARIAVEIS_REGRESSAO + ['area_km2', 'precip_mm_ano']))
    faltantes = [c for c in exigidas if c not in atributos.columns]
    if faltantes:
        raise ValueError(f'Atributos não encontrados: {faltantes}. Confira os nomes na planilha.')
    numericos = atributos[exigidas].apply(pd.to_numeric, errors='coerce')
    ok = np.isfinite(numericos.to_numpy(dtype=float)).all(axis=1)
    ok &= numericos.area_km2.gt(0).to_numpy() & numericos.precip_mm_ano.gt(0).to_numpy()
    relatorio = pd.DataFrame(index=atributos.index)
    relatorio['atributos_validos'] = ok
    relatorio['motivo'] = ''
    for codigo in atributos.index[~ok]:
        ruins = [c for c in exigidas if not np.isfinite(numericos.at[codigo, c])]
        if numericos.at[codigo, 'area_km2'] <= 0: ruins.append('area_km2 <= 0')
        if numericos.at[codigo, 'precip_mm_ano'] <= 0: ruins.append('precip_mm_ano <= 0')
        relatorio.at[codigo, 'motivo'] = '; '.join(ruins)
    salvar_csv(relatorio, PASTA_TABELAS, '01_atrib_validos.csv')
    atributos = atributos.loc[ok].copy()
    for c in exigidas:
        atributos[c] = numericos.loc[atributos.index, c].astype(float)
    vazoes = base['vazoes'].reindex(columns=atributos.index)
    alvo = None
    if BACIA_ALVO is not None:
        alvo = normalizar_id(BACIA_ALVO)
        if alvo not in atributos.index:
            raise ValueError(f'BACIA_ALVO={alvo} não está apta neste recorte. Confira o relatório de seleção.')
    elif RESERVAR_BACIA_ALVO and len(atributos) >= K + 2:
        candidatas = [c for c in sorted(atributos.index) if c != BACIA_EXEMPLO]
        alvo = candidatas[-1] if candidatas else None
    aa = atributos.loc[[alvo]].copy() if alvo is not None else atributos.iloc[:0].copy()
    qa = vazoes[alvo].copy() if alvo is not None else None
    if alvo is not None:
        atributos = atributos.drop(index=alvo)
        vazoes = vazoes.drop(columns=alvo)
        print(f'Bacia reservada: {alvo}; excluída dos ajustes e dos clusters.')
    else:
        print('Sem bacia reservada: opção desativada ou poucos postos para a reserva automática.')
    disponivel = bool(EXECUTAR_ETAPAS_REGIONAIS and len(atributos) > K)
    print(f'Treinamento regional: {len(atributos)} postos; K={K}.')
    if not disponivel:
        print('Clusterização não executável: requer etapas habilitadas e mais postos que K. '
              'As estatísticas locais serão calculadas normalmente.')
    if (~ok).any():
        print(f'{int((~ok).sum())} postos com vazão aprovada não entraram nos clusters por atributos inválidos.')
    return atributos, vazoes, aa, qa, alvo, disponivel


def plotar_hidrograma(serie, codigo):
    inicio = serie.first_valid_index()
    if inicio is None:
        return
    recorte = serie.loc[inicio: inicio + pd.DateOffset(years=ANOS_HIDROGRAMA)]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(recorte.index, recorte, lw=1)
    ax.set(title=f'{codigo} — trecho da série diária', ylabel='Vazão média diária (m³/s)', xlabel='Data')
    ax.grid(alpha=.25)
    figura_aula(fig, '02_hidrograma')


def plotar_clusters(atributos, agrupamentos, atributos_alvo):
    for metodo, ag in agrupamentos.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        for g in sorted(ag['labels'].unique()):
            m = ag['labels'].to_numpy() == g
            ax.scatter(ag['z'][m, 0], ag['z'][m, 1], label=f'C{g:02d} (n={m.sum()})')
            if metodo == 'kmeans':
                ax.scatter(*ag['centros'][g][:2], marker='x', s=100)
        ax.set(xlabel='z de ' + ATRIBUTOS_CLUSTER[0], ylabel='z de ' + ATRIBUTOS_CLUSTER[1],
               title=f'{metodo}: projeção nos dois primeiros atributos; K={K}')
        ax.legend(fontsize=8); ax.grid(alpha=.25)
        figura_aula(fig, f'04_{metodo}_atributos')
        if PLOTAR_MAPAS:
            malha = _ler_malha(ARQUIVO_MAPA_BRASIL)
            dados = _postos_validos(atributos)
            alvo = _postos_validos(atributos_alvo)
            fig, ax = plt.subplots(figsize=(8, 7))
            ext = _extensao_postos(pd.concat([dados, alvo])) if MAPA_REGIOES_ZOOM else EXTENSAO_BRASIL
            _fundo_brasil(ax, malha, ext, rotular_ufs=MAPA_REGIOES_ZOOM)
            for g in sorted(ag['labels'].unique()):
                ids = dados.index.intersection(ag['labels'].index[ag['labels'] == g])
                d = dados.loc[ids]
                ax.scatter(d.longitude, d.latitude, s=38, label=f'C{g:02d} (n={len(d)})', zorder=4)
            if len(alvo):
                ax.scatter(alvo.longitude, alvo.latitude, marker='*', s=160, label='Alvo reservado', zorder=5)
            if ROTULAR_BACIAS_NO_MAPA:
                _rotular_postos(ax, dados)
            ax.legend(fontsize=8); ax.set_title(f'{metodo} — regiões candidatas')
            fig.supxlabel('Fundo: IBGE / UFs. Cada ponto representa um posto.', fontsize=9)
            figura_aula(fig, f'04_{metodo}_mapa')
    if 'ward' in agrupamentos:
        arvore = agrupamentos['ward']['arvore']
        fig, ax = plt.subplots(figsize=(12, 5))
        dendrogram(arvore, labels=atributos.index.tolist(), leaf_rotation=90, leaf_font_size=6, ax=ax)
        altura = float(np.mean([arvore[-K, 2], arvore[-K+1, 2]]))
        ax.axhline(altura, ls='--', label=f'Corte ilustrativo: K={K}')
        ax.set(title='Ward — hierarquia de fusões', ylabel='Distância de ligação')
        ax.legend(); figura_aula(fig, '04_ward_arvore')


def executar_clusters(atributos, atributos_alvo):
    ajustes = {m: agrupar_bacias(atributos, ATRIBUTOS_CLUSTER, K, m, SEMENTE)
               for m in ('kmeans', 'ward')}
    contagem = pd.DataFrame({m: a['labels'].value_counts() for m, a in ajustes.items()}).sort_index()
    salvar_csv(contagem, PASTA_TABELAS, '04_contagem.csv')
    rotulos = pd.DataFrame({m: a['labels'] for m, a in ajustes.items()})
    salvar_csv(atributos.join(rotulos), PASTA_TABELAS, '04_clusters.csv')
    cruzamento, ari = comparar_grupos(ajustes['kmeans'], ajustes['ward'])
    salvar_csv(cruzamento, PASTA_TABELAS, '04_cruzamento.csv')
    print('Número de postos por grupo:\n' + contagem.to_string())
    print(f'Rand ajustado entre métodos = {ari:.3f}')
    for m, a in ajustes.items():
        print(f"{m}: silhueta = {a['silhueta']:.3f}")
        pequenos = {int(g): int(n) for g, n in a['tamanhos'].items() if n < MIN_BACIAS_POR_GRUPO}
        if pequenos:
            print(f'AVISO {m}: grupos pequenos {pequenos}. Seus mapas permanecem disponíveis; '
                  'ajustes insuficientes serão registrados, sem interromper os demais grupos.')
    plotar_clusters(atributos, ajustes, atributos_alvo)
    escolhido = ajustes[METODO_REGIONAL]
    grupos = escolhido['labels']
    regiao_alvo = int(atribuir_bacia(escolhido, atributos_alvo).iloc[0]) if len(atributos_alvo) else None
    for g in sorted(grupos.unique()):
        ids = grupos.index[grupos == g]
        pasta = pasta_cluster(g)
        salvar_csv(atributos.loc[ids], pasta / 'tab', 'atributos.csv')
    print(f'Método das aplicações por cluster: {METODO_REGIONAL}.')
    if regiao_alvo is not None:
        print(f'Alvo atribuído a C{regiao_alvo:02d} somente por atributos.')
    print('Ward: novas bacias usam a regra adicional do centroide mais próximo.')
    return ajustes, grupos, regiao_alvo


def grafico_observado_previsto(observado, previsto, titulo, nome, pasta=None):
    o, p = np.asarray(observado, float), np.asarray(previsto, float)
    ok = np.isfinite(o) & np.isfinite(p)
    o, p = o[ok], p[ok]
    if not len(o):
        return
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.scatter(o, p)
    lo, hi = min(o.min(), p.min()), max(o.max(), p.max())
    if lo == hi:
        lo, hi = lo - max(1., abs(lo)*.05), hi + max(1., abs(hi)*.05)
    ax.plot([lo, hi], [lo, hi], '--', label='1:1')
    m = metricas_regressao(o, p)
    ax.text(.04, .96, f"n={m['n']}\nRMSE={m['RMSE_m3s']:.3f} m³/s\nMediana |erro relativo|={m['erro_rel_abs_mediano_pct']:.1f}%",
            transform=ax.transAxes, va='top', fontsize=10)
    ax.set(xlabel='Referência (m³/s)', ylabel='Estimativa (m³/s)', title=titulo)
    ax.grid(alpha=.25); ax.legend(loc='lower right')
    figura_aula(fig, nome, pasta)


def ajustar_modelos_por_cluster(atributos, estatisticas, grupos, alvos):
    """Retorna modelos independentes por cluster e por alvo; nunca combina clusters."""
    modelos, registros = {}, []
    formas = ['linear_simples', 'linear_multipla', 'potencial_simples', 'potencial_multipla']
    for g in sorted(grupos.unique()):
        ids = grupos.index[grupos == g]
        pasta = pasta_cluster(g)
        modelos[int(g)] = {}
        linhas_grupo = []
        for alvo in alvos:
            obs = estatisticas.loc[ids, alvo]
            for forma in formas:
                linha = {'regiao': int(g), 'alvo': alvo, 'forma': forma, 'n_bacias': len(ids)}
                try:
                    if len(ids) < MIN_BACIAS_POR_GRUPO:
                        raise ValueError(f'Menos de {MIN_BACIAS_POR_GRUPO} postos neste grupo.')
                    modelo = ajustar_regressao(atributos.loc[ids], obs, forma)
                    pred = prever_regressao(modelo, atributos.loc[ids])
                    linha.update({'status': 'ajustado', 'equacao': texto_equacao(modelo, alvo=alvo),
                                  **metricas_regressao(obs, pred)})
                    if forma == FORMA_REGRESSAO:
                        modelos[int(g)][alvo] = modelo
                        saida = pd.DataFrame({'observado': obs, 'estimado': pred,
                                              'negativa': pred < 0})
                        salvar_csv(saida, pasta / 'tab', f'ajuste_{alvo}.csv')
                        grafico_observado_previsto(obs, pred, f'C{g:02d} | {alvo} — AJUSTE',
                                                   f'ajuste_{alvo}', pasta / 'fig')
                        print(f'C{g:02d}: {linha["equacao"]}')
                except (ValueError, RuntimeError, ArithmeticError) as exc:
                    linha.update(status='não ajustado', motivo=str(exc))
                linhas_grupo.append(linha)
                registros.append(linha)
        salvar_csv(pd.DataFrame(linhas_grupo), pasta / 'tab', 'equacoes.csv', index=False)
    tabela = pd.DataFrame(registros)
    salvar_csv(tabela, PASTA_TABELAS, 'regressoes.csv', index=False)
    print('Formas potenciais exigem valores positivos. Não se acrescenta epsilon a vazões nulas.')
    print('Previsões lineares negativas são mantidas e sinalizadas; não são truncadas.')
    return modelos, tabela


def finalizar(resumo):
    resumo.update({'analise': TIPO_ANALISE, 'pasta': str(PASTA_RESULTADOS),
                   'K': K, 'metodo_regional': METODO_REGIONAL,
                   'atributos_cluster': ATRIBUTOS_CLUSTER,
                   'estados': ESTADOS_INCLUIR, 'variaveis_regressao': VARIAVEIS_REGRESSAO,
                   'n_figuras': len(list(PASTA_RESULTADOS.rglob('*.png')))})
    salvar_json(resumo, PASTA_RESULTADOS, 'resumo.json')
    print('\nConcluído. Resultados: ' + str(PASTA_RESULTADOS))

# %% 00E | FUNÇÕES ESPECÍFICAS DA ANÁLISE
def curva_permanencia(serie):
    q = np.asarray(serie, dtype=float)
    q = np.sort(q[np.isfinite(q)])[::-1]
    if len(q) < 2:
        raise ValueError('Série diária insuficiente.')
    m = np.arange(1, len(q) + 1)
    return pd.DataFrame({'m': m, 'q_m3s': q, 'permanencia_pct': 100*m/(len(q)+1)})


def vazao_permanencia(serie, permanencia=90):
    """Interpolação de Weibull, igual à convenção dos slides 4–5."""
    tab = curva_permanencia(serie)
    p = tab.permanencia_pct.to_numpy()
    if not p.min() <= permanencia <= p.max():
        raise ValueError('A permanência está fora das posições amostrais disponíveis.')
    return float(np.interp(permanencia, p, tab.q_m3s))
def estatisticas_minimas(vazoes, permanencias):
    """Uma Qp por bacia; não extrai mínimas anuais nem calcula Q7,10."""
    linhas = []
    for codigo in vazoes.columns:
        q = vazoes[codigo]
        row = {'id_bacia': codigo, 'dias_validos': int(q.count()),
               'anos_aceitos': int((cobertura_por_ano(q, MES_INICIO_ANO_HIDROLOGICO)
                                   >= 1-MAX_FALHAS_ANO_PCT/100-1e-12).sum())}
        for p in permanencias:
            row[f'Q{p:g}'] = vazao_permanencia(q, p)
        linhas.append(row)
    return pd.DataFrame(linhas).set_index('id_bacia') if linhas else pd.DataFrame()


def plotar_permanencia(serie, codigo):
    tab = curva_permanencia(serie)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tab.permanencia_pct, tab.q_m3s, label='Série diária')
    for p in PERMANENCIAS:
        valor = vazao_permanencia(serie, p)
        ax.scatter([p], [valor], label=f'Q{p:g} = {valor:.3f} m³/s')
        ax.axvline(p, ls=':', lw=.7)
    if tab.q_m3s.gt(0).all(): ax.set_yscale('log')
    ax.set(title=f'{codigo} — curva de permanência', xlabel='Excedência diária (%)', ylabel='Vazão (m³/s)')
    ax.grid(alpha=.25); ax.legend()
    figura_aula(fig, '02_permanencia')
    salvar_csv(tab, PASTA_TABELAS, '02_permanencia.csv', index=False)


def curvas_minimas_por_cluster(vazoes, grupos, estatisticas):
    for g in sorted(grupos.unique()):
        ids = grupos.index[grupos == g]
        pasta = pasta_cluster(g)
        salvar_csv(estatisticas.loc[ids], pasta / 'tab', 'Qref.csv')
        fig, ax = plt.subplots(figsize=(9, 5))
        for codigo in ids:
            tab = curva_permanencia(vazoes[codigo])
            recorte = tab.loc[tab.permanencia_pct >= 50]
            ax.plot(recorte.permanencia_pct, recorte.q_m3s, lw=1, alpha=.7, label=codigo)
        ax.set(title=f'C{g:02d} — trecho de baixas vazões (sem normalizar)',
               xlabel='Excedência diária (%)', ylabel='Vazão (m³/s)')
        if len(ids) <= 15: ax.legend(fontsize=7, ncol=3)
        ax.grid(alpha=.25)
        figura_aula(fig, 'permanencia', pasta / 'fig')
    print('A avaliação das regiões de mínimas usa curvas, resíduos e validação espacial; não usa H1 de cheias.')


def validar_minimas(atributos, estatisticas, alvos):
    registros = []
    for passo, codigo in enumerate(atributos.index, 1):
        treino = atributos.drop(index=codigo)
        alvo = atributos.loc[[codigo]]
        falha_grupo = ''
        ids = []
        g = None
        try:
            ag = agrupar_bacias(treino, ATRIBUTOS_CLUSTER, K, METODO_REGIONAL, SEMENTE)
            g = int(atribuir_bacia(ag, alvo).iloc[0])
            ids = ag['labels'].index[ag['labels'] == g]
            if len(ids) < MIN_BACIAS_POR_GRUPO:
                raise ValueError(f'Grupo do alvo com {len(ids)} postos; mínimo {MIN_BACIAS_POR_GRUPO}.')
        except (ValueError, RuntimeError, ArithmeticError) as exc:
            falha_grupo = str(exc)
        for nome in alvos:
            tipos = ['regional'] + (['global'] if COMPARAR_EQUACAO_GLOBAL else [])
            for tipo in tipos:
                row = {'id_bacia': codigo, 'alvo': nome, 'modelo': tipo,
                       'regiao_no_fold': g, 'n_treino_regiao': len(ids),
                       'observado': float(estatisticas.at[codigo, nome]),
                       'previsto': np.nan, 'status': 'pendente'}
                try:
                    if tipo == 'regional' and falha_grupo:
                        raise ValueError(falha_grupo)
                    usados = ids if tipo == 'regional' else treino.index
                    modelo = ajustar_regressao(treino.loc[usados], estatisticas.loc[usados, nome], FORMA_REGRESSAO)
                    pred = float(prever_regressao(modelo, alvo).iloc[0])
                    row.update(previsto=pred, status='previsão negativa — investigar' if pred < 0 else 'ok')
                except (ValueError, RuntimeError, ArithmeticError) as exc:
                    row['status'] = str(exc)
                registros.append(row)
        if passo % 10 == 0 or passo == len(atributos):
            print(f'Validação mínimas: {passo}/{len(atributos)} postos.')
    return pd.DataFrame(registros)


def exportar_validacao_minimas(validacao):
    salvar_csv(validacao, PASTA_TABELAS, '08_validacao.csv', index=False)
    if validacao.empty:
        return pd.DataFrame()
    metricas = []
    for (alvo, modelo), parte in validacao.groupby(['alvo', 'modelo']):
        metricas.append({'alvo': alvo, 'modelo': modelo, 'n_tentativas': len(parte),
                         **metricas_regressao(parte.observado, parte.previsto)})
        grafico_observado_previsto(parte.observado, parte.previsto,
                                  f'{alvo} — validação {modelo}', f'08_cv_{alvo}_{modelo}')
    resumo = pd.DataFrame(metricas)
    salvar_csv(resumo, PASTA_TABELAS, '08_metricas.csv', index=False)
    # Comparação justa quando global e regional têm coberturas de previsão distintas.
    if COMPARAR_EQUACAO_GLOBAL:
        pareadas = []
        for nome, parte in validacao.groupby('alvo'):
            reg = parte[parte.modelo == 'regional'].set_index('id_bacia')
            glob = parte[parte.modelo == 'global'].set_index('id_bacia')
            ids = reg.index.intersection(glob.index)
            ids = ids[np.isfinite(reg.loc[ids, 'previsto']) & np.isfinite(glob.loc[ids, 'previsto'])]
            for tipo, quadro in [('regional', reg), ('global', glob)]:
                pareadas.append({'alvo': nome, 'modelo': tipo,
                                 **metricas_regressao(quadro.loc[ids, 'observado'], quadro.loc[ids, 'previsto'])})
        salvar_csv(pd.DataFrame(pareadas), PASTA_TABELAS, '08_metricas_pareadas.csv', index=False)
    print(resumo.round(3).to_string(index=False))
    return resumo


# %% 01 | LEITURA, FILTRO POR ESTADO E SELEÇÃO DAS SÉRIES
verificar_configuracao()
config_selecao = criar_configuracao_selecao()
PASTA_RESULTADOS = iniciar_saidas(TIPO_ANALISE)
PASTA_TABELAS = PASTA_RESULTADOS / 'tab'
PASTA_FIGURAS = PASTA_RESULTADOS / 'fig'
anunciar('01', 'Leitura CAMELS e seleção', 'Quais postos atendem ao recorte e à qualidade exigida?')
base = preparar_base_camels(ARQUIVO_ATRIBUTOS, PASTA_VAZAO, PASTA_PRECIPITACAO,
                            config=config_selecao, estados_incluir=ESTADOS_INCLUIR,
                            arquivo_malha=ARQUIVO_MAPA_BRASIL)
exportar_inventario(base)
plotar_selecao(base)
atributos, vazoes, atributos_alvo, vazao_alvo_oculta, CODIGO_ALVO, CLUSTERS_DISPONIVEIS = preparar_treino(base)
# Estações com vazão aprovada mas atributos incompletos podem ser usadas no exemplo local.
# A bacia reservada não entra nessa demonstração nem na tabela usada nos ajustes.
vazoes_locais = base['vazoes_selecionadas'].drop(columns=[CODIGO_ALVO], errors='ignore').copy()
DADOS_LOCAIS_DISPONIVEIS = len(vazoes_locais.columns) > 0
BACIA_EXEMPLO_EFETIVA = None
if DADOS_LOCAIS_DISPONIVEIS:
    BACIA_EXEMPLO_EFETIVA = (BACIA_EXEMPLO if BACIA_EXEMPLO in vazoes_locais else sorted(vazoes_locais.columns)[0])
    print(f'Posto usado na demonstração local: {BACIA_EXEMPLO_EFETIVA}.')
else:
    print('Nenhuma série disponível para o exemplo local. Consulte tab/01_selecao.csv.')
# Variáveis sempre inicializadas: evitar reaproveitamento acidental de execuções anteriores.
agrupamentos, grupos, regiao_alvo = {}, pd.Series(dtype=int, name='regiao'), None
modelos, validacao, avaliacao_alvo = {}, pd.DataFrame(), pd.DataFrame()

# %% 02 | HIDROGRAMA E CURVA DE PERMANÊNCIA — slides 4–5
anunciar('02', 'Vazões de referência', 'Qual vazão é igualada ou superada em 90% e 98% dos dias?')
if not PERMANENCIAS or any(not 0 < p < 100 for p in PERMANENCIAS):
    raise ValueError('PERMANENCIAS deve conter percentuais estritamente entre 0 e 100.')
PERMANENCIAS = sorted(set(float(p) for p in PERMANENCIAS))
ALVOS = [f'Q{p:g}' for p in PERMANENCIAS]
if DADOS_LOCAIS_DISPONIVEIS:
    serie_exemplo = vazoes_locais[BACIA_EXEMPLO_EFETIVA]
    plotar_hidrograma(serie_exemplo, BACIA_EXEMPLO_EFETIVA)
    plotar_permanencia(serie_exemplo, BACIA_EXEMPLO_EFETIVA)
else:
    print('02: sem dados locais aprovados.')

# %% 03 | Q90/Q98 POR BACIA — não depende de máximas anuais
anunciar('03', 'Tabela de mínimas de referência')
estatisticas_locais = estatisticas_minimas(vazoes_locais, PERMANENCIAS)
estatisticas = estatisticas_locais.reindex(atributos.index)
salvar_csv(estatisticas_locais, PASTA_TABELAS, '03_Qref.csv')
print(estatisticas_locais.round(4).head().to_string())
print('Qp usa todos os dias válidos da janela da série aprovada; o limite anual serve para contar anos aceitos.')
print('Este script não calcula máximas anuais, L-momentos, H1, GEV ou Q7,10.')

# %% 04 | K-MEANS E WARD: CLUSTERS PRÓPRIOS PARA MÍNIMAS — slides 8–12
anunciar('04', 'Agrupamentos de mínimas', 'Esses atributos ajudam a transferir Q90/Q98?')
if CLUSTERS_DISPONIVEIS:
    agrupamentos, grupos, regiao_alvo = executar_clusters(atributos, atributos_alvo)
else:
    print('04: agrupamento não executado; verifique o número de postos, K e os filtros.')

# %% 05 | CURVAS E ESTATÍSTICAS SEPARADAS POR CLUSTER — slides 25–27
anunciar('05', 'Permanência por cluster')
if CLUSTERS_DISPONIVEIS:
    curvas_minimas_por_cluster(vazoes, grupos, estatisticas)
else:
    print('05: sem clusters disponíveis.')

# %% 06 | REGRESSÕES DE Q90/Q98 POR CLUSTER — slides 25–27
anunciar('06', 'Regressões independentes por cluster')
if CLUSTERS_DISPONIVEIS:
    modelos, tabela_regressoes = ajustar_modelos_por_cluster(atributos, estatisticas, grupos, ALVOS)
else:
    tabela_regressoes = pd.DataFrame()
    print('06: sem clusters disponíveis.')

# %% 07 | APLICAÇÃO NA BACIA RESERVADA
anunciar('07', 'Previsões para a bacia sem posto')
previsoes_alvo = []
if CLUSTERS_DISPONIVEIS and regiao_alvo is not None:
    for nome in ALVOS:
        modelo = modelos.get(regiao_alvo, {}).get(nome)
        if modelo is None:
            previsoes_alvo.append({'id_bacia': CODIGO_ALVO, 'regiao': regiao_alvo, 'alvo': nome,
                                   'previsto_m3s': np.nan, 'status': 'Modelo regional indisponível'})
            continue
        valor = float(prever_regressao(modelo, atributos_alvo).iloc[0])
        previsoes_alvo.append({'id_bacia': CODIGO_ALVO, 'regiao': regiao_alvo, 'alvo': nome,
                               'previsto_m3s': valor, 'status': 'negativa — investigar' if valor < 0 else 'ok'})
    tabela_alvo = pd.DataFrame(previsoes_alvo)
    salvar_csv(tabela_alvo, pasta_cluster(regiao_alvo) / 'tab', 'alvo_previsto.csv', index=False)
    print(tabela_alvo.to_string(index=False))
    valores = tabela_alvo.previsto_m3s.to_numpy()
    if np.isfinite(valores).all() and np.any(np.diff(valores) > 0):
        print('AVISO: previsões não respeitam Q90 >= Q98; revisar as regressões. Não houve correção automática.')
else:
    print('07: sem bacia reservada ou sem clusters.')

# %% 08 | VALIDAÇÃO ESPACIAL DE MÍNIMAS — slides 28–29
anunciar('08', 'Uma bacia fora por vez')
if CLUSTERS_DISPONIVEIS and EXECUTAR_VALIDACAO_ESPACIAL:
    validacao = validar_minimas(atributos, estatisticas, ALVOS)
    metricas_validacao = exportar_validacao_minimas(validacao)
    print('Os atributos são padronizados e os clusters refeitos dentro de cada treinamento.')
    print('A equação global, quando habilitada, é apenas uma referência de comparação; não substitui as regionais.')
else:
    print('08: validação desativada ou sem clusters suficientes.')

# %% 09 | REVELAR O ALVO E RESUMIR
anunciar('09', 'Comparação com a bacia reservada')
if previsoes_alvo and vazao_alvo_oculta is not None:
    avaliacao_alvo = pd.DataFrame(previsoes_alvo)
    observados = {f'Q{p:g}': vazao_permanencia(vazao_alvo_oculta, p) for p in PERMANENCIAS}
    avaliacao_alvo['referencia_m3s'] = avaliacao_alvo.alvo.map(observados)
    avaliacao_alvo['erro_m3s'] = avaliacao_alvo.previsto_m3s - avaliacao_alvo.referencia_m3s
    denom = avaliacao_alvo.referencia_m3s.where(avaliacao_alvo.referencia_m3s != 0)
    avaliacao_alvo['erro_relativo_pct'] = 100 * avaliacao_alvo.erro_m3s / denom
    salvar_csv(avaliacao_alvo, pasta_cluster(regiao_alvo) / 'tab', 'alvo_revelado.csv', index=False)
    print(avaliacao_alvo.round(3).to_string(index=False))
finalizar({'n_postos_inventario': len(base['selecao']), 'n_treino': len(atributos),
           'clusters_executados': CLUSTERS_DISPONIVEIS, 'bacia_alvo': CODIGO_ALVO,
           'permanencias': PERMANENCIAS, 'n_validacao': len(validacao)})
