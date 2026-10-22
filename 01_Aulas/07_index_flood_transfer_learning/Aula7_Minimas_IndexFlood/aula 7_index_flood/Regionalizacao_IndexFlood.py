# -*- coding: utf-8 -*-
"""Regionalização de máximas diárias anuais — index-flood.
Versão dividida a partir de Regionalizacao_CAMELS_Filtro_UF.py.
Arquivo independente, organizado em células # %% para o Spyder.
Não importa o outro script nem módulos próprios do projeto.
Máximas de médias diárias, não picos instantâneos. H1 preservado; não calcula Z_DIST, H2 ou H3.
Resultados: aula 7_index_flood/resultados/indexflood/<execução>/.
Regressões e aplicações por cluster; os estados filtram a UF do posto.
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
from scipy.stats import kappa4
try:
    from lmoments3 import distr
except ImportError:
    distr = None
    warnings.warn('lmoments3 ausente: leitura e regressões continuam disponíveis, mas H1/GEV/PE3 exigem python -m pip install lmoments3==1.0.8.')

# %% 01 | CONFIGURAÇÕES — ALTERE AQUI

# Usa a pasta onde está o script.
# Sem __file__, usa o diretório de trabalho atual.
PASTA = (
    Path(__file__).resolve().parent
    if '__file__' in globals()
    else Path.cwd()
)

PASTA_AULA = PASTA
PASTA_CAMELS = PASTA / 'amostra_camels'

ARQUIVO_ATRIBUTOS = PASTA_CAMELS / 'Caracteristicas_bacias_CAMELS.xlsx'
PASTA_VAZAO = PASTA_CAMELS / 'Vazão'
PASTA_PRECIPITACAO = PASTA_CAMELS / 'Precipitação'

BUSCAR_EM_SUBPASTAS = True

print(f'Pasta dos dados: {PASTA_CAMELS}')

# Filtros preservados da última versão do script.
ANO_INICIAL_ANALISE = 1980
ANO_FINAL_ANALISE = 2010
ANOS_MINIMOS = 20
MAX_FALHAS_PERIODO_PCT = 10.0
MAX_FALHAS_ANO_PCT = 10
MES_INICIO_ANO_HIDROLOGICO = 1
FLAGS_VAZAO_ACEITAS = None
IDS_INCLUIR = None
ESTADOS_INCLUIR = ['SC', 'PR']  # Ex.: ['SC'] ou ['PR', 'SC', 'RS']; UF do POSTO.

FONTE_PRECIPITACAO = 'serie'
PRODUTO_PRECIPITACAO = 'p_chirps'
ANOS_MINIMOS_PRECIPITACAO = 10
# 'serie': média dos totais de anos civis completos. 'atributo': p_mean * 365.2425.

# Definir os atributos adequados a ESTA análise. Não se impõe que mínimas
# e máximas tenham os mesmos grupos. Defaults preservam a última configuração.
ATRIBUTOS_CLUSTER = [
    'precip_mm_ano',
    'slope_mean',
    'elev_gauge',
    'gauge_lat',
    'gauge_lon',
]
K = 7
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
# Simples: área. Múltipla: área e precipitação. Modelos potenciais usam log + smearing.
EXECUTAR_ETAPAS_REGIONAIS = True
EXECUTAR_VALIDACAO_ESPACIAL = True

MOSTRAR_GRAFICOS = True
ANOS_HIDROGRAMA = 3
DPI_FIGURAS = 160
PLOTAR_MAPAS = True
ARQUIVO_MAPA_BRASIL = PASTA / 'mapas' / 'brasil_uf_ibge.geojson'
MAPA_REGIOES_ZOOM = True
ROTULAR_BACIAS_NO_MAPA = False

TIPO_ANALISE = 'indexflood'
N_SIM_H1 = 500
N_SIM_H1_VALIDACAO = 200
FAMILIA_APLICACAO = 'GEV'
TEMPOS_RETORNO = np.array([2, 5, 10, 20, 50, 100], dtype=float)
# Máximas com falhas anuais são sensíveis à perda do dia de cheia; teste 0% separadamente.

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
def lmom_amostrais(x):
    """Calcule n, l1, l2, t, t3 e t4 por PWMs amostrais não enviesados.

    ``x`` deve ser uma série unidimensional com pelo menos quatro observações
    válidas. NaNs representam falhas e são omitidos; infinitos provocam erro.
    O número n devolvido é o total de observações válidas.

    Para os dados em ordem crescente, com j=0,...,n-1:
        b_r = média[comb(j,r) / comb(n-1,r) * x_(j+1)].
    Assim, l1=b0, l2=2b1-b0, l3=6b2-6b1+b0 e
    l4=20b3-30b2+12b1-b0. As razões são t=l2/l1, t3=l3/l2 e t4=l4/l2.

    PWMs e L-momentos são estimadores não enviesados; as razões de L-momentos
    não são, em geral, não enviesadas. Séries constantes não têm t3/t4 definidos.
    """
    valores = np.asarray(x, dtype=float)
    if valores.ndim != 1:
        raise ValueError("A série para L-momentos deve ser unidimensional.")
    if np.isinf(valores).any():
        raise ValueError("A série contém valores infinitos.")
    valores = np.sort(valores[~np.isnan(valores)])
    n = valores.size
    if n < 4:
        raise ValueError("São necessárias pelo menos quatro observações válidas.")

    j = np.arange(n, dtype=float)
    b0 = valores.mean()
    b1 = np.mean(valores * j / (n - 1))
    b2 = np.mean(valores * j * (j - 1) / ((n - 1) * (n - 2)))
    b3 = np.mean(
        valores * j * (j - 1) * (j - 2)
        / ((n - 1) * (n - 2) * (n - 3))
    )
    l1 = b0
    l2 = 2.0 * b1 - b0
    l3 = 6.0 * b2 - 6.0 * b1 + b0
    l4 = 20.0 * b3 - 30.0 * b2 + 12.0 * b1 - b0

    if not np.isfinite([l1, l2, l3, l4]).all():
        raise ValueError("Os L-momentos calculados não são finitos.")
    if valores[0] == valores[-1] or l2 <= 0:
        raise ValueError("Série constante ou L-escala não positiva: t3/t4 indefinidos.")
    if l1 == 0:
        raise ValueError("Média amostral igual a zero: o L-CV não está definido.")

    return {
        "n": int(n),
        "l1": float(l1),
        "l2": float(l2),
        "t": float(l2 / l1),
        "t3": float(l3 / l2),
        "t4": float(l4 / l2),
    }


def _ler_tabela_lmom(df_lmom):
    """Leia colunas por nome, preservando a ordem dos postos na entrada."""
    obrigatorias = ("n", "l1", "l2", "t", "t3", "t4")
    colunas = {}
    for nome in obrigatorias:
        try:
            coluna = np.asarray(df_lmom[nome], dtype=float)
        except (KeyError, TypeError) as erro:
            raise ValueError(f"A tabela deve conter a coluna '{nome}'.") from erro
        if coluna.ndim != 1:
            raise ValueError(f"A coluna '{nome}' deve ser unidimensional.")
        colunas[nome] = coluna

    n_postos = len(colunas["n"])
    if n_postos == 0 or any(len(v) != n_postos for v in colunas.values()):
        raise ValueError("A tabela está vazia ou suas colunas têm tamanhos diferentes.")
    if not all(np.isfinite(v).all() for v in colunas.values()):
        raise ValueError("A tabela de L-momentos contém valores ausentes ou infinitos.")
    if np.any(colunas["n"] < 4) or np.any(colunas["n"] % 1 != 0):
        raise ValueError("Comprimentos dos registros devem ser inteiros de pelo menos 4.")
    if np.any(colunas["l1"] <= 0) or np.any(colunas["l2"] <= 0):
        raise ValueError("A análise regional exige médias e L-escalas positivas.")
    if np.any(colunas["t"] <= 0) or np.any(colunas["t"] >= 1):
        raise ValueError("O L-CV dos postos deve estar estritamente entre 0 e 1.")
    if np.any(np.abs(colunas["t3"]) >= 1) or np.any(np.abs(colunas["t4"]) >= 1):
        raise ValueError("t3 e t4 dos postos devem estar estritamente entre -1 e 1.")
    if not np.allclose(
        colunas["t"], colunas["l2"] / colunas["l1"], rtol=1e-7, atol=1e-10
    ):
        raise ValueError("Coluna t inconsistente com l2/l1; use valores sem arredondamento.")
    colunas["n"] = colunas["n"].astype(int)
    return colunas


def lmom_regional(df_lmom):
    """Retorne t, t3 e t4 regionais, ponderados pelos comprimentos n dos postos.

    A média regional normalizada é l1=1; portanto sua L-escala é l2=t regional.
    As colunas l1 e l2 permanecem úteis para verificar t=l2/l1. Não se ajusta uma
    distribuição à concatenação das séries para substituir esse cálculo.
    """
    colunas = _ler_tabela_lmom(df_lmom)
    pesos = colunas["n"].astype(float)
    return {
        nome: float(np.average(colunas[nome], weights=pesos))
        for nome in ("t", "t3", "t4")
    }


def heterogeneidade_h1(df_lmom, n_sim=500, semente=42):
    """Calcule H1 com uma região nula Kappa ajustada aos L-momentos regionais.

    Entrada: DataFrame, ou mapeamento de vetores, com n,l1,l2,t,t3,t4 por posto.
    O peso de cada posto é n, seu número de máximas diárias anuais válidas.
    Sugestão didática: 500–1.000 réplicas; aumentar para avaliar estabilidade.

    V_obs = sqrt[soma(n_i*(t_i-t_R)**2)/soma(n_i)].
    Em CADA réplica calcula-se novamente t_R_sim antes de V_sim. A variabilidade
    entre réplicas é o desvio-padrão amostral (ddof=1).

    O único fallback é a Kappa com h=-1 (GLO), quando t4 regional atinge ou
    ultrapassa (1+5*t3**2)/6. Outras falhas de ajuste são reportadas. O caso de
    igualdade usa a distribuição limite GLO, pois lmoments3 rejeita esse limite
    no ajuste Kappa irrestrito. Não há truncamento nem substituição de valores
    simulados, pois isso mudaria a distribuição nula.

    O método não testa homogeneidade de Q90: sua aplicação aqui se refere às
    distribuições das máximas diárias anuais, depois de remover a escala local.
    """
    try:
        from lmoments3 import distr
    except ImportError as erro:
        raise ImportError(
            "O cálculo de H1 requer lmoments3. Instale com: pip install lmoments3"
        ) from erro

    if isinstance(n_sim, (bool, np.bool_)) or int(n_sim) != n_sim or n_sim < 2:
        raise ValueError("n_sim deve ser um inteiro de pelo menos 2.")
    n_sim = int(n_sim)
    colunas = _ler_tabela_lmom(df_lmom)
    n = colunas["n"]
    if n.size < 2:
        raise ValueError("H1 requer uma região com pelo menos dois postos.")
    pesos = n / n.sum()
    regional = lmom_regional(df_lmom)
    tR, t3R, t4R = (regional[chave] for chave in ("t", "t3", "t4"))
    V_obs = np.sqrt(np.sum(pesos * (colunas["t"] - tR) ** 2))

    limite_glo = (1.0 + 5.0 * t3R**2) / 6.0
    if t4R >= limite_glo:
        try:
            p_glo = distr.glo.lmom_fit(lmom_ratios=[1.0, tR, t3R])
        except Exception as erro:
            raise RuntimeError("Falha ao ajustar a distribuição limite GLO.") from erro
        parametros = {nome: float(p_glo[nome]) for nome in ("k", "loc", "scale")}
        parametros["h"] = -1.0
        metodo = "Kappa com h=-1 (limite GLO)"
    else:
        try:
            p_kappa = distr.kap.lmom_fit(lmom_ratios=[1.0, tR, t3R, t4R])
        except Exception as erro:
            raise RuntimeError(
                "O ajuste Kappa não convergiu ou os L-momentos regionais são "
                "incompatíveis com a família. Revise os dados e o agrupamento; "
                "esta falha não autoriza um fallback automático. "
                f"Detalhe da biblioteca: {erro}"
            ) from erro
        parametros = {
            nome: float(p_kappa[nome]) for nome in ("k", "h", "loc", "scale")
        }
        metodo = "Kappa de quatro parâmetros"

    if not np.isfinite(list(parametros.values())).all() or parametros["scale"] <= 0:
        raise RuntimeError("O ajuste produziu parâmetros Kappa inválidos.")
    nula = kappa4(
        h=parametros["h"], k=parametros["k"],
        loc=parametros["loc"], scale=parametros["scale"],
    )
    rng = np.random.default_rng(semente)

    # Todas as réplicas de um posto são sorteadas de uma vez. Postos podem ter
    # comprimentos diferentes; por isso o laço externo percorre apenas os postos.
    t_sim = np.empty((n_sim, n.size), dtype=float)
    for i, n_i in enumerate(n):
        simuladas = nula.rvs(size=(n_sim, int(n_i)), random_state=rng)
        if not np.isfinite(simuladas).all():
            raise RuntimeError("A distribuição nula gerou valores não finitos.")
        ordenadas = np.sort(simuladas, axis=1)
        j = np.arange(n_i, dtype=float)
        l1_sim = ordenadas.mean(axis=1)
        b1_sim = (ordenadas * (j / (n_i - 1))).mean(axis=1)
        l2_sim = 2.0 * b1_sim - l1_sim
        if np.any(l1_sim <= 0) or np.any(l2_sim <= 0):
            raise RuntimeError(
                "A distribuição nula gerou uma amostra com média ou L-escala "
                "não positiva. Não se reamostrou nem truncou essa réplica. "
                "Revise a adequação da Kappa à região."
            )
        t_sim[:, i] = l2_sim / l1_sim

    tR_sim = t_sim @ pesos
    V_sim = np.sqrt(((t_sim - tR_sim[:, None]) ** 2) @ pesos)
    media_V = V_sim.mean()
    dp_V = V_sim.std(ddof=1)
    if not np.isfinite(V_sim).all() or not np.isfinite(dp_V) or dp_V <= 0:
        raise RuntimeError("Dispersão simulada inválida para padronizar H1.")
    H1 = (V_obs - media_V) / dp_V
    if H1 < 1.0:
        classe = "aceitavelmente homogênea"
    elif H1 < 2.0:
        classe = "possivelmente heterogênea"
    else:
        classe = "heterogênea"

    return {
        "H1": float(H1),
        "V_obs": float(V_obs),
        "V_sim": V_sim,
        "media_V": float(media_V),
        "dp_V": float(dp_V),
        "classe": classe,
        "kappa_params": parametros,
        "metodo_simulacao": metodo,
        "regionais": {"l1": 1.0, "l2": tR, **regional},
        "n_postos": int(n.size),
        "n_registros": n.copy(),
        "n_sim": n_sim,
        "semente": semente,
        "hipotese_nula": (
            "Mesma distribuição adimensional; independência entre postos e anos; "
            "comprimentos de registro preservados."
        ),
    }


def discordancia_di(df_lmom):
    """Calcule D_i a partir de t,t3,t4 e da média NÃO ponderada dos postos.

    A = soma[(u_i-u_medio)(u_i-u_medio)^T];
    D_i = N/3 * (u_i-u_medio)^T A^(-1) (u_i-u_medio).

    A ordem de D_i é a mesma das linhas da entrada. Os limiares de 10% e 5%
    reproduzem a tabela usada em ``regtst``; não representam uma política de
    exclusão de postos. Para N<=3 ou A singular, o diagnóstico não é calculado.
    N=4 é matematicamente limitado: não oferece um diagnóstico útil de outliers.
    """
    colunas = _ler_tabela_lmom(df_lmom)
    U = np.column_stack([colunas[chave] for chave in ("t", "t3", "t4")])
    N = U.shape[0]
    crit10 = [3, 3, 3, 3, 1.3330, 1.6481, 1.9166, 2.1401, 2.3287,
              2.4906, 2.6321, 2.7573, 2.8694, 2.9709, 3, 3, 3, 3]
    crit05 = [4, 4, 4, 4, 1.3333, 1.6648, 1.9821, 2.2728, 2.5337,
              2.7666, 2.9748, 3.1620, 3.3310, 3.4844, 3.6246, 3.7532,
              3.8718, 3.9816]
    c10 = float(crit10[N - 1] if N <= len(crit10) else 3.0)
    c05 = float(crit05[N - 1] if N <= len(crit05) else 4.0)
    desvios = U - U.mean(axis=0)
    A = desvios.T @ desvios
    if N <= 3 or np.linalg.matrix_rank(A) < 3:
        return {
            "D_i": np.full(N, np.nan), "calculavel": False,
            "limiar_10pct": c10, "limiar_5pct": c05,
            "sinalizado_10pct": np.zeros(N, dtype=bool),
            "sinalizado_5pct": np.zeros(N, dtype=bool),
            "observacao": "Poucos postos ou matriz singular; não foi usada pseudoinversa.",
        }
    D = N / 3.0 * np.sum(desvios * np.linalg.solve(A, desvios.T).T, axis=1)
    observacao = (
        "Com quatro postos, D_i tem utilidade diagnóstica limitada."
        if N == 4 else "Valores altos pedem investigação dos dados e da região."
    )
    return {
        "D_i": D, "calculavel": True,
        "limiar_10pct": c10, "limiar_5pct": c05,
        "sinalizado_10pct": D >= c10,
        "sinalizado_5pct": D >= c05,
        "observacao": observacao,
    }


def maximas_anuais(vazoes, mes_inicio=1, completude_minima=1.0):
    """Máximas das médias diárias em anos hidrológicos com cobertura suficiente.

    Quando mes_inicio=1, o rótulo é o ano civil. Nos demais casos, é o ano
    em que o intervalo termina. Ex.: out/1991–set/1992 recebe rótulo 1992.
    Completude considera 365/366 dias esperados, inclusive nos anos de borda.
    Cobertura alta não garante que uma falha não coincida com uma cheia.
    """
    if mes_inicio not in range(1, 13) or not 0 < completude_minima <= 1:
        raise ValueError('Mês ou completude inválidos.')
    d = vazoes.index
    rotulo = d.year if mes_inicio == 1 else d.year + (d.month >= mes_inicio).astype(int)
    contagem = vazoes.groupby(rotulo).count()
    maximas = vazoes.groupby(rotulo).max()
    dias_esperados = []
    for ano in maximas.index:
        inicio = pd.Timestamp(int(ano) if mes_inicio == 1 else int(ano)-1, mes_inicio, 1)
        fim = inicio + pd.DateOffset(years=1)
        dias_esperados.append((fim-inicio).days)
    cobertura = contagem.div(pd.Series(dias_esperados, index=maximas.index), axis=0)
    # Mesma tolerância do seletor CAMELS, inclusive quando um ano está no limiar.
    aceitos = (cobertura > 0) & (100*(1-cobertura) <= 100*(1-completude_minima) + 1e-10)
    maximas = maximas.where(aceitos)
    maximas.index.name = 'ano_hidrologico'
    cobertura.index.name = 'ano_hidrologico'
    return maximas, cobertura


def curva_regional(parametros, tempos, familia='GEV'):
    T = np.asarray(tempos, dtype=float)
    if np.any(T <= 1):
        raise ValueError('T deve ser maior que 1 ano.')
    dist = {'GEV': distr.gev, 'PE3': distr.pe3}[familia]
    crescimento = dist.ppf(1-1/T, **parametros)
    if not np.isfinite(crescimento).all():
        raise ValueError('Quantil regional não finito.')
    return crescimento
def estatisticas_maximas(maximas):
    linhas = []
    for codigo in maximas.columns:
        x = maximas[codigo].dropna().to_numpy(dtype=float)
        row = {'id_bacia': codigo, 'n': len(x), 'indice_m3s': float(x.mean()) if len(x) else np.nan}
        try:
            row.update(lmom_amostrais(x))
            row['status_lmomentos'] = 'ok'
        except ValueError as exc:
            row.update(l1=row['indice_m3s'], l2=np.nan, t=np.nan, t3=np.nan, t4=np.nan,
                       status_lmomentos=str(exc))
        linhas.append(row)
    return pd.DataFrame(linhas).set_index('id_bacia') if linhas else pd.DataFrame()


def curvas_index_por_cluster(maximas, estatisticas, grupos):
    """Dividir CADA coluna pelo índice daquela estação, dentro de cada cluster."""
    normalizadas = {}
    for g in sorted(grupos.unique()):
        ids = grupos.index[grupos == g]
        pasta = pasta_cluster(g)
        bruto = maximas.loc[:, ids]
        indices = estatisticas.loc[ids, 'indice_m3s'].where(lambda x: x > 0)
        z = bruto.div(indices, axis='columns')
        normalizadas[int(g)] = z
        salvar_csv(bruto, pasta / 'tab', 'maximas.csv')
        salvar_csv(z, pasta / 'tab', 'max_normalizadas.csv')
        salvar_csv(estatisticas.loc[ids], pasta / 'tab', 'indices_lmom.csv')
        for quadro, nome, rotulo in [(bruto, 'cdf_absoluta', 'Máxima diária anual (m³/s)'),
                                     (z, 'cdf_normalizada', 'Máxima anual / média das máximas da estação')]:
            fig, ax = plt.subplots(figsize=(9, 5))
            for codigo in ids:
                x = np.sort(quadro[codigo].dropna().to_numpy())
                if not len(x): continue
                F = np.arange(1, len(x)+1)/(len(x)+1)
                ax.plot(x, F, lw=1, alpha=.75, label=codigo)
            ax.set(xlabel=rotulo, ylabel='Não excedência anual F', title=f'C{g:02d} — {nome}')
            if len(ids) <= 15: ax.legend(ncol=3, fontsize=7)
            ax.grid(alpha=.25)
            figura_aula(fig, nome, pasta / 'fig')
    return normalizadas


def diagnosticos_por_cluster(estatisticas, agrupamentos):
    """D_i e H1 em cada cluster. Uma falha não é uma classificação heterogênea."""
    diagnosticos, cache, linhas_h, linhas_d = {}, {}, [], []
    for metodo, ag in agrupamentos.items():
        for g in sorted(ag['labels'].unique()):
            ids = ag['labels'].index[ag['labels'] == g]
            parcial = estatisticas.loc[ids]
            linha = {'metodo': metodo, 'regiao': int(g), 'n_postos': len(ids),
                     'H1': np.nan, 'V_obs': np.nan, 'media_V': np.nan, 'dp_V': np.nan,
                     'classe': 'não calculado', 'metodo_simulacao': '', 'motivo': ''}
            try:
                d = discordancia_di(parcial)
                for codigo, valor, sinal in zip(ids, d['D_i'], d['sinalizado_10pct']):
                    linhas_d.append({'metodo': metodo, 'regiao': int(g), 'id_bacia': codigo,
                                     'D_i': valor, 'limiar_10pct': d['limiar_10pct'],
                                     'sinalizado': bool(sinal), 'observacao': d['observacao']})
            except (ValueError, RuntimeError, ArithmeticError) as exc:
                linhas_d.append({'metodo': metodo, 'regiao': int(g), 'observacao': str(exc)})
            try:
                if len(ids) < MIN_BACIAS_POR_GRUPO:
                    raise ValueError(f'{len(ids)} postos; mínimo operacional configurado: {MIN_BACIAS_POR_GRUPO}.')
                chave = tuple(sorted(ids))
                if chave not in cache:
                    cache[chave] = heterogeneidade_h1(parcial, N_SIM_H1, SEMENTE+10000+int(g))
                diag = cache[chave]
                linha.update({k: diag[k] for k in ['H1', 'V_obs', 'media_V', 'dp_V', 'classe', 'metodo_simulacao']})
                if metodo == METODO_REGIONAL:
                    diagnosticos[int(g)] = diag
            except (ValueError, RuntimeError, ArithmeticError, ImportError) as exc:
                linha['motivo'] = str(exc)
            linhas_h.append(linha)
            print(f"{metodo} C{g:02d}: n={len(ids)}, H1={linha['H1']:.3f}, {linha['classe']}")
    tabela_h, tabela_d = pd.DataFrame(linhas_h), pd.DataFrame(linhas_d)
    salvar_csv(tabela_h, PASTA_TABELAS, '06_Hosking_Wallis.csv', index=False)
    salvar_csv(tabela_d, PASTA_TABELAS, '06_discordancia.csv', index=False)
    grupos = agrupamentos[METODO_REGIONAL]['labels']
    for g in sorted(grupos.unique()):
        pasta = pasta_cluster(g)
        h = tabela_h[(tabela_h.metodo == METODO_REGIONAL) & (tabela_h.regiao == g)]
        d = tabela_d[(tabela_d.metodo == METODO_REGIONAL) & (tabela_d.regiao == g)]
        salvar_csv(h, pasta / 'tab', 'H1.csv', index=False)
        salvar_csv(d, pasta / 'tab', 'Di.csv', index=False)
        ids = grupos.index[grupos == g]
        parcial = estatisticas.loc[ids]
        if int(g) not in diagnosticos:
            continue
        diag = diagnosticos[int(g)]
        pesos = parcial.n / parcial.n.sum()
        tR = float(np.sum(pesos * parcial.t))
        detalhes = parcial[['n', 't', 't3', 't4']].copy()
        detalhes['t_regional'] = tR
        detalhes['contribuicao_V2'] = pesos * (parcial.t-tR)**2
        salvar_csv(detalhes, pasta / 'tab', 'auditoria_H1.csv')
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(diag['V_sim'], bins=25, alpha=.7, label='Regiões nulas simuladas')
        ax.axvline(diag['V_obs'], lw=2, label='V observado')
        ax.set(title=f"C{g:02d}: H1={diag['H1']:.3f} — {diag['classe']}",
               xlabel='Dispersão do L-CV, V', ylabel='Número de simulações')
        ax.legend(); figura_aula(fig, 'H1', pasta / 'fig')
        ordem = detalhes.sort_values('t')
        fig, ax = plt.subplots(figsize=(max(8, len(ordem)*.22), 5))
        ax.scatter(np.arange(len(ordem)), ordem.t)
        ax.axhline(tR, ls='--', label='L-CV regional ponderado')
        ax.set_xticks(np.arange(len(ordem)), ordem.index, rotation=90, fontsize=7)
        ax.set(title=f'C{g:02d} — L-CV por posto', ylabel='t = l2/l1')
        ax.legend(); ax.grid(alpha=.25)
        figura_aula(fig, 'LCV_postos', pasta / 'fig')
    print('Não se removeu nenhum posto por D_i ou H1. H1 alto permanece identificado como exploratório.')
    return diagnosticos, tabela_h


def status_cluster(g, diagnosticos):
    d = diagnosticos.get(int(g))
    if d is None:
        return 'exploratório — H1 indisponível'
    if d['H1'] >= 2:
        return 'exploratório — região heterogênea'
    if d['H1'] >= 1:
        return 'exploratório — região possivelmente heterogênea'
    return 'exemplo didático — H1 < 1; aderência ainda deve ser avaliada'


def ajustar_distribuicoes_por_cluster(estatisticas, grupos, diagnosticos):
    ajustes, registros = {}, []
    for g in sorted(grupos.unique()):
        ids = grupos.index[grupos == g]
        pasta = pasta_cluster(g)
        uso = status_cluster(g, diagnosticos)
        ajustes[int(g)] = {}
        try:
            if len(ids) < MIN_BACIAS_POR_GRUPO:
                raise ValueError('Grupo insuficiente para os ajustes configurados.')
            reg = lmom_regional(estatisticas.loc[ids])
            if distr is None:
                raise ImportError('Instale lmoments3==1.0.8 para ajustar GEV/Pearson III.')
            ajustes[int(g)]['regional'] = reg
        except (ValueError, RuntimeError, ArithmeticError, ImportError) as exc:
            registros.append({'regiao': int(g), 'familia': '', 'status': 'não ajustado', 'motivo': str(exc)})
            continue
        for nome, dist in [('GEV', distr.gev), ('PE3', distr.pe3)]:
            try:
                parametros = dict(dist.lmom_fit(lmom_ratios=[1., reg['t'], reg['t3']]))
                if not np.isfinite(list(parametros.values())).all() or parametros['scale'] <= 0:
                    raise ValueError('Parâmetros não finitos ou escala não positiva.')
                ajustes[int(g)][nome] = parametros
                registros.append({'regiao': int(g), 'familia': nome, 'status': 'ajustado', 'uso': uso,
                                  't_regional': reg['t'], 't3_regional': reg['t3'], 't4_regional': reg['t4'],
                                  't4_teorico': dist.lmom_ratios(nmom=4, **parametros)[3], **parametros})
            except (ValueError, RuntimeError, ArithmeticError) as exc:
                registros.append({'regiao': int(g), 'familia': nome, 'status': 'não ajustado', 'motivo': str(exc)})
        fig, ax = plt.subplots(figsize=(8, 5))
        for nome, dist, valores, chave in [
            ('GEV', distr.gev, np.linspace(-.45, .5, 100), 'c'),
            ('Pearson III', distr.pe3, np.linspace(-3., 4., 100), 'skew')]:
            razoes = np.array([dist.lmom_ratios(nmom=4, **{chave:v, 'loc':0., 'scale':1.}) for v in valores])
            ax.plot(razoes[:, 2], razoes[:, 3], label=nome)
        ax.scatter(estatisticas.loc[ids, 't3'], estatisticas.loc[ids, 't4'], label='Postos')
        ax.scatter([reg['t3']], [reg['t4']], marker='*', s=160, label='Média regional')
        ax.set(title=f'C{g:02d} — diagrama de L-momentos', xlabel='L-assimetria t3', ylabel='L-curtose t4')
        ax.legend(); ax.grid(alpha=.25); figura_aula(fig, 'L_momentos', pasta / 'fig')
        fig, ax = plt.subplots(figsize=(8, 5))
        curva = pd.DataFrame({'T_anos': TEMPOS_RETORNO})
        Ts = np.geomspace(1.05, max(200., float(max(TEMPOS_RETORNO))), 150)
        for nome in ('GEV', 'PE3'):
            if nome in ajustes[int(g)]:
                pars = ajustes[int(g)][nome]
                curva[f'gR_{nome}'] = curva_regional(pars, TEMPOS_RETORNO, nome)
                ax.plot(Ts, curva_regional(pars, Ts, nome), label=nome)
        curva['uso'] = uso
        salvar_csv(curva, pasta / 'tab', 'curva_regional.csv', index=False)
        ax.set_xscale('log'); ax.set(title=f'C{g:02d} — curva de crescimento (candidatas)',
                                    xlabel='T (anos)', ylabel='gR(T) adimensional')
        ax.grid(alpha=.25)
        if len(ax.lines): ax.legend()
        figura_aula(fig, 'crescimento', pasta / 'fig')
    tabela = pd.DataFrame(registros)
    salvar_csv(tabela, PASTA_TABELAS, '07_distribuicoes.csv', index=False)
    for g in sorted(grupos.unique()):
        salvar_csv(tabela.loc[tabela.regiao == g], pasta_cluster(g) / 'tab', 'distribuicoes.csv', index=False)
    print(f"Distribuições ajustadas: {int((tabela['status'] == 'ajustado').sum())}. Consulte 07_distribuicoes.csv para falhas.")
    print('Os ajustes utilizam [l1=1, l2=tR, t3R], separadamente por cluster; não se concatenam grupos distintos.')
    print('Z_DIST não implementado; GEV e PE3 são candidatas, não distribuições automaticamente aprovadas.')
    return ajustes, tabela


def aplicar_index_todos_clusters(atributos, estatisticas, grupos, modelos, distribuicoes, diagnosticos):
    registros = []
    for g in sorted(grupos.unique()):
        ids = grupos.index[grupos == g]
        modelo = modelos.get(int(g), {}).get('indice_m3s')
        if modelo is None:
            print(f'C{g:02d}: sem regressão do índice; aplicação não executada.')
            continue
        mu = prever_regressao(modelo, atributos.loc[ids])
        locais = []
        for familia in ('GEV', 'PE3'):
            pars = distribuicoes.get(int(g), {}).get(familia)
            if pars is None: continue
            crescimento = curva_regional(pars, TEMPOS_RETORNO, familia)
            for codigo in ids:
                for T, fator in zip(TEMPOS_RETORNO, crescimento):
                    indice_estimado = float(mu.loc[codigo])
                    row = {'id_bacia': codigo, 'regiao': int(g), 'familia': familia, 'T_anos': T,
                           'indice_observado': float(estatisticas.at[codigo, 'indice_m3s']),
                           'indice_estimado': indice_estimado, 'gR': fator,
                           'Q_indice_observado': float(estatisticas.at[codigo, 'indice_m3s'])*fator,
                           'Q_indice_regressao': indice_estimado*fator if indice_estimado > 0 else np.nan,
                           'uso': status_cluster(g, diagnosticos),
                           'status_indice': 'positivo' if indice_estimado > 0 else 'não positivo — investigar'}
                    locais.append(row); registros.append(row)
        salvar_csv(pd.DataFrame(locais), pasta_cluster(g) / 'tab', 'Q_T_postos.csv', index=False)
    tabela = pd.DataFrame(registros)
    salvar_csv(tabela, PASTA_TABELAS, '09_Q_T_postos.csv', index=False)
    print(f'Q(T): {len(tabela)} linhas calculadas; cada aplicação usa a regressão e a curva de seu próprio cluster.')
    print('Q_indice_observado também é uma estimativa regional, não um quantil diretamente observado.')
    print('Q_indice_regressao nos postos de treinamento é resultado de AJUSTE, não validação.')
    return tabela


def validar_index(atributos, estatisticas):
    linhas = []
    for passo, codigo in enumerate(atributos.index, 1):
        treino = atributos.drop(index=codigo)
        alvo = atributos.loc[[codigo]]
        row = {'id_bacia': codigo, 'indice_observado': float(estatisticas.at[codigo, 'indice_m3s']),
               'indice_previsto': np.nan, 'Q50_previsto': np.nan, 'H1_treino': np.nan, 'status': ''}
        try:
            ag = agrupar_bacias(treino, ATRIBUTOS_CLUSTER, K, METODO_REGIONAL, SEMENTE)
            g = int(atribuir_bacia(ag, alvo).iloc[0])
            ids = ag['labels'].index[ag['labels'] == g]
            row.update(regiao_no_fold=g, n_treino_regiao=len(ids))
            if len(ids) < MIN_BACIAS_POR_GRUPO:
                raise ValueError('Poucos postos no grupo de treinamento.')
            stats = estatisticas.loc[ids]
            modelo = ajustar_regressao(treino.loc[ids], stats.indice_m3s, FORMA_REGRESSAO)
            mu = float(prever_regressao(modelo, alvo).iloc[0])
            row['indice_previsto'] = mu
            if mu <= 0:
                raise ValueError('Índice previsto não positivo; Q(T) não calculado.')
            if distr is None:
                raise ImportError('lmoments3 não disponível.')
            reg = lmom_regional(stats)
            dist = {'GEV':distr.gev, 'PE3':distr.pe3}[FAMILIA_APLICACAO]
            pars = dist.lmom_fit(lmom_ratios=[1., reg['t'], reg['t3']])
            row['Q50_previsto'] = mu * float(curva_regional(pars, [50.], FAMILIA_APLICACAO)[0])
            try:
                h = heterogeneidade_h1(stats, N_SIM_H1_VALIDACAO, SEMENTE+30000+passo)
                row['H1_treino'] = h['H1']
                row['status'] = status_cluster(g, {g:h})
            except (ValueError, RuntimeError, ArithmeticError, ImportError) as exc:
                row['status'] = 'exploratório — H1 indisponível: ' + str(exc)
        except (ValueError, RuntimeError, ArithmeticError, ImportError) as exc:
            row['status'] = str(exc)
        linhas.append(row)
        if passo % 10 == 0 or passo == len(atributos):
            print(f'Validação index-flood: {passo}/{len(atributos)} postos.')
    return pd.DataFrame(linhas)


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

# %% 02 | MÁXIMAS DIÁRIAS ANUAIS E ÍNDICE — slide 6
anunciar('02', 'Máximas anuais', 'Qual máxima diária entra em cada ano aceito?')
maximas_locais, cobertura_maximas = maximas_anuais(
    vazoes_locais, MES_INICIO_ANO_HIDROLOGICO, 1-MAX_FALHAS_ANO_PCT/100)
maximas = maximas_locais.reindex(columns=atributos.index)
salvar_csv(maximas_locais, PASTA_TABELAS, '02_maximas.csv')
salvar_csv(cobertura_maximas, PASTA_TABELAS, '02_cobertura_max.csv')
if MAX_FALHAS_ANO_PCT > 0:
    print(f'ATENÇÃO: anos podem ter até {MAX_FALHAS_ANO_PCT:g}% de falhas. A cheia pode ter ocorrido em um dia ausente.')
if DADOS_LOCAIS_DISPONIVEIS:
    plotar_hidrograma(vazoes_locais[BACIA_EXEMPLO_EFETIVA], BACIA_EXEMPLO_EFETIVA)
    qmax = maximas_locais[BACIA_EXEMPLO_EFETIVA]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(qmax.index, qmax, 'o-', ms=4)
    ax.axhline(qmax.mean(), ls='--', label=f'Índice = {qmax.mean():.3f} m³/s')
    ax.set(title=f'{BACIA_EXEMPLO_EFETIVA} — máximas anuais das médias diárias',
           xlabel='Ano hidrológico', ylabel='Vazão (m³/s)')
    ax.legend(); ax.grid(alpha=.25); figura_aula(fig, '02_maximas')
print('O alvo é a máxima da vazão média diária em cada ano; não o pico instantâneo.')

# %% 03 | L-MOMENTOS LOCAIS E TABELA DO ÍNDICE — slides 7 e 16
anunciar('03', 'Estatísticas por posto')
estatisticas_locais = estatisticas_maximas(maximas_locais)
estatisticas = estatisticas_locais.reindex(atributos.index)
salvar_csv(estatisticas_locais, PASTA_TABELAS, '03_indices_lmom.csv')
print(estatisticas_locais.round(5).head().to_string())
print('Razões indefinidas são registradas. Não se excluem postos automaticamente para melhorar H1.')

# %% 04 | K-MEANS E WARD: CLUSTERS PRÓPRIOS PARA CHEIAS — slides 8–12
anunciar('04', 'Agrupamentos para index-flood')
if CLUSTERS_DISPONIVEIS:
    agrupamentos, grupos, regiao_alvo = executar_clusters(atributos, atributos_alvo)
else:
    print('04: agrupamento não executado; verifique o número de postos, K e os filtros.')

# %% 05 | NORMALIZAÇÃO E CURVAS POR CLUSTER — slides 13–15
anunciar('05', 'Normalização por índice local', 'As curvas se aproximam após remover a escala de cada estação?')
normalizadas = {}
if CLUSTERS_DISPONIVEIS:
    normalizadas = curvas_index_por_cluster(maximas, estatisticas, grupos)
    print('Cada estação foi dividida pela média de SUAS próprias máximas anuais; nunca pela média única do cluster.')
else:
    print('05: sem clusters disponíveis.')

# %% 06 | DISCORDÂNCIA E HOSKING–WALLIS POR CLUSTER — slides 17–18 e 35–38
anunciar('06', 'D_i, H1 e auditoria por cluster')
diagnosticos, tabela_h = {}, pd.DataFrame()
if CLUSTERS_DISPONIVEIS:
    diagnosticos, tabela_h = diagnosticos_por_cluster(estatisticas, agrupamentos)
else:
    print('06: sem clusters disponíveis.')

# %% 07 | GEV E PEARSON III POR CLUSTER — slides 19–21
anunciar('07', 'Distribuições regionais independentes')
distribuicoes, tabela_distribuicoes = {}, pd.DataFrame()
if CLUSTERS_DISPONIVEIS:
    distribuicoes, tabela_distribuicoes = ajustar_distribuicoes_por_cluster(estatisticas, grupos, diagnosticos)
else:
    print('07: sem clusters disponíveis.')

# %% 08 | REGRESSÕES DO ÍNDICE POR CLUSTER — slides 22–23
anunciar('08', 'Regionalização do índice local')
if CLUSTERS_DISPONIVEIS:
    modelos, tabela_regressoes = ajustar_modelos_por_cluster(atributos, estatisticas, grupos, ['indice_m3s'])
else:
    tabela_regressoes = pd.DataFrame()
    print('08: sem clusters disponíveis.')

# %% 09 | INDEX-FLOOD EM CADA CLUSTER — slide 24
anunciar('09', 'Aplicações independentes por cluster')
if CLUSTERS_DISPONIVEIS:
    tabela_QT = aplicar_index_todos_clusters(atributos, estatisticas, grupos, modelos, distribuicoes, diagnosticos)
else:
    tabela_QT = pd.DataFrame()
    print('09: sem clusters disponíveis.')

# %% 10 | BACIA RESERVADA: REGRESSÃO E CURVA DO MESMO CLUSTER
anunciar('10', 'Previsão da bacia sem posto')
mu_alvo = np.nan
previsao_alvo = pd.DataFrame()
if CLUSTERS_DISPONIVEIS and regiao_alvo is not None:
    modelo = modelos.get(regiao_alvo, {}).get('indice_m3s')
    if modelo is not None:
        mu_alvo = float(prever_regressao(modelo, atributos_alvo).iloc[0])
        linhas = []
        for familia in ('GEV', 'PE3'):
            pars = distribuicoes.get(regiao_alvo, {}).get(familia)
            if pars is None: continue
            crescimento = curva_regional(pars, TEMPOS_RETORNO, familia)
            for T, fator in zip(TEMPOS_RETORNO, crescimento):
                linhas.append({'id_bacia': CODIGO_ALVO, 'regiao': regiao_alvo, 'familia': familia,
                               'T_anos': T, 'indice_estimado': mu_alvo, 'gR': fator,
                               'Q_m3s': mu_alvo*fator if mu_alvo > 0 else np.nan,
                               'uso': status_cluster(regiao_alvo, diagnosticos)})
        previsao_alvo = pd.DataFrame(linhas)
        if not previsao_alvo.empty:
            pasta = pasta_cluster(regiao_alvo)
            salvar_csv(previsao_alvo, pasta / 'tab', 'alvo_Q_T.csv', index=False)
            print(previsao_alvo.round(4).to_string(index=False))
            fig, ax = plt.subplots(figsize=(8, 5))
            for familia, parte in previsao_alvo.groupby('familia'):
                ax.plot(parte.T_anos, parte.Q_m3s, 'o-', label=familia)
            ax.set_xscale('log')
            ax.set(title=f'{CODIGO_ALVO} — somente C{regiao_alvo:02d}; curvas candidatas',
                   xlabel='T (anos)', ylabel='Máxima diária anual Q(T) (m³/s)')
            ax.legend(); ax.grid(alpha=.25); figura_aula(fig, 'alvo_Q_T', pasta / 'fig')
        if previsao_alvo.empty:
            print('10: curva de crescimento indisponível; consulte o relatório de distribuições. Nenhum Q(T) foi calculado.')
        if mu_alvo <= 0:
            print('Índice estimado não positivo: Q(T) não calculado; revisar a regressão.')
    else:
        print('10: modelo indisponível no cluster da bacia reservada.')
else:
    print('10: sem bacia reservada ou sem clusters.')

# %% 11 | VALIDAÇÃO ESPACIAL DO INDEX-FLOOD — slides 28–29
anunciar('11', 'Uma bacia fora por vez')
if CLUSTERS_DISPONIVEIS and EXECUTAR_VALIDACAO_ESPACIAL:
    validacao = validar_index(atributos, estatisticas)
    salvar_csv(validacao, PASTA_TABELAS, '11_validacao.csv', index=False)
    if not validacao.empty:
        metrica = metricas_regressao(validacao.indice_observado, validacao.indice_previsto)
        salvar_csv(pd.DataFrame([metrica]), PASTA_TABELAS, '11_metricas_indice.csv', index=False)
        grafico_observado_previsto(validacao.indice_observado, validacao.indice_previsto,
                                  'Índice — validação espacial', '11_validacao_indice')
        print(metrica)
    print('Clusters, padronização, regressão e curva são refeitos sem o posto retido.')
    print('Q50_previsto não é comparado com uma Q50 diretamente observada; um quantil local também seria estimado.')
else:
    print('11: validação desativada ou sem clusters suficientes.')

# %% 12 | REVELAR A BACIA RESERVADA E RESUMIR
anunciar('12', 'Avaliação final do índice')
if vazao_alvo_oculta is not None and np.isfinite(mu_alvo):
    max_alvo, _ = maximas_anuais(vazao_alvo_oculta.to_frame(), MES_INICIO_ANO_HIDROLOGICO,
                                1-MAX_FALHAS_ANO_PCT/100)
    observado = float(max_alvo.iloc[:, 0].mean())
    avaliacao_alvo = pd.DataFrame([{'id_bacia': CODIGO_ALVO, 'regiao': regiao_alvo,
                                    'indice_observado': observado, 'indice_previsto': mu_alvo,
                                    'erro_relativo_pct': 100*(mu_alvo/observado-1) if observado else np.nan}])
    salvar_csv(avaliacao_alvo, pasta_cluster(regiao_alvo) / 'tab', 'alvo_revelado.csv', index=False)
    print(avaliacao_alvo.round(3).to_string(index=False))
finalizar({'n_postos_inventario': len(base['selecao']), 'n_treino': len(atributos),
           'clusters_executados': CLUSTERS_DISPONIVEIS, 'bacia_alvo': CODIGO_ALVO,
           'tempos_retorno': TEMPOS_RETORNO.tolist(), 'n_validacao': len(validacao),
           'H1_calculados': len(diagnosticos),
           'Z_DIST_implementado': False})
