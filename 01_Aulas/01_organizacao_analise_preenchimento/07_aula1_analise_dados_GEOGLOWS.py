# -*- coding: utf-8 -*-
"""Aula 1: mapa, qualidade, disponibilidade e correlação.
Executar por células no Spyder. Diários e horários são analisados separadamente.
Os caminhos abaixo são explícitos; usar os dados previamente preparados.
"""
# %% 1. Importar as bibliotecas
# Explicar: pandas = tabelas; GeoPandas = geodados; Matplotlib = gráficos.
from pathlib import Path
from datetime import datetime
import re
import unicodedata
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# %% 2. Configurar caminhos e parâmetros
# Ajustar uma vez antes da aula. Manter as resoluções diária e horária separadas.
# Mover/copiar o CONTEÚDO da execução do script 06 para Dados, mantendo
# catalogo_dados_finais.csv, coordenadas_estacoes.csv, diarios/ e horarios/.
# Colocar os arquivos geoespaciais em SIG. Nenhum arquivo é movido pelo script.
DIRETORIO_AULA = Path(r'C:\Users\afrod\OneDrive\Documents\HydroUAI_git\hydrouai-epagri-2026\02_Dados\Dia 1')
BACIA = 'bacia_02'  # Rótulo dos gráficos; usar os arquivos da bacia correspondente.
DIRETORIO_DADOS = DIRETORIO_AULA / 'Dados'
DIRETORIO_SIG = DIRETORIO_AULA / 'SIG'
DIRETORIO_RESULTADOS = DIRETORIO_AULA / 'Resultados'
DIRETORIO_CSVS_FINAIS = DIRETORIO_AULA / 'Dados_Preenchidos'  # Alterar para a pasta desejada.
# Os dois CSVs finais são atualizados nessa pasta ao executar novamente.
ARQUIVO_CATALOGO = DIRETORIO_DADOS / 'catalogo_dados_finais.csv'
ARQUIVO_COORDENADAS = DIRETORIO_DADOS / 'coordenadas_estacoes.csv'
ARQUIVO_BACIA = DIRETORIO_SIG / 'bacia.gpkg'  # Trocar por 'bacia.gpkg' se esse for o nome.
CAMADA_BACIA = None
CAMPO_BACIA = None  # Se houver várias bacias: coluna que identifica cada uma.
VALOR_BACIA = BACIA  # Valor dessa coluna correspondente à bacia escolhida.
ARQUIVO_ESTACOES = DIRETORIO_SIG / 'estacoes_internas.gpkg'
CAMADA_ESTACOES = None
CAMPO_CODIGO_ESTACOES = 'codigo'
ARQUIVO_DRENAGEM = None  # Opcional: DIRETORIO_SIG / 'HIDROGRAFIA.gpkg'
CAMADA_DRENAGEM = None
CRS_MAPA = 31982  # SIRGAS 2000 / UTM 22S — Santa Catarina.
PERIODO_ANALISE = None  # Ex.: ('2013-01-01', '2023-12-31'); None: cada série inteira.
MIN_PARES_CORRELACAO = 30
FATOR_IQR = 1.5
SENTINELAS = [-9999, -999, -99999]
SALVAR_FIGURAS = True
EXIBIR_FIGURAS = True
CHUVA_GRAFICO_DIARIO = None  # Ex.: ('2463', 'chuva') ou serie_id.
CHUVA_GRAFICO_HORARIO = None  # None: primeira chuva entre as entradas finais.
# Preenchimento: janelas em passos da resolução (dias ou horas).
JANELA_MEDIA_MOVEL = 7
MIN_OBS_MEDIA_MOVEL = 2
MAX_FALHA_MEDIA_MOVEL = 3
ID_REFERENCIA_REGRESSAO = None  # Maior correlação no treino, por alvo.
USAR_PROPRIA_VAZAO = True  # Regressão/KNN com defasagens da própria vazão.
DEFASAGENS_PROPRIA_SERIE = [1, 2, 3]  # Passos: dias ou horas conforme a série.
# Falhas maiores que o limite do modelo seguem para a média climatológica.
MAX_FALHA_AUTORREFERENCIA = 3  # Não reconstruir blocos maiores.
K_VIZINHOS = 5
MAX_REFERENCIAS_KNN = 5
MAX_AMOSTRAS_KNN = 20000
MIN_PARES_TREINO = 30
# Validação de preenchimento: blocos distribuídos no período inteiro, sem teste ML.
TAMANHO_BLOCO_VALIDACAO = 3
MAX_BLOCOS_VALIDACAO = 20
NAO_NEGATIVO_P_Q = True
# Produtos diários locais, por serie_id. Nunca desagregar em horas.
# Ex.: PRODUTOS_LOCAIS = {'ID_CHUVA': {'CHIRPS': {
# 'arquivo': DIRETORIO_PRODUTOS / 'CHIRPS.csv', 'data': 'data_hora', 'valor': 'valor',
# 'sep': ';', 'campo_codigo': 'codigo_estacao', 'codigo': 'CODIGO'}}}
# GEOGLOWS: mesma configuração para vazão; conferir trecho e unidade m³/s.
PRODUTOS_LOCAIS = {}  # Por serie_id, quando precisar distinguir fontes.
# Alternativa simples: configurar produtos por código da estação.
# Arquivos já baixados; não ocorre download neste script.
CHIRPS_POR_ESTACAO = {}
GEOGLOWS_POR_ESTACAO = {
    '2463': {
        'arquivo': DIRETORIO_DADOS / 'GEOGLOWS_Estacao_1_640447698_daily.csv',
        'data': 'Data',
        'valor': 'Vazao_GEOGLOWS',
        'sep': ';',
        'campo_codigo': None,
        'codigo': None
    }
}
# GEOGLOWS entra somente nas séries DIÁRIAS de vazão.
# A correção adotada é multiplicativa pela razão das médias no período de ajuste:
# Q_GEO_corr = Q_GEO * (media(Q_obs) / media(Q_GEO)).
# Na validação, o fator é calculado sem os blocos ocultados; no preenchimento final,
# ele é recalculado com todas as observações originais disponíveis.

# Exemplo (retirar # e ajustar caminho/código quando tiver os arquivos):
# CHIRPS_POR_ESTACAO = {'2463': {
#     'arquivo': DIRETORIO_PRODUTOS / 'CHIRPS_diario_2463.csv',
#     'data': 'data_hora', 'valor': 'valor', 'sep': ';',
#     'campo_codigo': 'codigo_estacao', 'codigo': '2463'}}
# GEOGLOWS_POR_ESTACAO = {'2463': {
#     'arquivo': DIRETORIO_PRODUTOS / 'GEOGLOWS_diario_2463.csv',
#     'data': 'data_hora', 'valor': 'valor', 'sep': ';',
#     'campo_codigo': None, 'codigo': None}}
# CHIRPS em mm/dia; GEOGloWS em m³/s, do trecho da estação alvo.
# Ajustar coluna/separador aos seus dados. Produto ausente: deixar {}.
# Se usar chuva média da bacia: campo_codigo=None; não confundir com chuva pontual.

plt.rcParams.update({'figure.dpi': 110, 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .2})


# %% 3. Preparar as funções de apoio — executar uma vez
# Apoio técnico: não é necessário explicar toda a implementação aos alunos.
# Esta célula define funções; as análises estão nas células seguintes.
def normalizar(x):
    return re.sub('[^a-z0-9]', '', unicodedata.normalize('NFKD', str(x)).encode('ascii', 'ignore').decode().lower())


def tabela(path):
    return pd.read_csv(path, sep=';', dtype={'codigo_estacao': str, 'codigo': str, 'serie_id': str})


def salvar_tabela(rows, name):
    pd.DataFrame(rows).to_csv(RESULTADOS / name, sep=';', index=False, encoding='utf-8-sig')


def mostrar(fig, name):
    fig.tight_layout()
    if SALVAR_FIGURAS:
        fig.savefig(RESULTADOS / (name + '.png'), dpi=200, bbox_inches='tight')
    if EXIBIR_FIGURAS:
        plt.show()
    else:
        plt.close(fig)


def conferir_arquivos_catalogo(catalogo, diretorio):
    """Exige caminhos relativos à nova pasta, sem consultar pastas anteriores."""
    from pathlib import PureWindowsPath
    if 'arquivo_final' not in catalogo:
        raise ValueError('Catálogo sem arquivo_final: utilizar o catálogo gerado pelo script 06.')
    problemas = []
    raiz = Path(diretorio).resolve()
    for valor in catalogo.arquivo_final:
        if pd.isna(valor) or not str(valor).strip():
            problemas.append('Caminho vazio no catálogo.')
            continue
        texto = str(valor).strip()
        relativo = Path(texto.replace('\\', '/'))
        win = PureWindowsPath(texto)
        if relativo.is_absolute() or win.drive or win.root or '..' in relativo.parts:
            problemas.append(f'Caminho não relativo à pasta da aula: {texto}')
            continue
        destino = raiz / relativo
        if not destino.is_file():
            problemas.append(f'Arquivo ausente: {destino}')
    if problemas:
        raise FileNotFoundError('Dados incompletos na pasta indicada. Levar o conteúdo da execução do script 06, preservando diarios/ e horarios/.\n'
                                + '\n'.join(problemas[:10]))
    # Padronizar separadores para usar o catálogo movido em Windows ou Linux.
    resultado = catalogo.copy()
    resultado['arquivo_final'] = resultado.arquivo_final.astype(str).str.replace('\\', '/', regex=False)
    return resultado


def carregar_series(catalog, root, periodo=None):
    records, summaries = {}, []
    for _, r in catalog.iterrows():
        sid = str(r.serie_id)
        if sid in records:
            raise ValueError(f'ID de série repetido no catálogo: {sid}')
        df = tabela(root / r.arquivo_final)
        raw_value = df.valor.astype('string').str.strip()
        brazilian = raw_value.str.contains(',', na=False)
        raw_value.loc[brazilian] = raw_value.loc[brazilian].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
        values = pd.to_numeric(raw_value, errors='coerce')
        times = pd.to_datetime(df.data_hora, format='mixed', errors='coerce')
        invalid_dates = int(times.isna().sum())
        sentinels = int(values.isin(SENTINELAS).sum())
        values = values.replace(SENTINELAS, np.nan)
        clean = pd.DataFrame({'data_hora': times, 'valor': values})
        clean = clean[clean.data_hora.notna()].sort_values('data_hora')
        if periodo:
            a, b = (pd.Timestamp(x).normalize() for x in periodo)
            clean = clean[clean.data_hora.ge(a) & clean.data_hora.lt(b + pd.Timedelta('1d'))]
        if clean.empty:
            summaries.append({'serie_id': sid, 'codigo_estacao': r.codigo_estacao, 'grupo': r.grupo,
                              'tipo': r.tipo, 'status': 'sem_registros_no_periodo', 'datas_invalidas': invalid_dates})
            records[sid] = {'catalogo': r.to_dict(), 'dados': clean, 'serie': pd.Series(dtype=float), 'dias': pd.DatetimeIndex([])}
            continue
        duplicates = int(clean.data_hora.duplicated(False).sum())
        # Linhas repetidas não são somadas/médiadas. Valores divergentes para
        # o mesmo timestamp são excluídos SOMENTE da visão analítica e sinalizados.
        unique_values = clean.groupby('data_hora').valor.nunique(dropna=True)
        conflicts = unique_values[unique_values > 1].index
        one = clean.groupby('data_hora').valor.first().sort_index()
        one.loc[conflicts] = np.nan
        first, last = clean.data_hora.min(), clean.data_hora.max()
        freq = 'D' if r.grupo == 'diarios' else 'h'
        step = pd.Timedelta('1d' if r.grupo == 'diarios' else '1h')
        unique_times = clean.data_hora.drop_duplicates().sort_values()
        grid = pd.date_range(first, last, freq=freq)
        off_grid = int((~pd.DatetimeIndex(unique_times).isin(grid)).sum())
        regularity = float((unique_times.diff().dropna() == step).mean()) if len(unique_times) > 1 else np.nan
        if off_grid:
            # Não arredondar/reindexar séries irregulares silenciosamente.
            analytical = one
        else:
            analytical = one.reindex(grid)
        flags = analytical < 0
        amostra_iqr = analytical[analytical > 0] if r.tipo == 'chuva' else analytical.dropna()
        q1, q3 = amostra_iqr.quantile([.25, .75])
        iqr = q3 - q1
        lower, upper = q1 - FATOR_IQR * iqr, q3 + FATOR_IQR * iqr
        iqr_flags = (analytical < lower) | (analytical > upper)
        if r.tipo == 'chuva':
            iqr_flags &= analytical > 0  # Zero é ausência de chuva; fica na série.
        flag_df = pd.DataFrame({'data_hora': analytical.index, 'valor': analytical.to_numpy(),
                               'negativo': flags.to_numpy(), 'sinalizado_IQR': iqr_flags.to_numpy()})
        flag_df['serie_id'] = sid
        flag_df['codigo_estacao'] = r.codigo_estacao
        valid_days = analytical.dropna().index.normalize().unique().sort_values()
        period_start = pd.Timestamp(periodo[0]).normalize() if periodo else first.normalize()
        period_end = pd.Timestamp(periodo[1]).normalize() if periodo else last.normalize()
        days = pd.date_range(period_start, period_end, freq='D')
        missing = days.difference(valid_days)
        if len(missing):
            breaks = np.r_[True, np.diff(missing.asi8) != pd.Timedelta('1d').value]
            gaps = [(v.min(), v.max(), len(v)) for _, v in pd.Series(missing).groupby(np.cumsum(breaks))]
        else:
            gaps = []
        longest = max((g[2] for g in gaps), default=0)
        missing_slots = max(0, len(grid) - analytical.dropna().index.intersection(grid).size)
        row = {'serie_id': sid, 'codigo_estacao': r.codigo_estacao, 'grupo': r.grupo, 'tipo': r.tipo,
               'fonte': r.fonte, 'status': 'irregular' if off_grid else 'grade_regular',
               'inicio': str(first), 'fim': str(last), 'registros': len(clean), 'valores_validos_analiticos': int(analytical.notna().sum()),
               'datas_invalidas': invalid_dates, 'sentinelas_na_origem': sentinels,
               'linhas_com_timestamp_repetido': duplicates, 'timestamps_conflitantes': len(conflicts),
               'timestamps_fora_grade': off_grid, 'intervalos_no_passo_esperado_pct': 100 * regularity,
               'dias_no_periodo_avaliado': len(days), 'dias_com_dados': len(valid_days),
               'disponibilidade_diaria_pct': 100 * len(valid_days) / len(days),
               'maior_falha_dias': longest, 'passos_sem_dados_grade': missing_slots if not off_grid else np.nan,
               'negativos': int(flags.sum()), 'sinalizados_IQR': int(iqr_flags.sum()),
               'IQR': iqr, 'limite_IQR_inferior': lower, 'limite_IQR_superior': upper}
        if r.grupo == 'horarios':
            hours = analytical.dropna().index.floor('h').unique()
            counts = pd.Series(1, index=hours).groupby(hours.normalize()).sum()
            row['dias_com_24_horas'] = int((counts == 24).sum())
        records[sid] = {'catalogo': r.to_dict(), 'dados': clean, 'serie': analytical,
                        'dias': valid_days, 'flags': flag_df, 'falhas': gaps, 'irregular': bool(off_grid),
                        'resumo': row}
        summaries.append(row)
    return records, pd.DataFrame(summaries)


def grafico_gantt(records, tipo):
    # Agrupar dias válidos por código e resolução. União entre fontes do mesmo
    # código/resolução significa que havia algum dado; não fusão dos valores.
    grupos = {}
    for obj in records.values():
        r = obj['catalogo']
        if r['tipo'] != tipo or not len(obj['dias']): continue
        key = (r['codigo_estacao'], r['grupo'])
        grupos.setdefault(key, []).append(obj['dias'])
    if not grupos:
        print('Sem séries para Gantt:', tipo)
        return
    fig, ax = plt.subplots(figsize=(12, max(3, .5 * len(grupos) + 1.5)))
    labels = []
    for y, (key, listas) in enumerate(sorted(grupos.items())):
        dias = pd.DatetimeIndex(np.concatenate([idx.to_numpy() for idx in listas])).unique().sort_values()
        numeros = mdates.date2num(dias.to_pydatetime())
        cortes = np.flatnonzero(np.diff(numeros) > 1) + 1
        inicios = np.r_[0, cortes]
        finais = np.r_[cortes - 1, len(numeros) - 1]
        barras = np.column_stack([numeros[inicios], numeros[finais] - numeros[inicios] + 1])
        # UMA coleção gráfica por linha, em vez de uma chamada por intervalo.
        ax.broken_barh(barras, (y - .3, .6), facecolors='#287c94' if tipo == 'vazao' else '#168a65')
        labels.append(f"{key[0]}_{'Q' if tipo == 'vazao' else 'P'} | {key[1]}")
    ax.set_yticks(range(len(labels)), labels)
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.invert_yaxis()
    ax.set_title(('Q — vazão' if tipo == 'vazao' else 'P — precipitação') + ': disponibilidade em dias')
    ax.set_xlabel('Data — barras representam dias com pelo menos um dado válido')
    mostrar(fig, '03_gantt_' + ('Q' if tipo == 'vazao' else 'P'))


def criar_rotulos(catalogo):
    """Nomes curtos desde a leitura; distinguir fontes apenas quando necessário."""
    sufixos = {'chuva': 'P', 'vazao': 'Q', 'nivel': 'H'}
    nomes, vistos = {}, set()
    bases = [(str(r.serie_id), str(r.grupo),
              f"{r.codigo_estacao}_{sufixos.get(r.tipo, str(r.tipo))}", r)
             for _, r in catalogo.iterrows()]
    contagens = pd.Series([(grupo, base) for _, grupo, base, _ in bases], dtype=object).value_counts()
    for sid, grupo, base, r in bases:
        nome = base
        if contagens[(grupo, base)] > 1:
            derivada = str(r.get('origem_resolucao', '')) == 'horaria_agregada' or sid.endswith('_diario_derivado')
            nome += '_' + str(r.fonte) + ('_derivada' if derivada else '')
        candidato, numero = nome, 2
        while (grupo, candidato) in vistos:
            candidato = f'{nome}_{numero}'
            numero += 1
        vistos.add((grupo, candidato))
        nomes[sid] = candidato
    return nomes


def rotulo_serie(sid):
    texto = str(sid)
    base, sep, janela = texto.partition(' | P_acum_')
    nome = ROTULOS_SERIES.get(base, base)
    if sep:
        grupo = catalogo.loc[catalogo.serie_id.astype(str).eq(base), 'grupo']
        unidade = 'd' if len(grupo) and grupo.iloc[0] == 'diarios' else 'h'
        nome += f'_acum_{janela}{unidade}'
    return nome


def boxplot_iqr(records, tipo):
    amostras, labels = [], []
    for obj in records.values():
        r = obj['catalogo']
        if r['tipo'] != tipo: continue
        x = obj['serie'].dropna()
        if tipo == 'chuva': x = x[x > 0]
        if x.empty: continue
        amostras.append(x.to_numpy())
        labels.append(rotulo_serie(str(r['serie_id'])))
    if not amostras:
        print('Sem valores para boxplot:', tipo)
        return
    fig, ax = plt.subplots(figsize=(max(8, len(amostras) * .8), 5))
    ax.boxplot(amostras, whis=FATOR_IQR, patch_artist=True,
               boxprops={'facecolor': '#c7e3df' if tipo == 'chuva' else '#c9deed'},
               flierprops={'markersize': 3, 'alpha': .4})
    ax.set_xticks(np.arange(1, len(labels) + 1), labels, rotation=35, ha='right')
    ax.set_ylabel('Precipitação (mm), apenas P > 0' if tipo == 'chuva' else 'Vazão (unidade da fonte)')
    ax.set_title('Boxplot e IQR — ' + ('chuva sem zeros' if tipo == 'chuva' else 'vazão'))
    mostrar(fig, '02_boxplot_' + ('P_positiva' if tipo == 'chuva' else 'Q'))


def histogramas_series(records, bins='auto', chuva_sem_zeros=True):
    """Frequência relativa das observações; painéis separados por variável/resolução."""
    for grupo in ['diarios', 'horarios']:
        for tipo in ['vazao', 'chuva']:
            for sid, obj in records.items():
                r = obj['catalogo']
                if r['grupo'] != grupo or r['tipo'] != tipo:
                    continue
                x = pd.to_numeric(obj['serie'], errors='coerce').dropna()
                x = x[np.isfinite(x)]
                zeros = int(x.eq(0).sum())
                total = len(x)
                if tipo == 'chuva' and chuva_sem_zeros:
                    x = x[x > 0]
                if x.empty:
                    print(rotulo_serie(sid), grupo, '— sem valores para histograma.')
                    continue
                fig, ax = plt.subplots(figsize=(8, 4.5))
                bordas = np.histogram_bin_edges(x.to_numpy(), bins=bins)
                ax.hist(x.to_numpy(), bins=bordas, weights=np.full(len(x), 100 / len(x)),
                        color='#287c94' if tipo == 'vazao' else '#168a65',
                        edgecolor='white', linewidth=.6)
                ax.set_xlabel('Vazão (unidade da fonte)' if tipo == 'vazao' else 'Precipitação (mm)')
                ax.set_ylabel('Frequência relativa (%)')
                nota = f'n = {len(x)}'
                if tipo == 'chuva' and chuva_sem_zeros:
                    nota += f'; apenas P > 0; zeros: {100 * zeros / total:.1f}% dos registros válidos'
                ax.set_title(f'{rotulo_serie(sid)} | {grupo} — histograma\n{nota}')
                mostrar(fig, f'02_histograma_{grupo}_{sid}')


def curva_permanencia_diaria(records, percentuais=(5, 10, 90, 98)):
    """Qp é a vazão igualada ou excedida em p% dos dias válidos observados.

    Ordenação decrescente; posição empírica m/(n+1) (Weibull).
    Qp calculada por interpolação linear na mesma curva; fora das posições
    empíricas disponíveis, Qp não é estimada. Zeros são mantidos.
    """
    resumo = []
    for sid, obj in records.items():
        r = obj['catalogo']
        if r['grupo'] != 'diarios' or r['tipo'] != 'vazao':
            continue
        x = pd.to_numeric(obj['serie'], errors='coerce').dropna()
        x = x[np.isfinite(x) & (x >= 0)]
        if len(x) < 2:
            print(rotulo_serie(sid), '— observações insuficientes para permanência.')
            continue
        q = np.sort(x.to_numpy(dtype=float))[::-1]
        permanencia = 100 * np.arange(1, len(q) + 1) / (len(q) + 1)
        fig, ax = plt.subplots(figsize=(8.5, 5))
        ax.plot(permanencia, q, color='#287c94', lw=1.8, label=rotulo_serie(sid))
        registro = {'serie_id': sid, 'estacao': rotulo_serie(sid), 'n_dias_validos': len(q),
                    'inicio': str(x.index.min()), 'fim': str(x.index.max()),
                    'posicao_empirica': '100*m/(n+1)'}
        for percentual, cor in zip(percentuais, ['#d55e00', '#e69f00', '#009e73', '#cc79a7']):
            if not 0 < percentual < 100:
                raise ValueError('Percentuais de permanência devem estar entre 0 e 100.')
            valor = (float(np.interp(percentual, permanencia, q))
                     if permanencia[0] <= percentual <= permanencia[-1] else np.nan)
            registro[f'Q{percentual}'] = valor
            if not np.isfinite(valor):
                print(rotulo_serie(sid), f'Q{percentual}: amostra insuficiente para esta posição.')
                continue
            ax.plot([percentual, percentual], [0, valor], ls='--', color=cor, alpha=.7)
            ax.plot([0, percentual], [valor, valor], ls=':', color=cor, alpha=.7)
            ax.scatter([percentual], [valor], color=cor, zorder=4,
                       label=f'Q{percentual} = {valor:.3g}')
        ax.set_xlim(0, 100)
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Permanência — probabilidade de excedência (%)')
        ax.set_ylabel('Vazão (unidade da fonte)')
        ax.set_title(f'{rotulo_serie(sid)} — curva de permanência diária\n{len(q)} dias válidos; série observada')
        ax.legend(loc='best')
        mostrar(fig, f'02_permanencia_diaria_{sid}')
        resumo.append(registro)
    resultado = pd.DataFrame(resumo)
    salvar_tabela(resultado, '02_vazoes_permanencia_diarias.csv')
    if not resultado.empty:
        print(resultado.drop(columns=['serie_id', 'posicao_empirica']).to_string(index=False))
    return resultado


def preparar_modelo(entradas, resposta, grupo, janela):
    """Escolha por (código, variável); em caso de múltiplas fontes, usar serie_id."""
    if not isinstance(janela, int) or isinstance(janela, bool) or janela < 1:
        raise ValueError('O acumulado deve ser um inteiro positivo de dias ou horas.')
    if not entradas or resposta is None:
        print(f'{grupo}: preencha ENTRADAS e RESPOSTA na célula de seleção.')
        return None
    def obter(escolha):
        if isinstance(escolha, str):
            ids = [escolha] if escolha in series else []
        else:
            codigo, tipo = escolha
            ids = [sid for sid, obj in series.items()
                   if str(obj['catalogo']['codigo_estacao']) == str(codigo)
                   and obj['catalogo']['tipo'] == tipo and obj['catalogo']['grupo'] == grupo]
        if len(ids) != 1:
            raise ValueError(f'{escolha}: encontrados {len(ids)} IDs {ids}. Escolha um serie_id da lista.')
        obj = series[ids[0]]
        if obj['catalogo']['grupo'] != grupo:
            raise ValueError(f'{escolha}: resolução diferente de {grupo}.')
        if obj.get('irregular') or obj['serie'].empty:
            raise ValueError(f'{escolha}: série vazia ou irregular; conferir diagnóstico.')
        return ids[0], obj
    alvo_id, alvo = obter(resposta)
    escolhidas = [obter(x) for x in entradas]
    ids = [sid for sid, _ in escolhidas]
    if alvo_id in ids or len(set(ids)) != len(ids):
        raise ValueError('Entradas repetidas ou iguais à resposta; revisar seleção.')
    freq = 'D' if grupo == 'diarios' else 'h'
    passo = pd.Timedelta('1d' if grupo == 'diarios' else '1h').value
    todas = [alvo] + [obj for _, obj in escolhidas]
    if len({obj['serie'].index[0].value % passo for obj in todas}) != 1:
        raise ValueError('Horários de referência diferentes; conferir alinhamento dos dados.')
    inicio = min(obj['serie'].index.min() for obj in todas)
    fim = max(obj['serie'].index.max() for obj in todas)
    grade = pd.date_range(inicio, fim, freq=freq)
    dados = pd.DataFrame(index=grade)
    dados.index.name = 'data_hora'
    chuvas = []
    for sid, obj in escolhidas:
        dados[sid] = obj['serie'].reindex(grade)
        if obj['catalogo']['tipo'] == 'chuva':
            nome = f'{sid} | P_acum_{janela}'
            # Soma móvel retrospectiva: P(t) + ... + P(t-janela+1).
            # Uma falha na janela torna o acumulado ausente; zero é preservado.
            dados[nome] = dados[sid].rolling(janela, min_periods=janela).sum()
            chuvas.append(nome)
    dados['RESPOSTA'] = alvo['serie'].reindex(grade)
    print('Resposta:', rotulo_serie(alvo_id), '| acumulado:', janela, 'dias' if freq == 'D' else 'horas')
    nomes_colunas = {col: rotulo_serie(col) for col in dados if col != 'RESPOSTA'}
    nomes_colunas['RESPOSTA'] = rotulo_serie(alvo_id)
    dados.rename(columns=nomes_colunas).to_csv(RESULTADOS / f'{grupo}_dados_selecionados.csv',
                                             sep=';', encoding='utf-8-sig')
    return {'dados': dados, 'chuvas': chuvas, 'resposta_id': alvo_id, 'grupo': grupo}


def analisar_correlacao(estudo, defasagens):
    if estudo is None:
        return
    if not defasagens or any(not isinstance(x, int) or x < 0 for x in defasagens):
        raise ValueError('Defasagens devem ser uma lista de inteiros não negativos.')
    dados, grupo = estudo['dados'], estudo['grupo']
    linhas = []
    for coluna in dados.columns.drop('RESPOSTA'):
        for lag in defasagens:
            # lag positivo: entrada no passado versus resposta em t.
            pares = pd.concat([dados[coluna].shift(lag).rename('x'),
                               dados.RESPOSTA.rename('y')], axis=1).dropna()
            r = pares.x.corr(pares.y) if (len(pares) >= MIN_PARES_CORRELACAO
                 and pares.x.nunique() > 1 and pares.y.nunique() > 1) else np.nan
            linhas.append({'entrada': rotulo_serie(coluna), 'entrada_id': coluna, 'defasagem': lag, 'r_Pearson': r,
                           'pares_validos': len(pares)})
    resultados = pd.DataFrame(linhas)
    salvar_tabela(resultados, f'{grupo}_correlacao_resposta.csv')
    print(resultados.drop(columns='entrada_id').to_string(index=False))
    # Correlação simultânea entre todas as entradas, acumulados e resposta.
    matriz = dados.corr(min_periods=MIN_PARES_CORRELACAO)
    rotulos = {col: rotulo_serie(col) for col in matriz.columns if col != 'RESPOSTA'}
    rotulos['RESPOSTA'] = 'Resposta: ' + rotulo_serie(estudo['resposta_id'])
    matriz = matriz.rename(index=rotulos, columns=rotulos)
    matriz.to_csv(RESULTADOS / f'{grupo}_matriz_correlacao.csv', sep=';', encoding='utf-8-sig')
    fig, ax = plt.subplots(figsize=(max(7, len(matriz) * .7), max(5, len(matriz) * .6)))
    im = ax.imshow(matriz, vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set_xticks(range(len(matriz)), matriz.columns, rotation=60, ha='right', fontsize=8)
    ax.set_yticks(range(len(matriz)), matriz.index, fontsize=8)
    if len(matriz) <= 12:
        for i in range(len(matriz)):
            for j in range(len(matriz)):
                valor = matriz.iloc[i, j]
                ax.text(j, i, f'{valor:.2f}' if pd.notna(valor) else '—', ha='center', va='center', fontsize=8)
    ax.grid(False)
    ax.set_title(f'{grupo} — correlação de Pearson (defasagem zero)')
    fig.colorbar(im, ax=ax, label='r de Pearson')
    mostrar(fig, f'{grupo}_matriz_correlacao')
    fig, ax = plt.subplots(figsize=(9, 5))
    for nome, tabela_r in resultados.groupby('entrada_id', sort=False):
        ax.plot(tabela_r.defasagem, tabela_r.r_Pearson, marker='o', markersize=3, label=rotulo_serie(nome))
    ax.axhline(0, color='gray', linewidth=.7)
    ax.set_ylim(-1, 1)
    ax.set_xlabel('Defasagem (dias)' if grupo == 'diarios' else 'Defasagem (horas)')
    ax.set_ylabel('r de Pearson com a resposta')
    ax.set_title(f'{grupo} — entradas e precipitação acumulada versus resposta')
    ax.legend(fontsize=7, loc='best')
    mostrar(fig, f'{grupo}_correlacao_defasada')
    return resultados


def carregar_produto_local(config, index):
    path = config.get('arquivo')
    if path is None: return None
    df = pd.read_csv(Path(path), sep=config.get('sep', ';'), dtype=str)
    cc = config.get('campo_codigo')
    if cc and cc in df:
        codes = df[cc].dropna().astype(str).unique()
        selected = config.get('codigo')
        if selected is None and len(codes) > 1:
            raise ValueError(f'{path}: vários códigos. Preencher codigo na configuração do produto.')
        if selected is not None:
            df = df[df[cc].astype(str).eq(str(selected))]
    date_col, value_col = config['data'], config['valor']
    if date_col not in df or value_col not in df:
        raise ValueError(f'{path}: conferir as colunas data e valor configuradas.')
    times = pd.to_datetime(df[date_col], format='mixed', dayfirst=True, errors='coerce')
    text = df[value_col].astype('string').str.strip()
    mask_br = text.str.contains(',', na=False)
    text.loc[mask_br] = text.loc[mask_br].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    values = pd.to_numeric(text, errors='coerce').replace(SENTINELAS, np.nan).replace([np.inf, -np.inf], np.nan)
    if times.isna().any(): raise ValueError(f'{path}: existem datas inválidas; revisar antes de usar o produto.')
    if not times.eq(times.dt.normalize()).all():
        raise ValueError(f'{path}: preparar uma tabela DIÁRIA, com uma data por registro.')
    sample = pd.DataFrame({'data': times, 'valor': values})
    if sample.groupby('data').valor.nunique().gt(1).any():
        raise ValueError(f'{path}: valores conflitantes para a mesma data.')
    external = sample.groupby('data').valor.first().sort_index()
    return external.reindex(index)


def media_movel_falhas(context):
    # Causal: usa somente observações anteriores e o passo atual.
    # Não usa previsões recursivas e não cria zeros artificiais.
    if JANELA_MEDIA_MOVEL % 2 != 1 or JANELA_MEDIA_MOVEL < 3:
        raise ValueError('JANELA_MEDIA_MOVEL deve ser ímpar e >=3.')
    if not 1 <= MIN_OBS_MEDIA_MOVEL <= JANELA_MEDIA_MOVEL:
        raise ValueError('MIN_OBS_MEDIA_MOVEL deve estar entre 1 e a janela.')
    missing = context.isna()
    groups = missing.ne(missing.shift()).cumsum()
    lengths = missing.groupby(groups).transform('size')
    eligible = missing & lengths.le(MAX_FALHA_MEDIA_MOVEL)
    return context.rolling(JANELA_MEDIA_MOVEL, center=False, min_periods=MIN_OBS_MEDIA_MOVEL).mean().where(eligible)


def ajustar_regressao(y_fit, x, index):
    train = pd.concat([y_fit.rename('y'), x.rename('x')], axis=1).dropna()
    output = pd.Series(np.nan, index=index, dtype=float)
    if len(train) < MIN_PARES_TREINO or train.x.nunique() < 2:
        return output, {'status': 'treino_insuficiente', 'n_treino': len(train)}
    model = LinearRegression().fit(train[['x']], train.y)
    valid = x.notna()
    output.loc[valid] = model.predict(x.loc[valid].to_frame('x'))
    return output, {'status': 'ajustado', 'n_treino': len(train),
                    'intercepto': float(model.intercept_), 'coeficiente': float(model.coef_[0])}


def serie_float(s):
    """Normalizar números e ausências pandas para float64/np.nan."""
    valores = pd.to_numeric(s, errors='coerce').to_numpy(dtype=float, na_value=np.nan, copy=True)
    valores[~np.isfinite(valores)] = np.nan
    return pd.Series(valores, index=s.index, name=s.name)


def estimar_propria_serie(contexto, treino, tipo):
    """Modelos separados; treino observado e preenchimento causal recursivo."""
    contexto, treino = serie_float(contexto), serie_float(treino)
    lags = DEFASAGENS_PROPRIA_SERIE
    if not lags or len(set(lags)) != len(lags) or any(not isinstance(l, int) or l < 1 for l in lags):
        raise ValueError('Defasagens da própria série devem ser inteiros positivos distintos.')
    if MAX_FALHA_AUTORREFERENCIA < 1:
        raise ValueError('MAX_FALHA_AUTORREFERENCIA deve ser positivo.')
    X = pd.DataFrame({f'lag_{lag}': treino.shift(lag) for lag in lags})
    validos = treino.notna() & X.notna().all(axis=1)
    Xfit, yfit = X.loc[validos], treino.loc[validos]
    estimativas = pd.DataFrame(np.nan, index=contexto.index,
                              columns=['regressao_linear', 'knn'])
    detalhes = []
    if len(yfit) < MIN_PARES_TREINO:
        return estimativas, [{'metodo': m, 'status': 'treino_autorreferencia_insuficiente',
                             'n_treino': len(yfit)} for m in estimativas]
    modelos = {'regressao_linear': LinearRegression(),
               'knn': make_pipeline(StandardScaler(), KNeighborsRegressor(
                   n_neighbors=min(K_VIZINHOS, len(yfit), MAX_AMOSTRAS_KNN), weights='distance'))}
    ausente = contexto.isna()
    blocos = ausente.ne(ausente.shift()).cumsum()
    tamanhos = ausente.groupby(blocos).transform('size')
    elegivel = ausente & tamanhos.le(MAX_FALHA_AUTORREFERENCIA)
    for nome, modelo in modelos.items():
        xx, yy = Xfit, yfit
        if nome == 'knn' and len(xx) > MAX_AMOSTRAS_KNN:
            amostra = np.linspace(0, len(xx)-1, MAX_AMOSTRAS_KNN).astype(int)
            xx, yy = xx.iloc[amostra], yy.iloc[amostra]
        modelo.fit(xx, yy)
        reconstruida = contexto.copy()
        for i in np.flatnonzero(elegivel.to_numpy()):
            if i < max(lags): continue
            valores = [reconstruida.iloc[i-lag] for lag in lags]
            if not np.isfinite(valores).all(): continue
            entrada = pd.DataFrame([valores], columns=X.columns)
            valor = float(modelo.predict(entrada)[0])
            if NAO_NEGATIVO_P_Q and tipo in ['chuva', 'vazao']: valor = max(0., valor)
            estimativas.iloc[i, estimativas.columns.get_loc(nome)] = valor
            reconstruida.iloc[i] = valor  # Recursão apenas dentro deste método.
        detalhes.append({'metodo': nome, 'status': 'autorreferencia_causal',
                         'n_treino': len(yy), 'defasagens': str(lags),
                         'max_falha_passos': MAX_FALHA_AUTORREFERENCIA})
    return estimativas, detalhes


def corrigir_vies_multiplicativo(y_fit, produto, index):
    """Correção simples de viés para GEOGLOWS pela razão das médias.

    O fator é estimado somente onde y_fit e GEOGLOWS coexistem. Assim, durante a
    validação, os blocos artificialmente ocultados não participam da correção.
    """
    pares = pd.concat([serie_float(y_fit).rename('obs'),
                       serie_float(produto).rename('geo')], axis=1).dropna()
    saida = pd.Series(np.nan, index=index, dtype=float)
    if len(pares) < MIN_PARES_TREINO:
        return saida, {'status': 'treino_insuficiente', 'n_treino': len(pares),
                       'fator_correcao': np.nan}
    media_geo = pares.geo.mean()
    if not np.isfinite(media_geo) or media_geo == 0:
        return saida, {'status': 'media_geoglows_invalida', 'n_treino': len(pares),
                       'fator_correcao': np.nan}
    fator = pares.obs.mean() / media_geo
    valid = produto.notna()
    saida.loc[valid] = serie_float(produto).loc[valid] * fator
    return saida, {'status': 'ajustado_razao_medias', 'n_treino': len(pares),
                   'media_observada': float(pares.obs.mean()),
                   'media_geoglows': float(media_geo),
                   'fator_correcao': float(fator)}


def diagnostico_geoglows_1a1(observado, geoglows, sid, pasta):
    """Diagnóstico 1:1 do GEOGLOWS bruto e após correção multiplicativa."""
    obs = serie_float(observado)
    geo = serie_float(geoglows)
    pares = pd.concat([obs.rename('observado'), geo.rename('GEOGLOWS_bruto')], axis=1).dropna()
    if len(pares) < 2:
        print(rotulo_serie(sid), '— GEOGLOWS: pares insuficientes para análise 1:1.')
        return
    fator = pares.observado.mean() / pares.GEOGLOWS_bruto.mean() if pares.GEOGLOWS_bruto.mean() != 0 else np.nan
    pares['GEOGLOWS_corrigido'] = pares.GEOGLOWS_bruto * fator
    linhas = []
    for nome in ['GEOGLOWS_bruto', 'GEOGLOWS_corrigido']:
        a, b = pares.observado, pares[nome]
        den = ((a-a.mean())**2).sum()
        r = a.corr(b) if a.nunique()>1 and b.nunique()>1 else np.nan
        linhas.append({'metodo': nome, 'n': len(a), 'fator_correcao': 1.0 if nome.endswith('bruto') else fator,
                       'r_Pearson': r, 'R2_correlacao': r**2 if pd.notna(r) else np.nan,
                       'MAE': (a-b).abs().mean(), 'RMSE': np.sqrt(((a-b)**2).mean()),
                       'NSE': 1-((a-b)**2).sum()/den if den>0 else np.nan,
                       'PBIAS_pct': 100*(b-a).sum()/a.sum() if a.sum()!=0 else np.nan})
    pd.DataFrame(linhas).to_csv(pasta/'geoglows_diagnostico_1a1.csv', sep=';', index=False, encoding='utf-8-sig')
    pares.to_csv(pasta/'geoglows_pares_1a1.csv', sep=';', index_label='data_hora', encoding='utf-8-sig')
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    valores = np.r_[pares.observado.to_numpy(), pares.GEOGLOWS_bruto.to_numpy(), pares.GEOGLOWS_corrigido.to_numpy()]
    minimo, maximo = np.nanmin(valores), np.nanmax(valores)
    for ax, nome, titulo in zip(axes, ['GEOGLOWS_bruto','GEOGLOWS_corrigido'], ['GEOGLOWS bruto','GEOGLOWS com correção de viés']):
        ax.scatter(pares.observado, pares[nome], s=12, alpha=.45)
        ax.plot([minimo,maximo],[minimo,maximo],'k--',lw=1,label='1:1')
        ax.set_xlabel('Vazão observada (m³/s)'); ax.set_ylabel('Vazão GEOGLOWS (m³/s)')
        ax.set_title(titulo); ax.set_aspect('equal', adjustable='box'); ax.legend(fontsize=8)
    fig.suptitle(rotulo_serie(sid) + ' — análise 1:1 GEOGLOWS')
    mostrar(fig, f'diarios_{sid}_GEOGLOWS_1a1')
    print('\nGEOGLOWS — análise 1:1:', rotulo_serie(sid))
    print(pd.DataFrame(linhas).to_string(index=False))


def estimar_metodos(context, y_fit, references, products, tipo):
    context, y_fit = serie_float(context), serie_float(y_fit)
    references = references.apply(serie_float)
    products = {nome: serie_float(produto) for nome, produto in products.items()}
    predictions, info = {}, []
    predictions['media_movel'] = media_movel_falhas(context)
    info.append({'metodo': 'media_movel', 'status': 'causal', 'janela': JANELA_MEDIA_MOVEL,
                 'max_falha_passos': MAX_FALHA_MEDIA_MOVEL})
    reference_scores = []
    for col in references:
        paired = pd.concat([y_fit.rename('y'), references[col].rename('x')], axis=1).dropna()
        if len(paired) >= MIN_PARES_TREINO and paired.x.nunique() > 1:
            corr = paired.y.corr(paired.x)
            reference_scores.append((0 if pd.isna(corr) else abs(corr), len(paired), col))
    ranked = [item[2] for item in sorted(reference_scores, reverse=True)]
    reg_id = ID_REFERENCIA_REGRESSAO or (ranked[0] if ranked else None)
    if reg_id is not None and reg_id not in references:
        raise ValueError('ID_REFERENCIA_REGRESSAO não pertence às referências selecionadas.')
    if reg_id is not None:
        predictions['regressao_linear'], details = ajustar_regressao(y_fit, references[reg_id], context.index)
        info.append({'metodo': 'regressao_linear', 'referencia': reg_id, **details})
    else:
        predictions['regressao_linear'] = pd.Series(np.nan, index=context.index)
        info.append({'metodo': 'regressao_linear', 'status': 'sem_referencia_com_treino'})
    features = ranked[:MAX_REFERENCIAS_KNN]
    predictions['knn'] = pd.Series(np.nan, index=context.index, dtype=float)
    if features:
        X = references[features]
        usable = X.notna().all(axis=1) & y_fit.notna()
        training = X.loc[usable]
        yy = y_fit.loc[usable]
        if len(training) >= MIN_PARES_TREINO:
            if len(training) > MAX_AMOSTRAS_KNN:
                sample = np.linspace(0, len(training) - 1, MAX_AMOSTRAS_KNN).astype(int)
                training, yy = training.iloc[sample], yy.iloc[sample]
            # KNN supervisionado: vizinhos dos preditores nas estações auxiliares.
            # Escala estimada apenas no treino: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
            model = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=min(K_VIZINHOS, len(training)), weights='distance'))
            model.fit(training, yy)
            valid = X.notna().all(axis=1)
            if valid.any(): predictions['knn'].loc[valid] = model.predict(X.loc[valid])
            info.append({'metodo': 'knn', 'status': 'ajustado', 'n_treino': len(training), 'referencias': ' / '.join(features)})
        else:
            info.append({'metodo': 'knn', 'status': 'treino_completo_insuficiente', 'n_treino': len(training)})
    else:
        info.append({'metodo': 'knn', 'status': 'sem_referencias'})
    if USAR_PROPRIA_VAZAO and tipo == 'vazao':
        propria, detalhes = estimar_propria_serie(context, y_fit, tipo)
        for metodo in ['regressao_linear', 'knn']:
            predictions[metodo] = propria[metodo]
        info = [r for r in info if r['metodo'] not in ['regressao_linear', 'knn']] + detalhes
    for name, product in products.items():
        predictions[name + '_bruto'] = product.copy()
        if name.upper() == 'GEOGLOWS' and tipo == 'vazao':
            predictions[name + '_corrigido'], details = corrigir_vies_multiplicativo(
                y_fit, product, context.index)
            info.append({'metodo': name + '_corrigido', 'correcao': 'razao_das_medias', **details})
        else:
            predictions[name + '_corrigido'], details = ajustar_regressao(y_fit, product, context.index)
            info.append({'metodo': name + '_corrigido', 'correcao': 'regressao_linear', **details})
    if NAO_NEGATIVO_P_Q and tipo in ['chuva', 'vazao']:
        predictions = {name: values.clip(lower=0) for name, values in predictions.items()}
    return pd.DataFrame(predictions, index=context.index), pd.DataFrame(info)


def criar_mascara_validacao(y):
    """Ocultar blocos observados distribuídos no período completo, sem aleatoriedade.
    Cada bloco tem dias/horas consecutivos. Intervalos de pelo menos 4 blocos
    preservam contexto; amostras de validação nunca entram no ajuste.
    """
    if TAMANHO_BLOCO_VALIDACAO < 1 or MAX_BLOCOS_VALIDACAO < 1:
        raise ValueError('Tamanhos de validação devem ser positivos.')
    size = TAMANHO_BLOCO_VALIDACAO
    candidatos = []
    # Preservar contexto antes e depois de cada bloco ocultado.
    margem = max(size, max(DEFASAGENS_PROPRIA_SERIE, default=1))
    for start in range(margem, len(y) - size - margem + 1):
        if candidatos and start < candidatos[-1] + size * 4:
            continue
        bloco = y.iloc[start:start + size]
        if len(bloco) == size and bloco.notna().all():
            candidatos.append(start)
    if len(candidatos) > MAX_BLOCOS_VALIDACAO:
        indices = np.linspace(0, len(candidatos)-1, MAX_BLOCOS_VALIDACAO).astype(int)
        candidatos = [candidatos[i] for i in indices]
    mask = pd.Series(False, index=y.index)
    for start in candidatos:
        mask.iloc[start:start + size] = True
    if not mask.any():
        raise ValueError('Nenhum bloco observado para validação. Reduzir tamanho do bloco ou ampliar período.')
    return mask


def metricas_validacao(observado, predictions, mask):
    obs = observado.loc[mask]
    pred = predictions.loc[mask]
    common = pred.notna().all(axis=1) & obs.notna()
    rows = []
    for scope in ['cada_metodo', 'mesmos_pontos']:
        for method in pred:
            valid = common if scope == 'mesmos_pontos' else (pred[method].notna() & obs.notna())
            a, b = obs.loc[valid], pred.loc[valid, method]
            denominator = ((a - a.mean()) ** 2).sum()
            rows.append({'comparacao': scope, 'metodo': method, 'n_avaliado': len(a),
                         'n_mascarado': len(obs), 'cobertura_pct': 100 * len(a) / len(obs),
                         'MAE': (a - b).abs().mean() if len(a) else np.nan,
                         'RMSE': np.sqrt(((a - b) ** 2).mean()) if len(a) else np.nan,
                         'NSE': 1 - ((a - b) ** 2).sum() / denominator if len(a) > 1 and denominator > 0 else np.nan})
    return pd.DataFrame(rows)




def selecionar_finais(entradas, resposta, grupo, acumulado, inicio=None, fim=None):
    estudo = preparar_modelo(entradas, resposta, grupo, acumulado)
    if estudo is None:
        return None
    ids = [x for x in estudo['dados'] if x in series] + [estudo['resposta_id']]
    # Recortar antes de validar, ajustar modelos e preencher. Não apagar lacunas internas.
    ids = list(dict.fromkeys(ids))
    limites = []
    for sid in ids:
        observado = series[sid]['serie'].dropna()
        if observado.empty:
            raise ValueError(f'{sid}: sem observações para definir o período comum.')
        limites.append((observado.index.min(), observado.index.max()))
    inicio_comum = max(x[0] for x in limites)
    fim_comum = min(x[1] for x in limites)
    inicio_periodo = inicio_comum if inicio is None else pd.Timestamp(inicio)
    fim_periodo = fim_comum if fim is None else pd.Timestamp(fim)
    # Uma data sem horário inclui o último dia inteiro nos dados horários.
    if grupo == 'horarios' and isinstance(fim, str) and len(fim.strip()) == 10:
        fim_periodo += pd.Timedelta(hours=23)
    if inicio_periodo > fim_periodo:
        raise ValueError('Período inválido ou séries sem sobreposição temporal.')
    if inicio_periodo < inicio_comum or fim_periodo > fim_comum:
        raise ValueError(f'Escolha datas dentro do período comum: {inicio_comum} a {fim_comum}.')
    grade = estudo['dados'].index
    grade = grade[(grade >= inicio_periodo) & (grade <= fim_periodo)]
    if len(grade) < 2:
        raise ValueError('O recorte precisa conter pelo menos dois instantes.')
    print('Período comum selecionado:', grade.min(), 'a', grade.max())
    print('Validação de preenchimento no período inteiro:', grade.min(), 'a', grade.max())
    return {'ids': ids, 'resposta': estudo['resposta_id'], 'grupo': grupo,
            'acumulado': acumulado, 'grade': grade}


def validar_finais(selecao):
    if selecao is None:
        return {}
    resultados = {}
    for sid in selecao['ids']:
        obj = series[sid]
        y = obj['serie'].reindex(selecao['grade']).copy()
        # Só estações selecionadas, da mesma variável e de código diferente.
        refs = [ref for ref in selecao['ids'] if ref != sid
                and (sid == selecao['resposta'] or ref != selecao['resposta'])
                and series[ref]['catalogo']['tipo'] == obj['catalogo']['tipo']
                and str(series[ref]['catalogo']['codigo_estacao']) != str(obj['catalogo']['codigo_estacao'])]
        X = pd.DataFrame({ref: series[ref]['serie'].reindex(y.index) for ref in refs}, index=y.index)
        produtos = {}
        configuracoes = {}
        codigo_alvo = str(obj['catalogo']['codigo_estacao'])
        if selecao['grupo'] == 'diarios':
            if obj['catalogo']['tipo'] == 'chuva' and codigo_alvo in CHIRPS_POR_ESTACAO:
                configuracoes['CHIRPS'] = CHIRPS_POR_ESTACAO[codigo_alvo]
            if obj['catalogo']['tipo'] == 'vazao' and codigo_alvo in GEOGLOWS_POR_ESTACAO:
                configuracoes['GEOGLOWS'] = GEOGLOWS_POR_ESTACAO[codigo_alvo]
        # A configuração específica por serie_id tem prioridade.
        configuracoes.update(PRODUTOS_LOCAIS.get(sid, {}))
        for nome, cfg in configuracoes.items():
            esperado = 'chuva' if nome == 'CHIRPS' else 'vazao' if nome == 'GEOGLOWS' else None
            if selecao['grupo'] != 'diarios' or esperado != obj['catalogo']['tipo']:
                raise ValueError(f'{sid}: produto incompatível com variável/resolução.')
            produto = carregar_produto_local(cfg, y.index)
            if produto is not None: produtos[nome] = produto
        if produtos:
            print(rotulo_serie(sid), '| produtos locais:',
                  {nome: int(produto.notna().sum()) for nome, produto in produtos.items()})
        # Blocos ocultados no período inteiro; ajustar somente observações não ocultadas.
        try:
            mask = criar_mascara_validacao(y)
        except ValueError as erro:
            print(sid, '— validação indisponível:', erro)
            continue
        contexto = y.copy()
        ocultos = mask.index[mask]
        contexto.loc[ocultos] = np.nan
        fit = contexto.copy()
        pred, parametros = estimar_metodos(contexto, fit, X, produtos, obj['catalogo']['tipo'])
        mask_completa = mask.reindex(y.index, fill_value=False)
        metricas = metricas_validacao(y, pred, mask_completa)
        pasta = RESULTADOS / selecao['grupo'] / sid
        pasta.mkdir(parents=True, exist_ok=True)
        # Diagnóstico 1:1 usa apenas pares originalmente observados e é separado da validação.
        if selecao['grupo'] == 'diarios' and obj['catalogo']['tipo'] == 'vazao' and 'GEOGLOWS' in produtos:
            diagnostico_geoglows_1a1(y, produtos['GEOGLOWS'], sid, pasta)
        metricas.to_csv(pasta / 'validacao.csv', sep=';', index=False)
        parametros.to_csv(pasta / 'parametros_validacao.csv', sep=';', index=False)
        teste = pred.loc[ocultos].copy()
        teste.insert(0, 'observado', y.loc[ocultos])
        teste.to_csv(pasta / 'falhas_artificiais.csv', sep=';', index_label='data_hora')
        print('\nSérie:', rotulo_serie(sid), '| referências:', [rotulo_serie(ref) for ref in refs])
        print(metricas[metricas.comparacao.eq('cada_metodo')].to_string(index=False))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(teste.index, teste.observado, 'ko', markersize=4, label='Observado ocultado')
        for metodo in pred:
            ax.plot(teste.index, teste[metodo], '.', label=metodo)
        ax.set_title('Validação do preenchimento — ' + rotulo_serie(sid))
        ax.legend(fontsize=7); ax.set_ylabel('Valor na unidade da fonte')
        mostrar(fig, f'{selecao["grupo"]}_{sid}_validacao')
        resultados[sid] = {'y': y, 'X': X, 'produtos': produtos, 'metricas': metricas}
    return resultados


def resolver_metodos(selecao, escolhas):
    """Aceita serie_id ou (código, variável); ambiguidades exigem serie_id."""
    resolvidos = {}
    for chave, metodo in escolhas.items():
        if isinstance(chave, tuple) and len(chave) == 2:
            codigo, tipo = chave
            ids = [sid for sid in selecao['ids']
                   if str(series[sid]['catalogo']['codigo_estacao']) == str(codigo)
                   and series[sid]['catalogo']['tipo'] == tipo]
        elif isinstance(chave, str) and chave in selecao['ids']:
            ids = [chave]
        else:
            raise ValueError(f'Escolha inválida {chave}: usar (código, variável) ou serie_id selecionado.')
        if len(ids) != 1:
            raise ValueError(f'{chave}: {len(ids)} séries selecionadas {ids}. Informar serie_id para distinguir fontes.')
        if ids[0] in resolvidos:
            raise ValueError(f'Método repetido para {rotulo_serie(ids[0])}.')
        resolvidos[ids[0]] = metodo
    return resolvidos


def plotar_antes_depois(selecao, finais):
    """Resposta e uma chuva: observado versus resultado do preenchimento."""
    escolha = CHUVA_GRAFICO_DIARIO if selecao['grupo'] == 'diarios' else CHUVA_GRAFICO_HORARIO
    chuvas = [sid for sid in selecao['ids'] if sid != selecao['resposta']
              and series[sid]['catalogo']['tipo'] == 'chuva']
    if escolha is None:
        chuva = chuvas[0] if chuvas else None
    else:
        candidatos = [sid for sid in chuvas if (sid == escolha if isinstance(escolha, str)
            else isinstance(escolha, tuple) and len(escolha) == 2
            and str(series[sid]['catalogo']['codigo_estacao']) == str(escolha[0])
            and series[sid]['catalogo']['tipo'] == escolha[1])]
        if len(candidatos) != 1:
            raise ValueError('Chuva do gráfico ausente ou ambígua: selecionar serie_id das entradas finais.')
        chuva = candidatos[0]
    ids = [selecao['resposta']] + ([chuva] if chuva else [])
    fig, axes = plt.subplots(len(ids), 2, figsize=(14, 4*len(ids)),
                             sharex=True, sharey='row', squeeze=False)
    for row, sid in enumerate(ids):
        original = serie_float(series[sid]['serie'].reindex(selecao['grade']))
        final = serie_float(finais[sid])
        preenchido = original.isna() & final.notna()
        nome = rotulo_serie(sid)
        unidade = {'chuva': 'Precipitação (mm)', 'vazao': 'Vazão (unidade da fonte)',
                   'nivel': 'Nível (unidade da fonte)'}[series[sid]['catalogo']['tipo']]
        axes[row, 0].plot(original.index, original, color='#64748b', linewidth=.8,
                          label='Observado')
        axes[row, 1].plot(final.index, final, color='#287c94', linewidth=.8,
                          label='Série após preenchimento')
        axes[row, 1].scatter(final.index[preenchido], final[preenchido],
                             c='#e07825', s=16, zorder=5, label=f'Preenchidos: {int(preenchido.sum())}')
        for col, titulo in enumerate(['Antes', 'Depois']):
            axes[row, col].set_title(titulo + ' — ' + nome, fontsize=9)
            axes[row, col].legend(fontsize=8)
            axes[row, col].set_ylabel(unidade)
    for ax in axes[-1]: ax.set_xlabel('Data/hora')
    fig.suptitle('Antes e depois do preenchimento — ' + selecao['grupo'] +
                 ' | período completo', fontsize=11)
    mostrar(fig, selecao['grupo'] + '_resposta_chuva_antes_depois')
    if chuva: print('Chuva utilizada na comparação:', rotulo_serie(chuva))


def media_climatologica_outros_anos(observada, grupo):
    """Média da própria estação, mesmo mês/dia/hora, excluindo o ano alvo.
    Usa somente observações originais no período selecionado, nunca estimativas.
    Reconstrução retrospectiva: inclui outros anos posteriores ao instante alvo.
    """
    obs = serie_float(observada).dropna()
    resultado = pd.Series(np.nan, index=observada.index, dtype=float)
    tabela = pd.DataFrame({'valor': obs, 'mes': obs.index.month, 'dia': obs.index.day,
                           'hora': obs.index.hour if grupo == 'horarios' else 0,
                           'ano': obs.index.year})
    # Pré-agrupar para não percorrer toda a série para cada lacuna horária.
    por_chave = {chave: g for chave, g in tabela.groupby(['mes', 'dia', 'hora'])}
    for data in observada.index[observada.isna()]:
        hora = data.hour if grupo == 'horarios' else 0
        chaves = [(data.month, data.day, hora)]
        if data.month == 2 and data.day == 29:
            chaves = [(2, 28, hora), (3, 1, hora)]
        amostras = [por_chave[c].loc[por_chave[c].ano.ne(data.year), 'valor']
                    for c in chaves if c in por_chave]
        if amostras:
            resultado.loc[data] = pd.concat(amostras).mean()
    return resultado


def preencher_finais(selecao, validacoes, escolhas):
    if selecao is None:
        return
    escolhas = resolver_metodos(selecao, escolhas)
    finais, flags, origens, resumo = {}, {}, {}, []
    for sid in selecao['ids']:
        y = serie_float(series[sid]['serie'].reindex(selecao['grade']))
        pasta = RESULTADOS / selecao['grupo'] / sid
        pasta.mkdir(parents=True, exist_ok=True)
        metodo = escolhas.get(sid)
        if y.isna().any() and metodo is None:
            raise ValueError(f'{rotulo_serie(sid)}: escolha o método principal em METODOS; base completa exige método.')
        v = validacoes.get(sid)
        if v is not None:
            # Método escolhido: reajustar com TODAS as observações originais disponíveis.
            fit = y.copy()
            pred, info = estimar_metodos(y, fit, v['X'], v['produtos'], series[sid]['catalogo']['tipo'])
            info.to_csv(pasta / 'parametros_preenchimento.csv', sep=';', index=False)
            if metodo is not None and metodo not in pred:
                raise ValueError(f'{sid}: método desconhecido/indisponível: {metodo}.')
            alternativas = pd.DataFrame({'observado': y})
            for nome in pred:
                alternativas[nome] = y.fillna(pred[nome])
            alternativas.to_csv(pasta / 'alternativas_preenchimento.csv', sep=';', index_label='data_hora')
            candidato = pred[metodo] if metodo is not None else pd.Series(np.nan, index=y.index)
        else:
            # Sem amostra suficiente de validação, não aplicar modelo não avaliado.
            # Ainda é possível reconstruir pela climatologia observada de outros anos.
            print(rotulo_serie(sid), '— sem validação disponível; usar climatologia nas lacunas.')
            candidato = pd.Series(np.nan, index=y.index)
        final = y.fillna(serie_float(candidato).reindex(y.index))
        origem = pd.Series('ausente', index=y.index, dtype=object)
        origem.loc[y.notna()] = 'observado'
        origem.loc[y.isna() & final.notna()] = metodo
        climatologia = media_climatologica_outros_anos(y, selecao['grupo'])
        mask_clima = final.isna() & climatologia.notna()
        final = final.fillna(climatologia)
        origem.loc[mask_clima] = 'media_climatologica'
        finais[sid] = final
        flags[sid] = y.isna() & final.notna()
        origens[sid] = origem
        exportar = pd.DataFrame({'observado': y, 'valor_final': final,
                                 'foi_preenchido': flags[sid], 'origem': origem})
        exportar.to_csv(pasta / 'serie_final.csv', sep=';', index_label='data_hora')
        resumo.append({'serie_id': sid, 'estacao': rotulo_serie(sid),
                       'falhas_originais': int(y.isna().sum()),
                       'preenchidos_metodo': int((y.isna() & final.notna() & ~mask_clima).sum()),
                       'preenchidos_climatologia': int(mask_clima.sum()),
                       'falhas_restantes': int(final.isna().sum())})
    diagnostico = pd.DataFrame(resumo)
    diagnostico.to_csv(RESULTADOS / f'{selecao["grupo"]}_diagnostico_preenchimento.csv', sep=';', index=False)
    print(diagnostico.to_string(index=False))
    pendencias = [{'serie_id': sid, 'estacao': rotulo_serie(sid), 'data_hora': data}
                  for sid, y in finais.items() for data in y.index[y.isna()]]
    if pendencias:
        arquivo = RESULTADOS / f'{selecao["grupo"]}_lacunas_pendentes.csv'
        pd.DataFrame(pendencias).to_csv(arquivo, sep=';', index=False)
        raise ValueError(f'Exportação bloqueada: {len(pendencias)} lacunas sem estimativa. '
                         f'Consultar {arquivo}. CSV final existente não foi atualizado.')
    dados = pd.DataFrame(finais)
    origem = pd.DataFrame(flags).add_suffix('_preenchido')
    for sid in selecao['ids']:
        if sid != selecao['resposta'] and series[sid]['catalogo']['tipo'] == 'chuva':
            janela = selecao['acumulado']
            dados[f'{sid} | P_acum_{janela}'] = dados[sid].rolling(janela, min_periods=janela).sum()
    dados = dados.rename(columns={selecao['resposta']: 'RESPOSTA'})
    pd.concat([dados, origem], axis=1).to_csv(RESULTADOS / f'{selecao["grupo"]}_base_final.csv', sep=';', index_label='data_hora')
    # Arquivo consolidado: uma linha por data/hora e uma coluna por série.
    # Manter o código da estação resposta no nome, em vez do rótulo RESPOSTA.
    sufixos = {'chuva': 'P', 'vazao': 'Q', 'nivel': 'H'}
    nomes = {}
    for sid in finais:
        registro = series[sid]['catalogo']
        tipo = registro['tipo']
        if tipo not in sufixos:
            raise ValueError(f'{sid}: variável sem sufixo de exportação: {tipo}.')
        nomes[sid] = f"{registro['codigo_estacao']}_{sufixos[tipo]}"
    if len(set(nomes.values())) != len(nomes):
        raise ValueError('Colunas finais duplicadas: selecione uma única série por estação/variável ' 
                         '(por exemplo, diária ANA ou diária derivada).')
    consolidado = pd.DataFrame(finais).rename(columns=nomes)
    nome_final = 'series_diarias_preenchidas.csv' if selecao['grupo'] == 'diarios' else 'series_horarias_preenchidas.csv'
    DIRETORIO_CSVS_FINAIS.mkdir(parents=True, exist_ok=True)
    # Validação já executada: exportar atomicamente para não deixar arquivo parcial.
    destino = DIRETORIO_CSVS_FINAIS / nome_final
    temporario = destino.with_suffix('.tmp')
    consolidado.sort_index().to_csv(temporario, sep=';',
                                   index_label='data_hora', encoding='utf-8-sig')
    temporario.replace(destino)
    pd.DataFrame(origens).rename(columns=nomes).to_csv(
        RESULTADOS / f'{selecao["grupo"]}_origem_valores.csv', sep=';', index_label='data_hora')
    plotar_antes_depois(selecao, finais)
    print('Séries consolidadas:', DIRETORIO_CSVS_FINAIS / nome_final)
    print('Base final completa: zero lacunas nas séries selecionadas. Origem dos valores salva.')


# %% 4. Usar o diretório de dados informado pelo professor
# Discussão: De quais dados vamos partir? Onde salvaremos os resultados?
RAIZ = Path(DIRETORIO_DADOS)
if not RAIZ.is_dir(): raise FileNotFoundError(f'Diretório dos dados não encontrado: {RAIZ}')
obrigatorios = [ARQUIVO_CATALOGO, ARQUIVO_COORDENADAS, ARQUIVO_BACIA,
                ARQUIVO_ESTACOES]
faltantes = [str(p) for p in obrigatorios if not Path(p).is_file()]
if faltantes:
    raise FileNotFoundError('Conferir os arquivos copiados e os caminhos da célula 2:\n' + '\n'.join(faltantes))
# Os caminhos arquivo_final do catálogo são relativos a DIRETORIO_DADOS;
# copiar o conteúdo preparado mantendo as subpastas diarios e horarios.
RESULTADOS = DIRETORIO_RESULTADOS / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
RESULTADOS.mkdir(parents=True, exist_ok=False)

print('Dados:', RAIZ)
print('Resultados:', RESULTADOS)


# %% 5. Ler catálogo e coordenadas
# Discussão: O código da estação identifica uma única série?
catalogo = conferir_arquivos_catalogo(tabela(ARQUIVO_CATALOGO), RAIZ)
coordenadas = tabela(ARQUIVO_COORDENADAS)
ROTULOS_SERIES = criar_rotulos(catalogo)
mapeamento_rotulos = catalogo[['serie_id', 'codigo_estacao', 'tipo', 'fonte', 'grupo']].copy()
mapeamento_rotulos['rotulo'] = mapeamento_rotulos.serie_id.astype(str).map(ROTULOS_SERIES)
salvar_tabela(mapeamento_rotulos, 'rotulos_series.csv')
print('Dados:', RAIZ)
print(mapeamento_rotulos.to_string(index=False))
print('\nCada fonte/arquivo mantém seu ID; não fundir séries somente pelo código.')


# %% 6. Contar séries por variável e resolução
# Discussão: Temos as mesmas variáveis no estudo diário e horário?
resumo_catalogo = catalogo.groupby(['grupo', 'tipo', 'fonte']).agg(estacoes=('codigo_estacao', 'nunique'), series=('serie_id', 'nunique')).reset_index()
print(resumo_catalogo.to_string(index=False))


# %% 7. Carregar a bacia e conferir o CRS
# Discussão: Por que precisamos identificar o sistema de coordenadas?
bacia = gpd.read_file(ARQUIVO_BACIA, **({'layer': CAMADA_BACIA} if CAMADA_BACIA else {}))
if CAMPO_BACIA:
    if CAMPO_BACIA not in bacia:
        raise ValueError(f'Campo da bacia ausente: {CAMPO_BACIA}. Colunas: {list(bacia.columns)}')
    bacia = bacia[bacia[CAMPO_BACIA].astype(str).eq(str(VALOR_BACIA))].copy()
elif len(bacia) > 1:
    raise ValueError(f'GeoPackage com vários polígonos. Configurar CAMPO_BACIA e VALOR_BACIA ou usar apenas o polígono escolhido. Colunas: {list(bacia.columns)}')
if bacia.empty or not bacia.geom_type.isin(['Polygon', 'MultiPolygon']).all():
    raise ValueError('Nenhum polígono válido selecionado para a bacia.')
if bacia.crs is None: raise ValueError('Bacia sem CRS.')
bacia = bacia.to_crs(CRS_MAPA)

print('CRS do mapa:', bacia.crs)
print(bacia.drop(columns='geometry').head())


# %% 8. Carregar a rede de drenagem
# Discussão: Qual arquivo representa os rios? Está no mesmo CRS?
arquivo_rede, camada_rede = ARQUIVO_DRENAGEM, CAMADA_DRENAGEM
rede = gpd.GeoDataFrame(geometry=[], crs=CRS_MAPA)
if arquivo_rede is not None:
    arquivo_rede = Path(arquivo_rede)
    if arquivo_rede.is_file():
        rede = gpd.read_file(arquivo_rede, **({'layer': camada_rede} if camada_rede else {}))
        if rede.crs is None: raise ValueError('Drenagem sem CRS.')
        rede = rede[rede.geom_type.isin(['LineString', 'MultiLineString'])].to_crs(CRS_MAPA)
    else:
        print('Arquivo de drenagem ausente; continuar com bacia e estações.')
else:
    print('Drenagem não configurada; mapa com bacia e estações.')

# %% 9. Recortar a drenagem, quando disponível
drenagem_bacia = gpd.clip(rede, bacia) if not rede.empty else rede.copy()
print('Trechos de drenagem selecionados:', len(drenagem_bacia))


# %% 10. Preparar as estações para o mapa
# Discussão: Uma estação pode ter chuva e também nível ou vazão?
pontos = gpd.read_file(ARQUIVO_ESTACOES,
                      **({'layer': CAMADA_ESTACOES} if CAMADA_ESTACOES else {}))
if pontos.crs is None: raise ValueError('Estações internas sem CRS.')
if CAMPO_CODIGO_ESTACOES not in pontos:
    raise ValueError(f'Campo de código ausente: {CAMPO_CODIGO_ESTACOES}. Colunas: {list(pontos.columns)}')
if pontos.geometry.isna().any() or not pontos.geom_type.eq('Point').all():
    raise ValueError('A camada das estações deve conter pontos válidos.')
pontos['codigo'] = pontos[CAMPO_CODIGO_ESTACOES].astype(str).str.strip()
pontos = pontos.to_crs(CRS_MAPA)
codigos_catalogo = set(catalogo.codigo_estacao.astype(str))
pontos = pontos[pontos.codigo.isin(codigos_catalogo)].copy()
limite_bacia = bacia.geometry.union_all()
pontos = pontos[pontos.geometry.apply(limite_bacia.covers)].copy()
pontos['_posicao'] = pontos.geometry.to_wkb()
pontos = pontos.drop_duplicates(['codigo', '_posicao']).drop(columns='_posicao')
faltantes = sorted(codigos_catalogo - set(pontos.codigo))
if faltantes:
    print('Séries sem posição interna no GeoPackage (seguem nas análises):', faltantes)
if pontos.empty:
    raise ValueError('Nenhuma estação do catálogo localizada dentro da bacia selecionada.')
variables = catalogo.groupby('codigo_estacao').tipo.agg(set).to_dict()
def classe_estacao(code):
    kinds = variables.get(str(code), set())
    return 'Chuva + nível/vazão' if 'chuva' in kinds and kinds.intersection({'nivel', 'vazao'}) else 'Pluviométrica' if 'chuva' in kinds else 'Fluviométrica (nível/vazão)'
pontos['classe'] = pontos.codigo.map(classe_estacao)

print(pontos[['codigo', 'classe']].to_string(index=False))


# %% 11. Plotar bacia, rios e estações
# Discussão: Como as estações estão distribuídas no espaço?
fig, ax = plt.subplots(figsize=(10, 9))
bacia.plot(ax=ax, facecolor='#f3f6f4', edgecolor='#203d48', linewidth=1.5)
if not drenagem_bacia.empty:
    drenagem_bacia.plot(ax=ax, color='#61a9cb', linewidth=.6)
styles = {'Pluviométrica': ('^', '#1a8a65'), 'Fluviométrica (nível/vazão)': ('o', '#bc4937'), 'Chuva + nível/vazão': ('D', '#754c9e')}
handles = []
if not drenagem_bacia.empty:
    handles.append(Line2D([0], [0], color='#61a9cb', label='Drenagem'))
for cls, (marker, color) in styles.items():
    subset = pontos[pontos.classe.eq(cls)]
    if subset.empty: continue
    ax.scatter(subset.geometry.x, subset.geometry.y, marker=marker, c=color, s=55, edgecolors='white', linewidth=.6, zorder=5)
    handles.append(Line2D([0], [0], marker=marker, color='none', markerfacecolor=color, markersize=7, label=cls))
for _, point in pontos.iterrows():
    ax.annotate(point.codigo, (point.geometry.x, point.geometry.y), xytext=(5, 5), textcoords='offset points', fontsize=8)
ax.set_title(f'{BACIA} — estações disponíveis para os estudos diário e horário')
ax.set_xlabel('Coordenada E (m)'); ax.set_ylabel('Coordenada N (m)'); ax.set_aspect('equal')
ax.legend(handles=handles, loc='best', fontsize=8)
mostrar(fig, '01_mapa_bacia_estacoes_drenagem')
print('Rede utilizada:', arquivo_rede, '| camada:', camada_rede)


# %% 12. Carregar as séries e construir a visão analítica
# Discussão: Como preservar as lacunas sem alterar os arquivos originais?
if PERIODO_ANALISE and pd.Timestamp(PERIODO_ANALISE[1]) < pd.Timestamp(PERIODO_ANALISE[0]):
    raise ValueError('PERIODO_ANALISE: fim anterior ao início.')
series, diagnostico = carregar_series(catalogo, RAIZ, PERIODO_ANALISE)
# Nome legível aplicado a cada série antes de qualquer gráfico ou análise.
for sid, obj in series.items():
    obj['serie'].name = rotulo_serie(sid)
salvar_tabela(diagnostico, '01_diagnostico_series.csv')
if not any(len(obj['serie']) for obj in series.values()):
    raise ValueError('Nenhuma série tem registros no período escolhido. Revisar PERIODO_ANALISE.')


# %% 13. Examinar datas e duplicatas
# Discussão: Quando duas observações no mesmo horário podem gerar conflito?
colunas_qualidade = ['serie_id', 'codigo_estacao', 'grupo', 'tipo', 'datas_invalidas', 'linhas_com_timestamp_repetido', 'timestamps_conflitantes', 'timestamps_fora_grade']
print(diagnostico.reindex(columns=colunas_qualidade).to_string(index=False))
print('Duplicatas concordantes: um valor na visão analítica. Conflitantes: ausentes nessa visão. Originais preservados.')


# %% 14. Sinalizar negativos e valores pelo IQR
# Discussão: Um valor extremo necessariamente representa um erro?
flags = [obj['flags'] for obj in series.values() if 'flags' in obj]
salvar_tabela(pd.concat(flags, ignore_index=True) if flags else [], '02_sinalizacoes_qualidade.csv')

print(diagnostico.reindex(columns=['serie_id', 'negativos', 'sinalizados_IQR', 'IQR']).to_string(index=False))
print('IQR da chuva usa somente P > 0. Os zeros permanecem nos dados, no Gantt e nas correlações.')


# %% 15. Boxplot da vazão — investigação de extremos
# Discussão: O boxplot permite concluir que uma cheia é um erro?
boxplot_iqr(series, 'vazao')


# %% 16. Boxplot da precipitação — somente chuva positiva
# Discussão: Por que o grande número de zeros altera o IQR?
boxplot_iqr(series, 'chuva')


# %% 16A. Histogramas — distribuição de frequência de vazão e chuva
# Um gráfico por estação, variável e resolução, com os dados observados.
# Alterar para False para incluir dias/horas sem chuva no histograma de P.
CHUVA_SEM_ZEROS_HISTOGRAMA = True
CLASSES_HISTOGRAMA = 'auto'  # Ou um inteiro, por exemplo 20.
histogramas_series(series, CLASSES_HISTOGRAMA, CHUVA_SEM_ZEROS_HISTOGRAMA)


# %% 16B. Curvas de permanência — vazões diárias e Q5, Q10, Q90, Q98
# Q90: vazão igualada ou excedida em 90% dos dias válidos.
# Não preencher falhas nem remover extremos pelo IQR nesta análise.
# Nesta etapa usamos o período observado disponível, anterior ao recorte final.
vazoes_permanencia = curva_permanencia_diaria(series)


# %% 17. Identificar dias inteiros sem observação
# Discussão: Falhas curtas e longas têm o mesmo efeito na análise?
falhas = [{'serie_id': sid, 'inicio': str(a.date()), 'fim': str(b.date()), 'duracao_dias': n}
          for sid, obj in series.items() for a, b, n in obj.get('falhas', [])]
salvar_tabela(falhas, '03_falhas_dias_inteiros.csv')

print(pd.DataFrame(falhas).head(20).to_string(index=False))


# %% 18. Gantt da vazão — Q
# Discussão: Onde estão as lacunas das séries de vazão?
grafico_gantt(series, 'vazao')


# %% 19. Gantt da precipitação — P
# Discussão: Quais estações têm chuva disponível no mesmo período?
grafico_gantt(series, 'chuva')
print('Resoluções aparecem em linhas separadas. Dia com algum dado não equivale a 24 horas completas.')


# %% 20. DIÁRIOS — listar e escolher entradas e resposta
print(catalogo.loc[catalogo.grupo.eq('diarios'),
                  ['codigo_estacao', 'tipo', 'fonte', 'serie_id']].to_string(index=False))
# Preencher com (código, variável), preservando o código entre aspas.
# Ex.: ENTRADAS_DIARIAS = [('CODIGO_PLU', 'chuva'), ('CODIGO_FLU', 'vazao')]
# Ex.: RESPOSTA_DIARIA = ('71300000', 'vazao')
# Se houver mais de uma série da estação/variável, usar o serie_id como string.
ENTRADAS_DIARIAS = [('2463', 'chuva'), ('84100000', 'chuva')]
RESPOSTA_DIARIA = ('2463', 'vazao')
ACUMULADO_DIAS = 7  # Número de dias, incluindo o dia atual.
DEFASAGENS_DIARIAS = list(range(0, 16))  # Dias: 0 a 15; ajustar durante a aula.

# %% 21. DIÁRIOS — alinhar dados e acumular precipitação
estudo_diario = preparar_modelo(ENTRADAS_DIARIAS, RESPOSTA_DIARIA,
                               'diarios', ACUMULADO_DIAS)

# %% 22. DIÁRIOS — correlação com a resposta e matriz de correlação
correlacoes_diarias = analisar_correlacao(estudo_diario, DEFASAGENS_DIARIAS)

# %% 23. HORÁRIOS — listar e escolher entradas e resposta
print(catalogo.loc[catalogo.grupo.eq('horarios'),
                  ['codigo_estacao', 'tipo', 'fonte', 'serie_id']].to_string(index=False))
# Ex.: ENTRADAS_HORARIAS = [('CODIGO_PLU', 'chuva'), ('CODIGO_FLU', 'nivel')]
# Ex.: RESPOSTA_HORARIA = ('CODIGO_ALVO', 'nivel')
ENTRADAS_HORARIAS = [('2463', 'chuva'), ('84100000', 'chuva')]
RESPOSTA_HORARIA = ('2463', 'vazao')
ACUMULADO_HORAS = 24  # Número de horas, incluindo a hora atual.
DEFASAGENS_HORARIAS = list(range(0, 25))  # Horas: 0 a 24.

# %% 24. HORÁRIOS — alinhar dados e acumular precipitação
estudo_horario = preparar_modelo(ENTRADAS_HORARIAS, RESPOSTA_HORARIA,
                                'horarios', ACUMULADO_HORAS)

# %% 25. HORÁRIOS — correlação com a resposta e matriz de correlação
correlacoes_horarias = analisar_correlacao(estudo_horario, DEFASAGENS_HORARIAS)
# Cada par usa apenas datas/horas com ambos os valores disponíveis.
# Acumulados exigem uma janela completa; lacunas não são tratadas como zero.
# Correlação é exploratória; seleção de entradas ML será tratada na etapa de modelagem.

# %% 26. DIÁRIOS — selecionar as estações que continuarão
# Preencher após avaliar a correlação. Mesma sintaxe da seleção inicial.
ENTRADAS_DIARIAS_FINAIS = [('2463', 'chuva'), ('84100000', 'chuva')]
RESPOSTA_DIARIA_FINAL = RESPOSTA_DIARIA
# Recorte inclusivo; None = limite do período comum das séries selecionadas.
INICIO_DIARIOS = None  # Ex.: '2015-01-01'
FIM_DIARIOS = None  # Ex.: '2024-12-31'; horários aceitam data e hora.
selecao_diaria = selecionar_finais(ENTRADAS_DIARIAS_FINAIS, RESPOSTA_DIARIA_FINAL,
                                  'diarios', ACUMULADO_DIAS,
                                  INICIO_DIARIOS, FIM_DIARIOS)

# %% 27. DIÁRIOS — validar média móvel, regressão e KNN
validacao_diaria = validar_finais(selecao_diaria)

# %% 28. DIÁRIOS — escolher métodos e preencher lacunas reais
# Informar serie_id e método após examinar validacao.csv.
# Ex.: METODOS_DIARIOS = {('2463', 'chuva'): 'regressao_linear',
#                        ('2463', 'vazao'): 'knn'}
# Se houver várias séries da mesma estação/variável, usar o serie_id.
# Opções: media_movel, regressao_linear, knn; produtos locais: CHIRPS_bruto,
# CHIRPS_corrigido, GEOGLOWS_bruto, GEOGLOWS_corrigido.
# GEOGLOWS_corrigido usa correção multiplicativa pela razão das médias.
# Informar um método para cada série com lacunas; climatologia completa as falhas restantes.
METODOS_DIARIOS = {('2463', 'chuva'): 'regressao_linear',
                        ('2463', 'vazao'): 'GEOGLOWS_bruto', ('84100000', 'chuva'): 'knn'}
preencher_finais(selecao_diaria, validacao_diaria, METODOS_DIARIOS)

# %% 29. HORÁRIOS — selecionar as estações que continuarão
ENTRADAS_HORARIAS_FINAIS = [('2463', 'chuva'), ('84100000', 'chuva')]
RESPOSTA_HORARIA_FINAL = RESPOSTA_HORARIA
# Recorte inclusivo; None = limite do período comum das séries selecionadas.
INICIO_HORARIOS = None  # Ex.: '2015-01-01'
FIM_HORARIOS = None  # Ex.: '2024-12-31'; horários aceitam data e hora.
selecao_horaria = selecionar_finais(ENTRADAS_HORARIAS_FINAIS, RESPOSTA_HORARIA_FINAL,
                                   'horarios', ACUMULADO_HORAS,
                                   INICIO_HORARIOS, FIM_HORARIOS)

# %% 30. HORÁRIOS — validar os três métodos nas séries selecionadas
validacao_horaria = validar_finais(selecao_horaria)

# %% 31. HORÁRIOS — escolher métodos e preencher lacunas reais
METODOS_HORARIOS = {('2463', 'chuva'): 'knn',
                        ('2463', 'vazao'): 'regressao_linear', ('84100000', 'chuva'): 'regressao_linear'}  # Ex.: {('2463', 'nivel'): 'knn'}
preencher_finais(selecao_horaria, validacao_horaria, METODOS_HORARIOS)
# Produtos diários não são usados em dados horários.
# Média móvel causal: apenas passado; limitada a falhas curtas.
# Regressão/KNN: referências observadas da mesma variável e resolução.
# Não há preenchimento encadeado com valores estimados de outras estações.
# Sem separação treino/teste ML nesta etapa: ajuste final em todo o período observado.

# Vazão: USAR_PROPRIA_VAZAO=True habilita regressão e KNN usando Q passada.
# Falhas consecutivas são reconstruídas recursivamente, separadamente por método,
# limitadas por MAX_FALHA_AUTORREFERENCIA. Sem histórico suficiente, ficam NaN.
# Produtos diários são opções adicionais: GEOGLOWS_bruto/GEOGLOWS_corrigido
# para vazão; CHIRPS_bruto/CHIRPS_corrigido para chuva. GEOGLOWS usa razão das médias;
# o fator é estimado apenas com observações não ocultadas na validação.
# Escolher o nome do método em METODOS_DIARIOS após examinar a validação.
# Preencher todo o período; produtos diários não são desagregados em horas.

# %% 32. Conferir os dois arquivos consolidados para a próxima etapa
for nome in ['series_diarias_preenchidas.csv', 'series_horarias_preenchidas.csv']:
    arquivo = DIRETORIO_CSVS_FINAIS / nome
    if arquivo.is_file():
        print('Gerado:', arquivo)
    else:
        print('Ainda não gerado:', nome, '— selecionar séries e executar o bloco correspondente.')
# Arquivos contêm somente as séries selecionadas, com os métodos escolhidos.
# Sem método escolhido, a série observada é mantida. Lacunas não preenchíveis
# O preenchimento final usa climatologia para lacunas restantes; nunca converter ausências em zero.
