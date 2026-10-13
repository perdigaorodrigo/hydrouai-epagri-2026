# -*- coding: utf-8 -*-
"""Diagnóstico EPAGRI por bacia, estação, variável, fonte e resolução.
Executar no ambiente epagri2026. Não exige diagnósticos anteriores.
Não altera os originais, não preenche lacunas e não baixa dados.
"""
from pathlib import Path
import calendar
import csv
import hashlib
import html
import json
import re
import shutil
import unicodedata
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ================= CONFIGURAÇÃO EDITÁVEL =================
BASE = Path(r'C:\Users\afrod\OneDrive\Documents\HydroUAI_git\hydrouai-epagri-2026\02_Dados\BD01\BD01')
SIG = BASE / 'SIG'
ARQUIVO_BACIAS = SIG / 'subbacias.shp'
CAMADA_BACIAS = None  # Nome da camada quando o arquivo é GeoPackage.
CAMPO_EXUTORIO = None  # None: detectar; pode informar o nome exato.
DIRETORIOS = {'ANA': BASE / 'BD_ANA' / 'vazoes', 'Epagri': BASE / 'BD_Epagri'}
# Todas as estações internas + externas a até esta distância da borda.
DISTANCIA_BORDA_KM = 0
CRS_METRICO = 31982  # SIRGAS 2000 / UTM 22S; adequado à região de SC.
ARQUIVOS_ESTACOES = None  # None: procurar camadas de pontos em SIG.
# Nova pasta a cada execução: evita resultados antigos misturados.
SAIDA = BASE / 'Diagnostico_Geral_Epagri' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
EXPORTAR_DADOS_POR_BACIA = True  # False: só catálogo, mapas e relatórios.
PERIODO_ANALISE = None  # Exemplo: ('2018-01-01', '2025-12-31'); None = cada série inteira.
COLUNA_DATA = None
COLUNA_CODIGO = None
SENTINELAS = [-9999, -999, -99999]  # Conferir os valores usados na fonte.
PRIORIDADES = ['2284', '2463', '2464', '2468']
# Equivalências documentadas: informativas; não fundir observações.
EQUIVALENCIAS = {'Cubatão': ['2464', '84150050', '84150100'],
                'Vargem do Braço': ['2463', '84207000']}
# Correções específicas de nomes, código e coordenadas, se necessárias:
# {'arquivo.csv': {'codigo': 'Station', 'latitude': 'Lat', 'longitude': 'Lon', 'nome': 'Name'}}
COLUNAS_POSICAO = {}
MAX_SERIES_GANTT = 25

def normalizar(x):
    x = unicodedata.normalize('NFKD', str(x)).encode('ascii', 'ignore').decode()
    return re.sub(r'[^a-z0-9]', '', x.lower())


def codigo(x):
    if pd.isna(x):
        return ''
    return re.sub(r'\.0$', '', str(x).strip())


def numeros(s):
    # Converte decimal brasileiro sem assumir que ponto é separador de milhar.
    t = s.astype('string').str.strip()
    brasileiro = t.str.contains(',', na=False)
    t.loc[brasileiro] = t.loc[brasileiro].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    return pd.to_numeric(t, errors='coerce').replace(SENTINELAS, np.nan)


def datas(s):
    # Formatos explícitos evitam inversão de dia/mês. Excel serial é opcional
    # e não é inferido de números, para não confundir códigos com datas.
    out = pd.Series(pd.NaT, index=s.index, dtype='datetime64[ns]')
    for fmt in ['%d/%m/%Y %H:%M:%S', '%d/%m/%Y %H:%M', '%d/%m/%Y',
                '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%d',
                '%Y-%m-%dT%H:%M:%S', '%d-%m-%Y', '%Y%m%d']:
        mask = out.isna()
        out.loc[mask] = pd.to_datetime(s.loc[mask], format=fmt, errors='coerce')
    return out


def salvar_tabela(rows, nome):
    pd.DataFrame(rows).to_csv(SAIDA / nome, sep=';', encoding='utf-8-sig', index=False)


def ler_csv(path):
    for encoding in ['utf-8-sig', 'cp1252', 'latin-1']:
        try:
            with path.open(encoding=encoding) as f:
                lines = [next(f, '') for _ in range(150)]
            break
        except UnicodeError:
            continue
    candidates = []
    for i, line in enumerate(lines):
        for sep in [';', '\t', ',', '|']:
            cells = next(csv.reader([line], delimiter=sep))
            norm = [normalizar(c) for c in cells]
            if len(cells) >= 2:
                score = 8 * sum(n in ['data', 'date', 'datetime', 'timestamp', 'datahora', 'datamedicao', 'time'] for n in norm)
                score += 4 * sum(bool(re.match(r'(vazao|chuva|precipitacao|nivel)\d{2}$', n)) for n in norm)
                score += min(len(cells), 10) / 10
                candidates.append((score, -i, sep, i))
    if not candidates:
        raise ValueError('Cabeçalho tabular não reconhecido. Conferir formato.')
    _, _, sep, skip = max(candidates)
    df = pd.read_csv(path, sep=sep, skiprows=skip, encoding=encoding,
                     dtype=str, low_memory=False)
    df.columns = [str(c).strip().strip('"') for c in df.columns]
    return df, f'encoding={encoding}; separador={repr(sep)}; linhas_pre_cabecalho={skip}'


def coluna_data(df):
    if COLUNA_DATA and COLUNA_DATA in df:
        return COLUNA_DATA
    choices = [c for c in df if normalizar(c) in
               ['datahora', 'datetime', 'timestamp', 'datamedicao', 'data', 'date', 'time']]
    if not choices:
        choices = [c for c in df if 'data' in normalizar(c) or 'date' in normalizar(c)]
    return next((c for c in choices if datas(df[c]).notna().any()), None)


def coluna_codigo(df):
    if COLUNA_CODIGO and COLUNA_CODIGO in df:
        return COLUNA_CODIGO
    return next((c for c in df if normalizar(c) in
                 ['codigoestacao', 'estacaocodigo', 'codestacao', 'codigoana',
                  'codigoepagri', 'stationid', 'stationcode', 'estacao', 'codigo']), None)


def intervalo_falhas(idx, inicio, fim):
    observed = pd.DatetimeIndex(idx).normalize().unique()
    calendar_idx = pd.date_range(inicio.normalize(), fim.normalize(), freq='D')
    absent = calendar_idx.difference(observed)
    if not len(absent):
        return 0, 0, []
    breaks = np.r_[True, np.diff(absent.asi8) != pd.Timedelta('1d').value]
    groups = np.cumsum(breaks)
    gaps = [(g.min(), g.max(), len(g)) for _, g in pd.Series(absent).groupby(groups)]
    return len(calendar_idx), len(absent), gaps


def resumo_serie(s, fonte, arquivo, codigo_estacao, variavel, leituras, diarios):
    s = s.sort_index()
    n_datas_invalidas = int(s.index.isna().sum())
    s = s.loc[s.index.notna()]
    if s.empty:
        return
    ini, fim = s.index.min(), s.index.max()
    valid = s.dropna()
    unique = s.index.unique().sort_values()
    steps = pd.Series(unique[1:] - unique[:-1])
    passo = steps.mode().iloc[0] if len(steps) else pd.NaT
    if pd.notna(passo) and passo > pd.Timedelta(0):
        nominal = int((fim - ini) / passo) + 1
    else:
        nominal = np.nan
    # Disponibilidade diária: pelo menos um valor naquele dia, NÃO dia completo.
    dias = pd.DatetimeIndex(valid.index).normalize().unique().sort_values()
    total_dias = len(pd.date_range(ini.normalize(), fim.normalize(), freq='D'))
    _, n_faltantes, gaps = intervalo_falhas(valid.index, ini, fim)
    key = f'{fonte}:{codigo_estacao or Path(arquivo).stem}:{variavel}'
    row = dict(fonte=fonte, arquivo=arquivo, codigo_estacao=codigo_estacao,
               variavel=variavel, inicio=str(ini), fim=str(fim), registros=len(s),
               valores_validos=len(valid), valores_ausentes=int(s.isna().sum()),
               datas_invalidas=n_datas_invalidas,
               timestamps_duplicados=int(s.index.duplicated(keep=False).sum()),
               passo_predominante=str(passo), registros_esperados_nominais=nominal,
               dias_no_periodo=total_dias, dias_com_algum_dado=len(dias),
               dias_sem_dado=n_faltantes, disponibilidade_diaria_pct=round(100 * len(dias) / total_dias, 2),
               maior_falha_dias=max([g[2] for g in gaps], default=0),
               negativos=int((valid < 0).sum()), zeros=int((valid == 0).sum()),
               minimo=float(valid.min()) if len(valid) else np.nan,
               mediana=float(valid.median()) if len(valid) else np.nan,
               maximo=float(valid.max()) if len(valid) else np.nan,
               prioridade=codigo_estacao in PRIORIDADES,
               unidade='não inferida: conferir cabeçalho/metadados')
    leituras.append(row)
    # Uma linha por arquivo/variável: sobreposições de arquivos não são fundidas.
    diarios.append((key + ':' + Path(arquivo).name, dias))


def analisar_tabela(df, fonte, arquivo, leituras, diarios, colunas):
    dc = coluna_data(df)
    cc = coluna_codigo(df)
    for c in df:
        vals = df[c].dropna().astype(str)
        colunas.append(dict(fonte=fonte, arquivo=arquivo, coluna=c,
                            nao_nulos=len(vals), exemplo=' / '.join(vals.head(3)),
                            papel='data' if c == dc else 'codigo' if c == cc else 'a_conferir'))
    if not dc:
        return 'Sem coluna temporal reconhecida: inventário de colunas produzido.'
    times = datas(df[dc])
    hc = next((c for c in df if normalizar(c) in ['hora', 'time', 'horario']), None)
    if hc and hc != dc and normalizar(dc) not in ['datahora', 'datetime', 'timestamp']:
        combined = datas(df[dc].astype(str).str.strip() + ' ' + df[hc].fillna('00:00:00').astype(str).str.strip())
        times = combined.fillna(times)
    filename_code = re.findall(r'(?<!\d)(\d{8}|\d{4})(?!\d)', Path(arquivo).stem)
    groups = df.groupby(cc, dropna=False, sort=False) if cc else [(filename_code[0] if filename_code else '', df)]
    for station, group in groups:
        monthly = {}
        for c in df:
            match = re.match(r'^(vazao|chuva|precipitacao|nivel)(\d{2})$', normalizar(c))
            if match:
                monthly.setdefault(match[1], []).append((int(match[2]), c))
        for var, daycols in monthly.items():
            # Hidroweb: uma linha por mês e Vazao01...Vazao31.
            # Níveis de consistência são reportados separadamente, sem escolher
            # silenciosamente entre dados brutos e consistidos.
            lc = next((c for c in df if normalizar(c) == 'nivelconsistencia'), None)
            subsets = group.groupby(lc, dropna=False, sort=False) if lc else [('não informado', group)]
            for level, subset in subsets:
                dates_out, values_out = [], []
                for rowidx, row in subset.iterrows():
                    base = times.loc[rowidx]
                    if pd.isna(base):
                        continue
                    for day, c in daycols:
                        if day <= calendar.monthrange(base.year, base.month)[1]:
                            dates_out.append(pd.Timestamp(base.year, base.month, day))
                            values_out.append(row[c])
                if dates_out:
                    vals = numeros(pd.Series(values_out)).to_numpy()
                    series = pd.Series(vals, index=pd.DatetimeIndex(dates_out))
                    resumo_serie(series, fonte, arquivo, codigo(station),
                                 f'{var} [consistencia={level}]', leituras, diarios)
        skip = {dc, cc, hc}
        skip.update(c for pairs in monthly.values() for _, c in pairs)
        for c in group:
            # Em Hidroweb mensal, VazaoMedia/VazaoMaxima são estatísticas
            # mensais, não observações diárias adicionais.
            if monthly:
                continue
            if c in skip:
                continue
            nc = normalizar(c)
            is_measurement = (nc in ['q', 'p', 'h', 'flow', 'flowrate', 'discharge', 'rainfall', 'waterlevel', 'mediadatemperaturadoaroc', 'mediadaumidaderelativa', 'molhamentofoliar'] or
                              any(nc.startswith(k) for k in ['vazao', 'chuva', 'precipitacao', 'nivel',
                                                            'temperatura', 'umidade', 'vento', 'radiacao']))
            if not is_measurement or nc in ['nivelconsistencia', 'flowrateconsistencylevel']:
                continue
            vals = numeros(group[c])
            if vals.notna().any():
                consistency_col = next((col for col in group if normalizar(col) in
                                        ['nivelconsistencia', 'flowrateconsistencylevel']), None)
                subsets = group.groupby(consistency_col, dropna=False, sort=False) if consistency_col else [(None, group)]
                for level, subset in subsets:
                    subset_vals = numeros(subset[c])
                    label = c if level is None else f'{c} [consistencia={level}]'
                    resumo_serie(pd.Series(subset_vals.to_numpy(), index=pd.DatetimeIndex(times.loc[subset.index])),
                                 fonte, arquivo, codigo(station), label, leituras, diarios)
    return 'Análise temporal concluída; variáveis e unidades precisam de conferência.'



# ================= DIAGNÓSTICO E EXPORTAÇÃO =================
PENDENCIAS, POSICOES, CATALOGO, DIAS = [], [], [], {}


def pendencia(arquivo, detalhe):
    PENDENCIAS.append({'arquivo': str(arquivo), 'detalhe': detalhe})


def salvar(df, path):
    pd.DataFrame(df).to_csv(path, sep=';', index=False, encoding='utf-8-sig')


def codigo_estacao(value):
    if pd.isna(value):
        return ''
    text = str(value).strip()
    # Extrai código no início de descrições como 83029900-Barragem Taió.
    match = re.match(r'^(\d+)(?:\.0)?(?:$|[\s\-_/])', text)
    return match.group(1) if match else (text if re.fullmatch(r'[A-Za-z0-9]+', text) else '')


def tipo_variavel(name):
    n = normalizar(name.split(' [')[0])
    if n in ['p', 'rainfall'] or n.startswith(('chuva', 'precipitacao')):
        return 'chuva'
    if n in ['q', 'flow', 'flowrate', 'discharge'] or n.startswith('vazao'):
        return 'vazao'
    if n in ['h', 'waterlevel'] or n.startswith('nivel'):
        return 'nivel'
    return 'outras'


def resolucao_temporal(index):
    times = pd.DatetimeIndex(index).dropna().unique().sort_values()
    deltas = pd.Series(times[1:] - times[:-1])
    if deltas.empty:
        return 'indeterminada', '', np.nan
    step = deltas.mode().iloc[0]
    fraction = float((deltas == step).mean())
    if step == pd.Timedelta('1h'):
        label = 'horaria'
    elif step == pd.Timedelta('1d'):
        label = 'diaria'
    elif step < pd.Timedelta('1h'):
        label = 'subhoraria'
    elif step < pd.Timedelta('1d'):
        label = 'subdiaria'
    else:
        label = 'outra'
    return label, str(step), round(100 * fraction, 2)


def coletar_posicoes(df, arquivo):
    cols = {normalizar(c): c for c in df}
    override = COLUNAS_POSICAO.get(Path(str(arquivo).split('::')[0]).name, {})
    def field(role, candidates):
        return override.get(role) or next((cols[k] for k in candidates if k in cols), None)
    cc = field('codigo', ['codigoestacao', 'estacaocodigo', 'codestacao', 'codigoana', 'codigoepagri', 'stationid', 'stationcode', 'codigo', 'estacao'])
    lat = field('latitude', ['latitude', 'lat'])
    lon = field('longitude', ['longitude', 'lon', 'long'])
    name = field('nome', ['nome', 'nomeestacao', 'estacaonome', 'name'])
    if not all([cc, lat, lon]):
        return
    for _, r in df[list(dict.fromkeys(c for c in [cc, lat, lon, name] if c))].drop_duplicates().iterrows():
        x, y = numeros(pd.Series([r[lon], r[lat]])).to_numpy()
        code = codigo_estacao(r[cc])
        if not code or not np.isfinite(x + y) or not (-180 <= x <= 180 and -90 <= y <= 90):
            pendencia(arquivo, 'Código ou coordenada inválidos em registro de posição.')
            continue
        POSICOES.append(dict(codigo=code, longitude=x, latitude=y,
                             nome=str(r[name]) if name and pd.notna(r[name]) else '', fonte_posicao=str(arquivo)))


# Captura as séries já interpretadas pelo leitor dos formatos ANA/EPAGRI.
_resumo_original = resumo_serie

def resumo_serie(s, fonte, arquivo, codigo_estacao_, variavel, leituras, diarios):
    code = codigo_estacao(codigo_estacao_)
    if not code:
        pendencia(arquivo, f'Série sem código reconhecido: {variavel}; não associada espacialmente.')
    start = len(leituras)
    _resumo_original(s, fonte, arquivo, code, variavel, leituras, diarios)
    if len(leituras) == start:
        pendencia(arquivo, f'Série sem datas válidas: {variavel}')
        return
    row = dict(leituras[-1])
    full = s[s.index.notna()].sort_index()
    resolution, step, regularity = resolucao_temporal(full.index)
    sid = hashlib.sha256(json.dumps([fonte, arquivo, code, variavel], ensure_ascii=False).encode()).hexdigest()[:16]
    unit = re.search(r'\(([^)]+)\)', variavel)
    row['unidade'] = unit.group(1) if unit else 'não informada: conferir metadados'
    row.update(serie_id=sid, tipo=tipo_variavel(variavel), resolucao=resolution,
               passo_predominante=step, intervalos_no_passo_pct=regularity)
    valid_times = full.dropna().index.unique()
    row['timestamps_validos_unicos'] = len(valid_times)
    expected = row['registros_esperados_nominais']
    row['cobertura_nominal_pct'] = round(100 * len(valid_times) / expected, 2) if pd.notna(expected) and expected else np.nan
    if row['cobertura_nominal_pct'] > 100:
        pendencia(arquivo, f'Série {code}/{variavel}: cobertura nominal >100%; conferir passo irregular e timestamps.')
    row['dias_24_horas_observadas'] = np.nan
    if resolution == 'horaria':
        hours = valid_times.floor('h').unique()
        per_day = pd.Series(1, index=hours).groupby(hours.normalize()).sum()
        row['dias_24_horas_observadas'] = int((per_day == 24).sum())
    row['maior_intervalo_sem_valor_horas'] = np.nan
    if len(valid_times) and step:
        delta = pd.Timedelta(step)
        if delta > pd.Timedelta(0):
            padded = valid_times.sort_values()
            gaps = list(padded[1:] - padded[:-1] - delta)
            gaps += [padded[0] - full.index.min(), full.index.max() - padded[-1]]
            row['maior_intervalo_sem_valor_horas'] = max(0, max(gaps, default=pd.Timedelta(0)).total_seconds() / 3600)
    DIAS[sid] = valid_times.normalize().unique().sort_values()
    if PERIODO_ANALISE:
        a, b = (pd.Timestamp(x).normalize() for x in PERIODO_ANALISE)
        selected_days = DIAS[sid][(DIAS[sid] >= a) & (DIAS[sid] <= b)]
        row['disponibilidade_periodo_comum_pct'] = round(100 * len(selected_days) / len(pd.date_range(a, b)), 2)
        row['periodo_comum_inicio'], row['periodo_comum_fim'] = str(a.date()), str(b.date())
    # Não agregar nem remover duplicatas: manter diagnóstico da resolução nativa.
    out = SAIDA / 'catalogo_dados' / row['tipo'] / resolution / (code or 'sem_codigo')
    out.mkdir(parents=True, exist_ok=True)
    target = out / f'{sid}.csv'
    data = pd.DataFrame({'data_hora': full.index, 'valor': full.to_numpy()})
    data['codigo_estacao'] = code
    data['fonte'] = fonte
    data['variavel'] = variavel
    data['resolucao'] = resolution
    salvar(data, target)
    row['arquivo_separado'] = str(target.relative_to(SAIDA))
    CATALOGO.append(row)


def ler_bases():
    for fonte, directory in DIRETORIOS.items():
        if not directory.exists():
            pendencia(directory, 'Diretório não encontrado.')
            continue
        for path in sorted(directory.rglob('*')):
            if not path.is_file():
                continue
            ext = path.suffix.lower()
            if ext not in ['.csv', '.txt', '.dat', '.xls', '.xlsx']:
                if ext in ['.mdb', '.accdb', '.zip', '.nc', '.parquet']:
                    pendencia(path, 'Formato não lido nesta versão; não contabilizado como série disponível.')
                continue
            print(f'Lendo {fonte}: {path.name}', flush=True)
            try:
                if ext in ['.xls', '.xlsx']:
                    with pd.ExcelFile(path) as book:
                        tables = [(str(path) + '::' + sheet, pd.read_excel(book, sheet_name=sheet, dtype=str)) for sheet in book.sheet_names]
                else:
                    tables = [(str(path), ler_csv(path)[0])]
                for origin, df in tables:
                    coletar_posicoes(df, origin)
                    previous = len(CATALOGO)
                    analisar_tabela(df, fonte, origin, [], [], [])
                    if len(CATALOGO) == previous:
                        pendencia(origin, 'Nenhuma série temporal numérica reconhecida; conferir colunas e formato.')
            except Exception as e:
                pendencia(path, f'{type(e).__name__}: {e}')


def somente_poligonos(g):
    from shapely.ops import unary_union
    if g.geom_type in ['Polygon', 'MultiPolygon']:
        return g
    if hasattr(g, 'geoms'):
        parts = [somente_poligonos(p) for p in g.geoms]
        parts = [p for p in parts if p is not None and not p.is_empty]
        return unary_union(parts) if parts else None
    return None


def area_geodesica(g):
    from pyproj import Geod
    from shapely.geometry import MultiPolygon
    from shapely.geometry.polygon import orient
    oriented = orient(g) if g.geom_type == 'Polygon' else MultiPolygon([orient(p) for p in g.geoms])
    return abs(Geod(ellps='WGS84').geometry_area_perimeter(oriented)[0]) / 1e6


def ler_sig():
    import geopandas as gpd
    from pyproj import CRS
    from shapely import make_valid
    crs = CRS.from_user_input(CRS_METRICO)
    if not crs.is_projected or any(abs(a.unit_conversion_factor - 1) > 1e-9 for a in crs.axis_info):
        raise ValueError('CRS_METRICO deve ser projetado e ter unidades em metros.')
    raw = gpd.read_file(ARQUIVO_BACIAS, **({'layer': CAMADA_BACIAS} if CAMADA_BACIAS else {}))
    if raw.crs is None:
        raise ValueError('Bacias sem CRS. Conferir o arquivo .prj; não assumir coordenadas.')
    raw = raw.to_crs(4326)
    cols = {normalizar(c): c for c in raw if c != raw.geometry.name}
    outlet = CAMPO_EXUTORIO or next((cols[k] for k in ['estacao', 'codigo', 'codigoestacao', 'exutorio', 'stationid'] if k in cols), None)
    basins = []
    for pos, (_, r) in enumerate(raw.iterrows()):
        geom = r.geometry
        if geom is None or geom.is_empty:
            pendencia(ARQUIVO_BACIAS, f'Feição {pos}: geometria vazia.')
            continue
        repair = not geom.is_valid
        if repair:
            geom = somente_poligonos(make_valid(geom))
            pendencia(ARQUIVO_BACIAS, f'Feição {pos}: geometria reparada em memória; conferir mapa e área.')
        if geom is None or geom.is_empty or geom.geom_type not in ['Polygon', 'MultiPolygon'] or not geom.is_valid:
            pendencia(ARQUIVO_BACIAS, f'Feição {pos}: não é um polígono válido; excluída.')
            continue
        basins.append(dict(bacia_id=f'bacia_{pos:02d}', exutorio=codigo_estacao(r[outlet]) if outlet else '',
                           area_km2=area_geodesica(geom), geometria_reparada=repair, geometry=geom))
    if not basins:
        raise ValueError('Nenhuma bacia válida encontrada.')
    paths = ARQUIVOS_ESTACOES if ARQUIVOS_ESTACOES is not None else sorted(SIG.rglob('*.shp')) + sorted(SIG.rglob('*.gpkg')) + sorted(SIG.rglob('*.geojson'))
    code_names = ['codigo', 'codigoana', 'codigoepagri', 'codestacao', 'codigoestacao', 'estacaocodigo', 'stationid', 'stationcode', 'codana', 'codepagri', 'cdestacao', 'estacao']
    for path in map(Path, paths):
        if path.resolve() == ARQUIVO_BACIAS.resolve():
            continue
        try:
            layers = list(gpd.list_layers(path)['name']) if path.suffix.lower() == '.gpkg' else [None]
            for layer in layers:
                pts = gpd.read_file(path, **({'layer': layer} if layer else {}))
                if not pts.geom_type.eq('Point').any():
                    continue
                if pts.crs is None:
                    pendencia(path, 'Estações sem CRS; camada não associada.')
                    continue
                pts = pts.to_crs(4326)
                cs = [c for c in pts if normalizar(c) in code_names]
                nc = next((c for c in pts if normalizar(c) in ['nome', 'name', 'nomeestacao', 'estacaonome']), None)
                if not cs:
                    pendencia(path, 'Sem campo de código reconhecido.')
                for _, r in pts.iterrows():
                    if r.geometry is None or r.geometry.is_empty or r.geometry.geom_type != 'Point':
                        continue
                    for code in sorted({codigo_estacao(r[c]) for c in cs} - {''}):
                        POSICOES.append(dict(codigo=code, longitude=r.geometry.x, latitude=r.geometry.y,
                                             nome=str(r[nc]) if nc and pd.notna(r[nc]) else '', fonte_posicao=f'{path}::{layer or ""}'))
        except Exception as e:
            pendencia(path, f'{type(e).__name__}: {e}')
    return sorted(basins, key=lambda b: -b['area_km2'])


def consolidar_posicoes():
    if not POSICOES:
        raise ValueError('Nenhuma estação com coordenadas reconhecida. Conferir SIG e colunas de posição.')
    positions = pd.DataFrame(POSICOES)
    # Não juntar posições aproximadas: uma estação pode ter sido deslocada.
    keys = ['codigo', 'longitude', 'latitude']
    positions = positions.groupby(keys, as_index=False, dropna=False).agg(
        nome=('nome', lambda s: ' / '.join(sorted(set(s) - {''}))),
        fonte_posicao=('fonte_posicao', lambda s: ' / '.join(sorted(set(s)))))
    positions['posicao_id'] = [f'pos_{i:05d}' for i in range(len(positions))]
    counts = positions.groupby('codigo').size()
    positions['numero_posicoes_codigo'] = positions.codigo.map(counts)
    for code, count in counts.items():
        if count > 1:
            pendencia(code, f'{count} posições distintas: vínculo temporal das coordenadas desconhecido. Conferir histórico da estação.')
    return positions


def relacao_espacial(point, polygon, distance_km):
    inside = polygon.covers(point)
    edge = point.distance(polygon.boundary) / 1000
    return ('interna' if inside else 'externa_proxima' if edge <= distance_km else 'fora'), edge


def tabela_html(df):
    if df.empty:
        return '<p>Nenhum registro identificado.</p>'
    return df.to_html(index=False, escape=True, na_rep='—', float_format=lambda x: f'{x:.2f}')


def documento(titulo, conteudo):
    return ('<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>' + html.escape(titulo) + '</title>'
            '<style>body{font:15px Arial;max-width:1450px;margin:32px auto;padding:0 20px;color:#203746}'
            'h1,h2{color:#164b66}table{border-collapse:collapse;font-size:12px;display:block;overflow:auto}'
            'td,th{border:1px solid #ccd7de;padding:7px}th{background:#edf3f6}img{max-width:100%}'
            'li{margin:8px 0}a{color:#006b91}</style><h1>' + html.escape(titulo) + '</h1>' + conteudo + '</html>')


def gantt(catalog, folder):
    images = []
    for start in range(0, len(catalog), MAX_SERIES_GANTT):
        group = catalog.iloc[start:start + MAX_SERIES_GANTT]
        fig, ax = plt.subplots(figsize=(14, max(4, len(group) * .34 + 1.5)))
        labels = []
        for j, (_, r) in enumerate(group.iterrows()):
            labels.append(f'{r.codigo_estacao} | {r.tipo} | {r.fonte} | {r.resolucao} | {r.variavel}')
            days = DIAS[r.serie_id]
            if len(days):
                breaks = np.r_[True, np.diff(days.asi8) != pd.Timedelta('1d').value]
                for _, seg in pd.Series(days).groupby(np.cumsum(breaks)):
                    ax.broken_barh([(mdates.date2num(seg.min()), (seg.max() - seg.min()).days + 1)], (j - .3, .6), facecolors='#16819d')
        ax.set_yticks(range(len(group)), labels, fontsize=7)
        ax.xaxis_date()
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.invert_yaxis()
        ax.set_title('Disponibilidade: dias com pelo menos uma observação válida')
        ax.grid(axis='x', alpha=.2)
        fig.tight_layout()
        name = f'disponibilidade_{start // MAX_SERIES_GANTT + 1:02d}.png'
        fig.savefig(folder / name, dpi=160)
        plt.close(fig)
        images.append(name)
    return images


def main():
    import geopandas as gpd
    if DISTANCIA_BORDA_KM < 0 or not np.isfinite(DISTANCIA_BORDA_KM):
        raise ValueError('DISTANCIA_BORDA_KM deve ser finita e não negativa.')
    if PERIODO_ANALISE and pd.Timestamp(PERIODO_ANALISE[1]) < pd.Timestamp(PERIODO_ANALISE[0]):
        raise ValueError('PERIODO_ANALISE: fim anterior ao início.')
    SAIDA.mkdir(parents=True, exist_ok=False)
    basins = ler_sig()
    ler_bases()
    positions = consolidar_posicoes()
    catalog = pd.DataFrame(CATALOGO)
    if catalog.empty:
        salvar(PENDENCIAS, SAIDA / 'pendencias.csv')
        raise ValueError('Nenhuma série lida. Consultar pendencias.csv.')
    catalog['numero_posicoes_codigo'] = catalog.codigo_estacao.map(positions.groupby('codigo').size()).fillna(0).astype(int)
    for _, r in catalog[catalog.numero_posicoes_codigo.eq(0)].iterrows():
        pendencia(r.arquivo, f'Série {r.codigo_estacao}/{r.variavel}: sem coordenada; mantida no catálogo geral.')
    salvar(catalog, SAIDA / 'catalogo_series.csv')
    salvar(positions, SAIDA / 'posicoes_estacoes.csv')
    # Coordenadas vinculadas por código, sem atribuir uma posição histórica a todas as datas.
    for _, r in catalog.iterrows():
        metadata = r.to_dict()
        metadata['posicoes'] = positions[positions.codigo.eq(r.codigo_estacao)].to_dict('records')
        metadata['equivalencias_documentadas'] = [v for v in EQUIVALENCIAS.values() if r.codigo_estacao in v]
        (SAIDA / r.arquivo_separado).with_suffix('.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    points = gpd.GeoDataFrame(positions, geometry=gpd.points_from_xy(positions.longitude, positions.latitude), crs=4326).to_crs(CRS_METRICO)
    summaries, all_members = [], []
    for b in basins:
        folder = SAIDA / b['bacia_id']
        folder.mkdir()
        polygon = gpd.GeoSeries([b['geometry']], crs=4326).to_crs(CRS_METRICO).iloc[0]
        members = []
        for _, p in points.iterrows():
            relation, edge = relacao_espacial(p.geometry, polygon, DISTANCIA_BORDA_KM)
            if relation == 'fora':
                continue
            members.append({**{k: p[k] for k in positions.columns}, 'relacao': relation, 'distancia_borda_km': edge, 'bacia_id': b['bacia_id']})
        m = pd.DataFrame(members, columns=list(positions.columns) + ['relacao', 'distancia_borda_km', 'bacia_id'])
        all_members.extend(m.to_dict('records'))
        inside = set(m.loc[m.relacao.eq('interna'), 'codigo'])
        outside = set(m.loc[m.relacao.eq('externa_proxima'), 'codigo']) - inside
        selected = catalog[catalog.codigo_estacao.isin(inside | outside)].copy()
        selected['relacao_codigo'] = np.where(selected.codigo_estacao.isin(inside), 'interna', 'externa_proxima')
        selected['posicoes_em_ambas_relacoes'] = selected.codigo_estacao.isin(inside & set(m.loc[m.relacao.eq('externa_proxima'), 'codigo']))
        selected['arquivo_por_bacia'] = ''
        if EXPORTAR_DADOS_POR_BACIA:
            for index, r in selected.iterrows():
                relative = Path('dados') / r.tipo / r.resolucao / r.codigo_estacao / Path(r.arquivo_separado).name
                target = folder / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(SAIDA / r.arquivo_separado, target)
                shutil.copy2((SAIDA / r.arquivo_separado).with_suffix('.json'), target.with_suffix('.json'))
                selected.loc[index, 'arquivo_por_bacia'] = str(relative)
        salvar(m, folder / 'estacoes_posicoes.csv')
        salvar(selected, folder / 'series.csv')
        gpd.GeoDataFrame([{'bacia_id': b['bacia_id'], 'exutorio': b['exutorio'], 'area_km2': b['area_km2'], 'geometry': b['geometry']}], crs=4326).to_file(folder / 'bacia.gpkg', driver='GPKG')
        if not m.empty:
            gpd.GeoDataFrame(m, geometry=gpd.points_from_xy(m.longitude, m.latitude), crs=4326).to_file(folder / 'estacoes.gpkg', driver='GPKG')
        summary = {k: b[k] for k in ['bacia_id', 'exutorio', 'area_km2', 'geometria_reparada']}
        summary.update(codigos_internos=len(inside), codigos_externos_proximos=len(outside), posicoes=len(m), series=len(selected))
        for typ in ['chuva', 'vazao', 'nivel']:
            for rel in ['interna', 'externa_proxima']:
                subset = selected[selected.tipo.eq(typ) & selected.relacao_codigo.eq(rel)]
                summary[f'{typ}_{rel}_codigos_com_dados'] = subset.loc[subset.valores_validos.gt(0), 'codigo_estacao'].nunique()
        summary['vazao_exutorio_identificada'] = bool((selected.codigo_estacao.eq(b['exutorio']) & selected.tipo.eq('vazao') & selected.valores_validos.gt(0)).any())
        summaries.append(summary)
        fig, ax = plt.subplots(figsize=(10, 8))
        gpd.GeoSeries([polygon.buffer(DISTANCIA_BORDA_KM * 1000)], crs=CRS_METRICO).boundary.plot(ax=ax, color='#a2adb3', linestyle='--')
        gpd.GeoSeries([polygon], crs=CRS_METRICO).boundary.plot(ax=ax, color='#164b66', linewidth=1.6)
        for rel, color in [('interna', '#007f9c'), ('externa_proxima', '#c57b17')]:
            subset = m[m.relacao.eq(rel)]
            if not subset.empty:
                xy = gpd.GeoSeries(gpd.points_from_xy(subset.longitude, subset.latitude), crs=4326).to_crs(CRS_METRICO)
                ax.scatter(xy.x, xy.y, c=color, s=22, label=rel)
                if len(m) <= 60:
                    for geom, code in zip(xy, subset.codigo):
                        ax.annotate(code, (geom.x, geom.y), xytext=(3, 3), textcoords='offset points', fontsize=6)
        ax.set_title(f'{b["bacia_id"]} | exutório {b["exutorio"]} | {b["area_km2"]:.1f} km²')
        ax.set_xlabel('E (m)'); ax.set_ylabel('N (m)'); ax.set_aspect('equal')
        if not m.empty:
            ax.legend()
        fig.tight_layout(); fig.savefig(folder / 'mapa.png', dpi=180); plt.close(fig)
        images = gantt(selected, folder)
        viewcols = ['codigo_estacao', 'fonte', 'tipo', 'variavel', 'resolucao', 'relacao_codigo', 'inicio', 'fim', 'disponibilidade_diaria_pct', 'cobertura_nominal_pct', 'intervalos_no_passo_pct', 'dias_24_horas_observadas', 'maior_falha_dias', 'maior_intervalo_sem_valor_horas', 'timestamps_duplicados', 'numero_posicoes_codigo', 'unidade', 'negativos', 'zeros', 'minimo', 'mediana', 'maximo']
        if PERIODO_ANALISE:
            viewcols.append('disponibilidade_periodo_comum_pct')
        body = f'<p>Área: {b["area_km2"]:.2f} km². Faixa externa: {DISTANCIA_BORDA_KM:g} km. Geometria reparada: {b["geometria_reparada"]}.</p>'
        body += '<p><a href="../relatorio_geral.html">Comparação entre bacias</a></p><img src="mapa.png" alt="Mapa de bacia e estações">'
        body += '<h2>Critérios para escolher a bacia</h2><ul><li>Conferir limites, exutório, coordenadas e eventuais reparos no mapa.</li><li>Verificar vazão na estação alvo e disponibilidade de chuva no mesmo período.</li><li>Avaliar falhas longas, resolução e duplicatas antes de agregar ou preencher.</li><li>Estações externas são candidatas auxiliares; proximidade não garante relação hidrológica.</li></ul>'
        body += '<h2>Resumo</h2>' + tabela_html(pd.DataFrame([summary]))
        body += '<h2>Séries disponíveis</h2>' + tabela_html(selected[viewcols])
        body += '<h2>Coordenadas preservadas</h2>' + tabela_html(m)
        body += '<h2>Disponibilidade temporal</h2>' + ''.join(f'<img src="{im}" alt="Disponibilidade temporal">' for im in images)
        body += notas_html()
        (folder / 'relatorio.html').write_text(documento(b['bacia_id'], body), encoding='utf-8')
        print(f'Relatório concluído: {b["bacia_id"]}', flush=True)
    comparison = pd.DataFrame(summaries)
    salvar(comparison, SAIDA / 'resumo_bacias.csv')
    salvar(all_members, SAIDA / 'estacoes_por_bacia.csv')
    salvar(PENDENCIAS, SAIDA / 'pendencias.csv')
    body = f'<p>Diagnóstico gerado em {datetime.now():%d/%m/%Y %H:%M}. Faixa externa: {DISTANCIA_BORDA_KM:g} km.</p>'
    body += '<h2>Comparação</h2>' + tabela_html(comparison)
    body += '<h2>Relatórios por bacia</h2><ul>' + ''.join(f'<li><a href="{b["bacia_id"]}/relatorio.html">{b["bacia_id"]} — exutório {html.escape(b["exutorio"])} — {b["area_km2"]:.1f} km²</a></li>' for b in basins) + '</ul>'
    body += notas_html() + '<h2>Pendências</h2>' + tabela_html(pd.DataFrame(PENDENCIAS))
    (SAIDA / 'relatorio_geral.html').write_text(documento('Diagnóstico geral EPAGRI', body), encoding='utf-8')
    (SAIDA / 'configuracao.json').write_text(json.dumps({'base': str(BASE), 'arquivo_bacias': str(ARQUIVO_BACIAS), 'distancia_borda_km': DISTANCIA_BORDA_KM, 'crs_metrico': CRS_METRICO, 'periodo_analise': PERIODO_ANALISE, 'equivalencias': EQUIVALENCIAS, 'sentinelas': SENTINELAS}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nAbrir no navegador: {SAIDA / "relatorio_geral.html"}')


def notas_html():
    return ('<h2>Como interpretar</h2><ul>'
            '<li>Contagens são por códigos únicos, não por estações físicas. Equivalências documentadas não fundem séries.</li>'
            '<li>Cada série mantém arquivo, variável, fonte e nível de consistência. Não há soma de arquivos sobrepostos.</li>'
            '<li>Disponibilidade diária: pelo menos um valor no dia, não um dia completo. A resolução é inferida dos timestamps, não assegurada pelo rótulo.</li>'
            '<li>Cobertura nominal: timestamps válidos únicos / grade esperada no passo predominante. Série irregular exige conferência. Dias com 24 horas observadas contam horas distintas, sem assegurar a validade hidrológica dos valores.</li>'
            '<li>Cada série usa seu próprio período; percentuais não são diretamente comparáveis. PERIODO_ANALISE permite comparar dias no mesmo intervalo.</li>'
            '<li>Coordenadas alternativas são preservadas. Sem datas de mudança, não se pode atribuir uma posição específica a toda a série.</li>'
            '<li>Dados separados preservam a resolução original. Sentinelas configuradas viram valores ausentes; negativos e duplicatas permanecem sinalizados.</li>'
            '<li>Não há preenchimento nem conversão de nível em vazão. Unidades sem cabeçalho explícito precisam de metadados.</li>'
            '<li>Dados sem coordenadas permanecem no catálogo geral; formatos não lidos e erros aparecem em pendencias.csv.</li>'
            '<li>Os resultados permanecem locais; observar as condições de uso das bases EPAGRI.</li></ul>')


if __name__ == '__main__':
    main()
