# -*- coding: utf-8 -*-
"""ERA5-Land: P e evaporação/ET, pixel de cada estação, horário e diário.
Executar por células no Spyder, no ambiente epagri2026_chirps.
Instalação (Anaconda Prompt):
conda install -n epagri2026_chirps -c conda-forge geopandas pandas numpy xarray netcdf4 cdsapi tzdata
Credenciais: ~/.cdsapirc já configurado; aceitar a licença do dataset no CDS.
https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land
https://confluence.ecmwf.int/spaces/CKB/pages/140385202/ERA5-Land+data+documentation
Os CSVs são produtos auxiliares para substituição de lacunas: observações
não são modificadas por este script. Produtos brutos não têm correção de viés.
"""

# %% 1. Importar bibliotecas
from pathlib import Path
import hashlib
import json
import re
import tempfile
import zipfile
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import cdsapi

# %% 2. Informar arquivos, período e convenções das estações
DIRETORIO_AULA = Path(r'C:\Users\afrod\OneDrive\Documents\HydroUAI_git\hydrouai-epagri-2026\02_Dados\Dia 1')
ARQUIVO_ESTACOES = DIRETORIO_AULA / 'SIG' / 'estacoes_internas.gpkg'
CAMADA_ESTACOES = None  # None: primeira camada; informar se houver mais de uma.
CAMPO_CODIGO = 'codigo'  # Ajustar ao campo do GPKG, por exemplo 'codigo'.
CODIGOS_ESTACOES = None  # None = todas; exemplo: ['2463', '84100000'].
DIRETORIO_SAIDA = DIRETORIO_AULA / 'Dados' / 'ERA5Land'
DATA_INICIO = '2015-01-01'
DATA_FIM = '2024-12-31'  # Inclusivo: dias completos no fuso abaixo.
# Usar o MESMO fuso dos dados observados. Etc/GMT+3 = UTC-3 fixo, sem horário de verão.
# 'UTC' para UTC; 'America/Sao_Paulo' para hora civil histórica (com horário de verão).
FUSO_ESTACOES = 'Etc/GMT+3'
# Inicio: 10:00 representa 10:00–11:00; fim: 11:00 representa 10:00–11:00.
SALVAR_DATAS_COM_FUSO = False  # False: data local sem offset, compatível com a aula 1.
REFERENCIA_HORA = 'inicio'  # 'inicio' ou 'fim', conforme os registros observados.
TIPO_ET = 'potencial'  # 'potencial' (pev) ou 'real' (total_evaporation).
# EP é evaporação potencial de água livre do ERA5-Land, não ETo FAO-56.
# Na saída usa-se _EP para potencial e _ET para real, evitando confundi-las.
MIN_FRACAO_HORAS_DIA = 1.0  # 1.0 exige dia completo; dias de 23/25h tratados pelo fuso.
MARGEM_GRAUS = 0.2
TOLERANCIA_COORDENADAS_M = 10  # Mesma estação: aceitar pequenas diferenças entre fontes.
REFAZER_DOWNLOAD = False

# %% 3. Definir funções de download e leitura

def codigo_texto(valor):
    if pd.isna(valor):
        raise ValueError('Estação sem código.')
    codigo = str(valor).strip()
    if re.fullmatch(r'\d+\.0', codigo):
        codigo = codigo[:-2]
    if not re.fullmatch(r'[A-Za-z0-9_-]+', codigo):
        raise ValueError(f'Código inválido para nome de arquivo: {codigo!r}')
    return codigo


def normalizar_dataset(ds):
    renomear = {nome: destino for nome, destino in
                [('valid_time', 'time'), ('lat', 'latitude'), ('lon', 'longitude')]
                if nome in ds.dims and destino not in ds.dims}
    ds = ds.rename(renomear)
    aliases = {'P': ['tp', 'total_precipitation'],
               'EP' if TIPO_ET == 'potencial' else 'ET':
               ['pev', 'potential_evaporation'] if TIPO_ET == 'potencial' else ['e', 'total_evaporation']}
    encontrados = {}
    for destino, nomes in aliases.items():
        presentes = [n for n in nomes if n in ds.data_vars]
        if len(presentes) > 1:
            raise ValueError('Variáveis duplicadas no NetCDF.')
        if not presentes:
            continue  # ZIP pode conter uma variável em cada NetCDF.
        da = ds[presentes[0]]
        unidade = str(da.attrs.get('units', '')).lower().strip()
        if unidade not in ['m', 'm of water equivalent']:
            raise ValueError(f'Unidade inesperada: {unidade}; esperado metros.')
        for dim in list(da.dims):
            if dim not in ['time', 'latitude', 'longitude']:
                if da.sizes[dim] != 1:
                    raise ValueError(f'Dimensão adicional {dim}: selecionar a versão explicitamente.')
                da = da.isel({dim: 0}, drop=True)
        if set(da.dims) != {'time', 'latitude', 'longitude'}:
            raise ValueError('Grade/tempo incompatíveis com a extração.')
        encontrados[destino] = da
    if not encontrados:
        raise ValueError('NetCDF sem P ou variável de ET solicitada.')
    ds = xr.Dataset(encontrados)
    ds = ds.assign_coords(longitude=(ds.longitude + 180) % 360 - 180)
    return ds.sortby('longitude').sortby('latitude').sortby('time')


def extrair_arquivo(arquivo, estacoes):
    """Carregar apenas séries pontuais em memória; suportar NetCDF ou ZIP."""
    def extrair_nc(caminho):
        with xr.open_dataset(caminho) as original:
            ds = normalizar_dataset(original)
            blocos, pixels = {}, []
            for _, est in estacoes.iterrows():
                x, y = est.geometry.x, est.geometry.y
                # Pixel da grade regular cujo centro é mais próximo, sem interpolação.
                if not (float(ds.longitude.min()) <= x <= float(ds.longitude.max())
                        and float(ds.latitude.min()) <= y <= float(ds.latitude.max())):
                    raise ValueError(f'Estação {est.codigo} fora do recorte baixado.')
                ponto = ds.sel(longitude=x, latitude=y, method='nearest')
                tempo = pd.DatetimeIndex(ponto.time.values)
                if tempo.has_duplicates:
                    raise ValueError('NetCDF com tempos duplicados.')
                blocos[est.codigo] = pd.DataFrame({n: ponto[n].to_numpy() for n in ds.data_vars}, index=tempo)
                pixels.append({'codigo_estacao': est.codigo, 'longitude': x, 'latitude': y,
                               'longitude_pixel': float(ponto.longitude), 'latitude_pixel': float(ponto.latitude)})
            return blocos, pixels
    if not zipfile.is_zipfile(arquivo):
        return extrair_nc(arquivo)
    with tempfile.TemporaryDirectory() as tmp:
        partes, pixels = {}, []
        with zipfile.ZipFile(arquivo) as pacote:
            membros = [n for n in pacote.namelist() if n.lower().endswith('.nc')]
            if not membros:
                raise ValueError('ZIP sem NetCDF.')
            for i, membro in enumerate(membros):
                destino = Path(tmp) / f'parte_{i}.nc'
                destino.write_bytes(pacote.read(membro))
                atual, pix = extrair_nc(destino)
                pixels.extend(pix)
                for codigo, dados in atual.items():
                    partes.setdefault(codigo, []).append(dados)
        resultado = {}
        for codigo, tabelas in partes.items():
            merged = pd.concat(tabelas, axis=1)
            if merged.columns.duplicated().any():
                raise ValueError('ZIP com variável repetida; conferir arquivos CDS.')
            resultado[codigo] = merged
        return resultado, pixels


def desacumular(acumulados):
    """CDS original: 01UTC reinicia; 02..00UTC = diferença consecutiva.
    Valores são acumulados desde 00UTC; 00UTC encerra o dia anterior.
    O índice de saída aqui ainda é o FIM do intervalo horário em UTC.
    """
    grade = pd.date_range(acumulados.index.min(), acumulados.index.max(), freq='h')
    a = acumulados.reindex(grade).astype(float)
    horarios = a.diff()
    reinicio = a.index.hour == 1
    horarios.loc[reinicio] = a.loc[reinicio]
    horarios['P'] *= 1000
    # Pequenos negativos por arredondamento -> zero; negativos materiais -> ausência.
    tolerancia = 0.001  # mm por hora
    horarios.loc[horarios.P.between(-tolerancia, 0), 'P'] = 0.0
    horarios.loc[horarios.P < -tolerancia, 'P'] = np.nan
    et = 'EP' if TIPO_ET == 'potencial' else 'ET'
    horarios[et] *= -1000  # Fluxo ascendente é negativo no ECMWF.
    # ET negativa (condensação/fluxo descendente) permanece: não usar abs().
    return horarios.replace([np.inf, -np.inf], np.nan)


def agregar_diarios(horario_inicio):
    """Agrupar pelo início de cada intervalo, no fuso da estação."""
    dias = horario_inicio.index.tz_localize(None).normalize()
    soma = horario_inicio.groupby(dias).sum(min_count=1)
    contagem = horario_inicio.groupby(dias).count()
    esperadas = pd.Series(1, index=horario_inicio.index).groupby(dias).sum()
    limites = np.ceil(esperadas * MIN_FRACAO_HORAS_DIA)
    return soma.where(contagem.ge(limites, axis=0)), contagem, esperadas


def indice_exportacao(indice):
    if indice.tz is not None and not SALVAR_DATAS_COM_FUSO:
        indice = indice.tz_localize(None)
        if indice.has_duplicates:
            raise ValueError('Hora local repetida no horário de verão: usar SALVAR_DATAS_COM_FUSO=True ou UTC-3 fixo.')
    return pd.Index([t.isoformat() for t in indice], name='data_hora')

# %% 4. Ler o GPKG e selecionar estações
if TOLERANCIA_COORDENADAS_M < 0:
    raise ValueError('TOLERANCIA_COORDENADAS_M não pode ser negativa.')
if TIPO_ET not in ['potencial', 'real'] or REFERENCIA_HORA not in ['inicio', 'fim']:
    raise ValueError('Conferir TIPO_ET e REFERENCIA_HORA.')
if not 0 < MIN_FRACAO_HORAS_DIA <= 1:
    raise ValueError('MIN_FRACAO_HORAS_DIA deve estar entre 0 e 1.')
estacoes = gpd.read_file(ARQUIVO_ESTACOES, **({'layer': CAMADA_ESTACOES} if CAMADA_ESTACOES else {}))
if CAMPO_CODIGO not in estacoes:
    raise ValueError(f'Campo {CAMPO_CODIGO!r} ausente. Campos: {list(estacoes.columns)}')
if estacoes.crs is None:
    raise ValueError('GPKG sem CRS; corrigir o arquivo antes de extrair.')
estacoes = estacoes.to_crs(4326)
if estacoes.geometry.isna().any() or estacoes.geometry.is_empty.any() or not estacoes.geom_type.eq('Point').all():
    raise ValueError('Todas as estações devem ter geometria Point válida.')
estacoes['codigo'] = estacoes[CAMPO_CODIGO].map(codigo_texto)
if CODIGOS_ESTACOES is not None:
    selecionados = {codigo_texto(c) for c in CODIGOS_ESTACOES}
    ausentes = selecionados - set(estacoes.codigo)
    if ausentes:
        raise ValueError(f'Códigos ausentes no GPKG: {sorted(ausentes)}')
    estacoes = estacoes[estacoes.codigo.isin(selecionados)].copy()
# Comparar distâncias geodésicas em metros, preservando os pontos originais.
# Aceitar apenas registros da mesma estação praticamente coincidentes.
from pyproj import Geod
geodesia = Geod(ellps='GRS80')
for codigo, grupo in estacoes.groupby('codigo'):
    pontos = list(grupo.geometry)
    distancia_maxima = max(
        abs(geodesia.inv(a.x, a.y, b.x, b.y)[2])
        for a in pontos for b in pontos
    )
    if distancia_maxima > TOLERANCIA_COORDENADAS_M:
        raise ValueError(
            f'{codigo}: coordenadas separadas por {distancia_maxima:.1f} m. '
            'Selecionar o ponto correto no GPKG.'
        )
    if len(grupo) > 1:
        print(f'{codigo}: {len(grupo)} registros coincidentes '
              f'(distância máxima {distancia_maxima:.3f} m); manter o primeiro ponto.')
estacoes = estacoes.drop_duplicates('codigo').copy()
if estacoes.empty:
    raise ValueError('Nenhuma estação selecionada.')
print(estacoes[['codigo', 'geometry']].to_string(index=False))

# %% 5. Preparar período, recorte espacial e pastas
inicio_local = pd.Timestamp(DATA_INICIO).normalize().tz_localize(FUSO_ESTACOES, nonexistent='shift_forward', ambiguous=True)
fim_exclusivo_local = (pd.Timestamp(DATA_FIM).normalize() + pd.Timedelta(days=1)).tz_localize(FUSO_ESTACOES, nonexistent='shift_forward', ambiguous=True)
if inicio_local >= fim_exclusivo_local:
    raise ValueError('Período inválido.')
grade_inicio = pd.date_range(inicio_local, fim_exclusivo_local, freq='h', inclusive='left')
# Padding cobre as diferenças nas fronteiras e a última hora do período local.
inicio_utc = inicio_local.tz_convert('UTC').tz_localize(None).normalize() - pd.Timedelta(days=1)
fim_utc = fim_exclusivo_local.tz_convert('UTC').tz_localize(None).normalize() + pd.Timedelta(days=1)
dias_download = pd.date_range(inicio_utc, fim_utc, freq='D')
xmin, ymin, xmax, ymax = estacoes.total_bounds
area = [min(90., np.ceil((ymax + MARGEM_GRAUS)*10)/10),
        max(-180., np.floor((xmin - MARGEM_GRAUS)*10)/10),
        max(-90., np.floor((ymin - MARGEM_GRAUS)*10)/10),
        min(180., np.ceil((xmax + MARGEM_GRAUS)*10)/10)]
variaveis = ['total_precipitation', 'potential_evaporation' if TIPO_ET == 'potencial' else 'total_evaporation']
et_nome = 'EP' if TIPO_ET == 'potencial' else 'ET'
assinatura = hashlib.sha256(json.dumps({'area': area, 'variaveis': variaveis}, sort_keys=True).encode()).hexdigest()[:10]
PASTA = DIRETORIO_SAIDA / f'{DATA_INICIO}_{DATA_FIM}_{TIPO_ET}_{assinatura}'
for nome in ['originais', 'horarios', 'diarios']:
    (PASTA / nome).mkdir(parents=True, exist_ok=True)
print('Saída:', PASTA, '| área N/W/S/E:', area)

# %% 6. Baixar mensalmente e extrair o pixel de cada estação
cliente = cdsapi.Client()  # Credenciais lidas de .cdsapirc; não inserir chave aqui.
partes = {c: [] for c in estacoes.codigo}
pixels, log = [], []
for mes in dias_download.to_period('M').unique():
    dias = dias_download[dias_download.to_period('M') == mes]
    pedido = {'variable': variaveis, 'year': [str(mes.year)], 'month': [f'{mes.month:02d}'],
              'day': [f'{d.day:02d}' for d in dias], 'time': [f'{h:02d}:00' for h in range(24)],
              'area': area, 'data_format': 'netcdf', 'download_format': 'unarchived'}
    chave = hashlib.sha256(json.dumps(pedido, sort_keys=True).encode()).hexdigest()[:12]
    arquivo = PASTA / 'originais' / f'ERA5Land_{mes}_{chave}.nc'
    print('Mês:', mes, '| reutilizar' if arquivo.exists() and not REFAZER_DOWNLOAD else '| baixar', flush=True)
    try:
        if REFAZER_DOWNLOAD or not arquivo.exists():
            parcial = arquivo.with_suffix('.parcial')
            parcial.unlink(missing_ok=True)
            cliente.retrieve('reanalysis-era5-land', pedido, str(parcial))
            dados, pix = extrair_arquivo(parcial, estacoes)
            parcial.replace(arquivo)
        else:
            dados, pix = extrair_arquivo(arquivo, estacoes)
        for codigo, tabela in dados.items():
            if set(tabela.columns) != {'P', et_nome}:
                raise ValueError('Download sem as duas variáveis solicitadas.')
            partes[codigo].append(tabela)
        pixels.extend(pix)
        log.append({'mes': str(mes), 'status': 'ok', 'arquivo': arquivo.name})
    except Exception as erro:
        log.append({'mes': str(mes), 'status': 'erro', 'classe_erro': type(erro).__name__})
        pd.DataFrame(log).to_csv(PASTA / 'log_download.csv', sep=';', index=False)
        raise RuntimeError(f'Falha no mês {mes}: {type(erro).__name__}. Downloads anteriores preservados; executar novamente.') from None
    pd.DataFrame(log).to_csv(PASTA / 'log_download.csv', sep=';', index=False)

# %% 7. Desacumular, alinhar horários e acumular por dia
horarios_por_estacao, diarios_por_estacao, diagnosticos = {}, {}, []
for codigo, blocos in partes.items():
    acumulados = pd.concat(blocos).sort_index()
    if acumulados.index.has_duplicates:
        raise ValueError(f'{codigo}: timestamps repetidos nos arquivos mensais.')
    horario = desacumular(acumulados)
    # CDS usa fim da hora: deslocar uma hora para obter início do intervalo.
    horario.index = (horario.index.tz_localize('UTC') - pd.Timedelta(hours=1)).tz_convert(FUSO_ESTACOES)
    horario = horario.reindex(grade_inicio)
    diario, contagem, esperadas = agregar_diarios(horario)
    diarios_por_estacao[codigo] = diario
    horarios_por_estacao[codigo] = horario
    for var in ['P', et_nome]:
        diagnosticos.append({'codigo_estacao': codigo, 'variavel': var,
                             'horas_validas': int(horario[var].notna().sum()),
                             'horas_ausentes': int(horario[var].isna().sum()),
                             'dias_validos': int(diario[var].notna().sum()),
                             'dias_ausentes': int(diario[var].isna().sum()),
                             'horas_negativas': int((horario[var] < 0).sum())})
        for resolucao, tabela, pasta in [('horaria', horario, 'horarios'), ('diaria', diario, 'diarios')]:
            saida = tabela[[var]].rename(columns={var: 'valor'}).copy()
            if resolucao == 'horaria' and REFERENCIA_HORA == 'fim':
                saida.index += pd.Timedelta(hours=1)
            # ISO com offset mantém horas únicas mesmo em mudança de horário de verão.
            saida.index = indice_exportacao(saida.index)
            saida['codigo_estacao'] = codigo
            saida['variavel'] = var
            saida['unidade'] = 'mm/h' if resolucao == 'horaria' else 'mm/dia'
            saida['fonte'] = 'ERA5-Land'
            if resolucao == 'diaria':
                saida['horas_validas'] = contagem[var].to_numpy()
                saida['horas_esperadas'] = esperadas.to_numpy()
            saida.to_csv(PASTA / pasta / f'ERA5Land_{codigo}_{var}_{resolucao}.csv',
                         sep=';', index_label='data_hora', encoding='utf-8-sig')

# %% 8. Exportar CSVs consolidados, coordenadas e metadados
for resolucao, series_estacoes in [('horarias', horarios_por_estacao), ('diarias', diarios_por_estacao)]:
    amplo = pd.concat([df.rename(columns={v: f'{c}_{v}' for v in df}) for c, df in series_estacoes.items()], axis=1)
    if resolucao == 'horarias' and REFERENCIA_HORA == 'fim':
        amplo.index += pd.Timedelta(hours=1)
    amplo.index = indice_exportacao(amplo.index)
    amplo.to_csv(PASTA / f'ERA5Land_series_{resolucao}.csv', sep=';', encoding='utf-8-sig')
coordenadas = pd.DataFrame(pixels).drop_duplicates()
if coordenadas.codigo_estacao.duplicated().any():
    raise ValueError('Pixel de uma estação mudou entre arquivos; conferir grades mensais.')
coordenadas.to_csv(PASTA / 'coordenadas_estacoes_pixels.csv', sep=';', index=False, encoding='utf-8-sig')
pd.DataFrame(diagnosticos).to_csv(PASTA / 'diagnostico_disponibilidade.csv', sep=';', index=False)
estacoes.to_file(PASTA / 'estacoes_extraidas.gpkg', layer='estacoes', driver='GPKG')
metadados = {'dataset': 'reanalysis-era5-land', 'variaveis_CDS': variaveis,
             'inicio': DATA_INICIO, 'fim': DATA_FIM, 'fuso': FUSO_ESTACOES,
             'referencia_hora': REFERENCIA_HORA, 'datas_com_fuso': SALVAR_DATAS_COM_FUSO, 'area_NWSE': area,
             'tolerancia_coordenadas_m': TOLERANCIA_COORDENADAS_M,
             'extracao': 'centro do pixel mais próximo ao ponto, sem interpolação',
             'conversao': '01UTC=acumulado; outras horas=diferenca consecutiva; P*1000; ET*(-1000)',
             'acumulacao_diaria': 'soma de intervalos com inicio no dia local',
             'fracao_minima_dia': MIN_FRACAO_HORAS_DIA,
             'ET': 'evaporacao potencial de agua livre, nao ETo FAO56' if TIPO_ET == 'potencial' else 'evapotranspiracao real total',
             'negativos_ET': 'preservados; representam fluxo descendente/condensacao',
             'substituicao': 'produto bruto; alinhar fuso e referencia horaria e substituir somente NaN observados'}
(PASTA / 'metadados.json').write_text(json.dumps(metadados, ensure_ascii=False, indent=2), encoding='utf-8')
print(pd.DataFrame(diagnosticos).to_string(index=False))
print('Arquivos prontos para uso como produtos locais:', PASTA)
