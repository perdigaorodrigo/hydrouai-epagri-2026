# -*- coding: utf-8 -*-
"""CHIRPS diário v2.0: download público, recorte e extração nas pluviométricas.
Executar por células no Spyder, ambiente epagri2026, ANTES do curso.
Dependências (Anaconda Prompt):
conda install -n epagri2026 -c conda-forge geopandas rasterio requests pandas numpy
Fonte: https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/tifs/p05/
Documentação: https://chc.ucsb.edu/data/chirps (consultada 04/10/2026).
Produto fixo v2.0, 0.05°, mm/dia, desde 1981; não misturar versões.
CHIRPS v3 está disponível; a produção de v2 termina após dezembro/2026.
Aqui usamos v2 para a série histórica do curso, não para operação futura.
Cada download traz um raster global comprimido; só os recortes ficam salvos.
O CSV corresponde ao pixel da estação, não a observação pluviométrica.
Não há correção de viés automática nem alteração dos dados da ANA.
"""

# %% 1. Bibliotecas
from pathlib import Path
import gzip
import hashlib
import json
import shutil
import tempfile
import time
import sqlite3
from pyproj import Transformer
from shapely.ops import transform as transformar
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping, box
import requests

# %% 2. Diretórios, período e estações — AJUSTAR
DIRETORIO_SIG = Path(r'C:\Users\afrod\OneDrive\UFMG\Projetos extensão\HydroUAI-courses\Cursos externos\Epagri-SC 2026\Base de Dados\Dia 1\SIG')
DIRETORIO_SAIDA = Path(r'C:\Users\afrod\OneDrive\UFMG\Projetos extensão\HydroUAI-courses\Cursos externos\Epagri-SC 2026\Base de Dados\Dia 1\Dados\CHIRPS')
ARQUIVO_BACIA = DIRETORIO_SIG / 'bacia.gpkg'
ARQUIVO_ESTACOES = DIRETORIO_SIG / 'estacoes_pluviometricas.gpkg'
CAMADA_BACIA = None
CAMADA_ESTACOES = None
CAMPO_CODIGO = 'codigo'
DATA_INICIO = '2010-01-01'
DATA_FIM = '2025-12-31'
CODIGOS_ESTACOES = []  # Vazia: todas as pluviométricas dentro da bacia.
# Testar inicialmente uma semana, depois ampliar o período.
SALVAR_RASTERS_BACIA = True
GUARDAR_GZ_GLOBAL = False  # True: ocupa vários MB por dia; False: descartar após recorte.
COBERTURA_MINIMA_BACIA = 1.0  # Fração válida exigida; 1 = toda a área com dados.
LIMIAR_CHUVA_MM = 0.0  # Opcional: 1.0; aplicado só na coluna valor_tratado.
TENTATIVAS = 3
URL_BASE = 'https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/tifs/p05'

# %% 3. Funções de apoio

def ler_geo(arquivo, camada):
    g = gpd.read_file(arquivo, **({'layer': camada} if camada else {}))
    if g.crs is None:
        raise ValueError(f'Arquivo sem CRS: {arquivo}')
    if g.empty or g.geometry.isna().any() or g.geometry.is_empty.any():
        raise ValueError(f'Geometrias ausentes/vazias: {arquivo}')
    if not g.geometry.is_valid.all():
        raise ValueError(f'Geometrias inválidas: revisar {arquivo}')
    return g.to_crs(4326)


def baixar_gz(sessao, url, destino):
    """Arquivo temporário: um download interrompido nunca vira cache válido."""
    parcial = destino.with_suffix(destino.suffix + '.part')
    for tentativa in range(1, TENTATIVAS + 1):
        try:
            with sessao.get(url, stream=True, timeout=(20, 120)) as r:
                if r.status_code == 404:
                    raise FileNotFoundError(f'Dia indisponível no servidor: {url}')
                r.raise_for_status()
                with parcial.open('wb') as f:
                    for bloco in r.iter_content(1024 * 1024):
                        if bloco: f.write(bloco)
            # Ler até o fim verifica integridade/CRC do gzip.
            with gzip.open(parcial, 'rb') as f:
                while f.read(1024 * 1024): pass
            parcial.replace(destino)
            return
        except FileNotFoundError:
            parcial.unlink(missing_ok=True)
            raise
        except (requests.RequestException, OSError, EOFError):
            parcial.unlink(missing_ok=True)
            if tentativa == TENTATIVAS: raise
            time.sleep(min(2 ** tentativa, 8))


def gravar_recorte(origem, destino, geometria):
    with rasterio.open(origem) as src:
        if src.count != 1 or src.crs is None or src.crs.to_epsg() != 4326:
            raise ValueError('Raster CHIRPS deve ter uma banda e CRS EPSG:4326.')
        dados, transf = mask(src, [mapping(geometria)], crop=True,
                             filled=False, all_touched=False)
        a = dados[0].astype('float32').filled(-9999)
        a[(~np.isfinite(a)) | (a < 0)] = -9999
        perfil = src.profile.copy()
        perfil.update(driver='GTiff', width=a.shape[1], height=a.shape[0],
                      count=1, transform=transf, dtype='float32', nodata=-9999,
                      compress='deflate')
        # Remover blocos globais e usar TIFF simples para pequenos recortes.
        for chave in ['blockxsize', 'blockysize', 'tiled']:
            perfil.pop(chave, None)
        temporario = destino.with_suffix('.part.tif')
        with rasterio.open(temporario, 'w', **perfil) as dst:
            dst.write(a, 1)
            dst.update_tags(produto='CHIRPS v2.0', unidade='mm/dia')
        temporario.replace(destino)


def extrair_pontos(arquivo, pontos):
    """Pixel que contém a coordenada; mantém o centro do pixel para auditoria."""
    linhas = []
    with rasterio.open(arquivo) as src:
        for _, est in pontos.iterrows():
            x, y = est.geometry.x, est.geometry.y
            row, col = src.index(x, y)
            if not (0 <= row < src.height and 0 <= col < src.width):
                raise ValueError(f'Estação fora do raster regional: {est.codigo}')
            v = next(src.sample([(x, y)], masked=True))[0]
            valor = float(v) if not np.ma.is_masked(v) and np.isfinite(v) and v >= 0 else np.nan
            lon_pixel, lat_pixel = src.xy(row, col)
            linhas.append({'codigo_estacao': str(est.codigo), 'valor': valor,
                           'longitude': x, 'latitude': y,
                           'longitude_pixel': lon_pixel, 'latitude_pixel': lat_pixel})
    return linhas

def pesos_area_bacia(arquivo, poligono):
    """Áreas de interseção pixel/bacia em projeção equivalente (EPSG:6933)."""
    reprojetar = Transformer.from_crs(4326, 6933, always_xy=True).transform
    with rasterio.open(arquivo) as src:
        pesos = np.zeros((src.height, src.width), dtype='float64')
        for row in range(src.height):
            for col in range(src.width):
                x0, y0 = src.transform * (col, row)
                x1, y1 = src.transform * (col + 1, row + 1)
                pixel = box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
                intersecao = poligono.intersection(pixel)
                if not intersecao.is_empty:
                    pesos[row, col] = transformar(reprojetar, intersecao).area
        if pesos.sum() <= 0:
            raise ValueError('Bacia sem interseção com os pixels do raster.')
        return pesos, (src.transform, src.crs, src.height, src.width)


def media_diaria_bacia(arquivo, pesos, grade):
    with rasterio.open(arquivo) as src:
        if (src.transform, src.crs, src.height, src.width) != grade:
            raise ValueError('Grade CHIRPS mudou; não reutilizar pesos de outra grade.')
        a = src.read(1, masked=True)
        valores = a.filled(np.nan).astype(float)
        valido = (~np.ma.getmaskarray(a)) & np.isfinite(valores) & (valores >= 0) & (pesos > 0)
        area_valida = float(pesos[valido].sum())
        cobertura = area_valida / pesos.sum()
        media = (float(np.sum(valores[valido] * pesos[valido]) / area_valida)
                 if area_valida > 0 and cobertura + 1e-10 >= COBERTURA_MINIMA_BACIA else np.nan)
        return {'valor': media, 'cobertura_area_pct': 100 * cobertura,
                'area_valida_km2': area_valida / 1e6, 'area_bacia_grade_km2': float(pesos.sum() / 1e6)}


# %% 4. Ler bacia e selecionar pluviométricas internas
bacias = ler_geo(ARQUIVO_BACIA, CAMADA_BACIA)
if not bacias.geom_type.isin(['Polygon', 'MultiPolygon']).all():
    raise ValueError('A camada da bacia deve conter polígonos.')
bacia = bacias.geometry.union_all()
estacoes = ler_geo(ARQUIVO_ESTACOES, CAMADA_ESTACOES)
if CAMPO_CODIGO not in estacoes:
    raise ValueError(f'Campo de código ausente: {CAMPO_CODIGO}')
if not estacoes.geom_type.eq('Point').all():
    raise ValueError('A camada das estações deve conter pontos.')
estacoes['codigo'] = estacoes[CAMPO_CODIGO].astype(str).str.strip()
if estacoes['codigo'].isin(['', 'None', 'nan']).any():
    raise ValueError('Existem códigos de estação ausentes.')
estacoes = estacoes[estacoes.geometry.apply(bacia.covers)].copy()
if CODIGOS_ESTACOES:
    pedidos = set(map(str, CODIGOS_ESTACOES))
    faltantes = pedidos - set(estacoes.codigo)
    if faltantes: raise ValueError(f'Códigos ausentes ou externos: {sorted(faltantes)}')
    estacoes = estacoes[estacoes.codigo.isin(pedidos)].copy()
estacoes['_geom'] = estacoes.geometry.to_wkb()
estacoes = estacoes.drop_duplicates(['codigo', '_geom']).drop(columns='_geom')
if estacoes.codigo.duplicated().any():
    raise ValueError('Há posições alternativas do mesmo código. Escolher a posição válida no GeoPackage antes de extrair.')
if estacoes.empty: raise ValueError('Nenhuma pluviométrica selecionada.')
estacoes = estacoes.sort_values('codigo').reset_index(drop=True)
if bacia.bounds[1] < -50 or bacia.bounds[3] > 50:
    raise ValueError('Bacia fora da cobertura latitudinal do CHIRPS v2.')
print('Estações:', estacoes.codigo.tolist())

# %% 5. Preparar período e cache específico da geometria
inicio, fim = pd.Timestamp(DATA_INICIO), pd.Timestamp(DATA_FIM)
if inicio != inicio.normalize() or fim != fim.normalize():
    raise ValueError('Informar datas sem horário.')
if inicio < pd.Timestamp('1981-01-01') or fim < inicio:
    raise ValueError('Período inválido: CHIRPS inicia em 1981.')
datas = pd.date_range(inicio, fim, freq='D')
# Uma margem inclui o pixel inteiro de estações junto à borda da bacia.
xmin, ymin, xmax, ymax = bacia.bounds
retangulo = box(xmin - .1, ymin - .1, xmax + .1, ymax + .1)
assinatura = hashlib.sha256((bacia.wkb_hex + str(list(zip(estacoes.codigo, estacoes.geometry.to_wkt()))) + URL_BASE).encode()).hexdigest()[:12]
PASTA = DIRETORIO_SAIDA / f'{inicio:%Y%m%d}_{fim:%Y%m%d}_{assinatura}'
REGIONAL = PASTA / 'rasters_regionais'
RASTERS_BACIA = PASTA / 'rasters_bacia'
SERIES = PASTA / 'series_estacoes'
for pasta in [REGIONAL, RASTERS_BACIA, SERIES]: pasta.mkdir(parents=True, exist_ok=True)
estacoes.to_file(PASTA / 'estacoes_CHIRPS.gpkg', driver='GPKG', index=False)
bacias.to_file(PASTA / 'bacia.gpkg', driver='GPKG', index=False)
(PASTA / 'configuracao.json').write_text(json.dumps({
    'produto': 'CHIRPS v2.0', 'resolucao_graus': .05, 'unidade': 'mm/dia',
    'inicio': DATA_INICIO, 'fim': DATA_FIM, 'url_base': URL_BASE,
    'arquivo_bacia': str(ARQUIVO_BACIA), 'arquivo_estacoes': str(ARQUIVO_ESTACOES),
    'extracao': 'pixel que contém a estação', 'limiar_mm_coluna_tratada': LIMIAR_CHUVA_MM,
    'media_bacia': 'ponderada pela área de interseção pixel/polígono, EPSG:6933',
    'cobertura_minima_bacia': COBERTURA_MINIMA_BACIA,
    'nota': 'valor é bruto; valor_tratado aplica limiar opcional. Não houve correção de viés.'}, ensure_ascii=False, indent=2), encoding='utf-8')
if LIMIAR_CHUVA_MM < 0: raise ValueError('Limiar deve ser >= 0.')
print('Dias:', len(datas), '| pasta:', PASTA)
print('Retomar: executar novamente com as mesmas datas/geometrias reutiliza recortes válidos.')

# %% 6. Baixar e recortar — executar antes do curso
log, registros, medias_bacia = [], [], []
pesos_bacia, grade_bacia = None, None
if not 0 < COBERTURA_MINIMA_BACIA <= 1:
    raise ValueError("COBERTURA_MINIMA_BACIA deve estar entre 0 e 1.")
with requests.Session() as sessao, tempfile.TemporaryDirectory(prefix='chirps_') as tmp:
    sessao.headers['User-Agent'] = 'EPAGRI2026-CHIRPS-course/1.0'
    for i, dia in enumerate(datas, 1):
        nome = f'chirps-v2.0.{dia:%Y.%m.%d}.tif'
        url = f'{URL_BASE}/{dia.year}/{nome}.gz'
        regional = REGIONAL / nome
        status, erro, itens = 'cache', '', []
        media_bacia = {'valor': np.nan, 'cobertura_area_pct': np.nan,
                       'area_valida_km2': np.nan, 'area_bacia_grade_km2': np.nan}
        try:
            if regional.exists():
                try:
                    itens = extrair_pontos(regional, estacoes)
                except (OSError, ValueError):
                    regional.unlink()
            if not regional.exists():
                gz = Path(tmp) / (nome + '.gz')
                tif = Path(tmp) / nome
                try:
                    baixar_gz(sessao, url, gz)
                    with gzip.open(gz, 'rb') as src, tif.open('wb') as dst:
                        shutil.copyfileobj(src, dst)
                    gravar_recorte(tif, regional, retangulo)
                    if GUARDAR_GZ_GLOBAL:
                        orig = PASTA / 'globais_gz'; orig.mkdir(exist_ok=True)
                        shutil.copy2(gz, orig / gz.name)
                finally:
                    gz.unlink(missing_ok=True); tif.unlink(missing_ok=True)
                status = 'baixado'
                itens = extrair_pontos(regional, estacoes)
            for item in itens: item['data_hora'] = dia.strftime('%Y-%m-%d')
            if pesos_bacia is None:
                pesos_bacia, grade_bacia = pesos_area_bacia(regional, bacia)
            media_bacia = media_diaria_bacia(regional, pesos_bacia, grade_bacia)
            if SALVAR_RASTERS_BACIA and not (RASTERS_BACIA / nome).exists():
                gravar_recorte(regional, RASTERS_BACIA / nome, bacia)
        except Exception as exc:
            status, erro = 'erro', f'{type(exc).__name__}: {exc}'
            # Dados ausentes permanecem NaN, nunca zero.
            if not itens:
                itens = [{'data_hora': dia.strftime('%Y-%m-%d'), 'codigo_estacao': str(e.codigo),
                          'valor': np.nan, 'longitude': e.geometry.x, 'latitude': e.geometry.y,
                          'longitude_pixel': np.nan, 'latitude_pixel': np.nan} for _, e in estacoes.iterrows()]
        media_bacia['data_hora'] = dia.strftime('%Y-%m-%d')
        media_bacia['bacia_id'] = 1
        medias_bacia.append(media_bacia)
        pd.DataFrame(medias_bacia).to_csv(PASTA / 'CHIRPS_diario_media_bacia.csv',
                                        sep=';', index=False, encoding='utf-8-sig')
        registros.extend(itens)
        log.append({'data_hora': str(dia.date()), 'status': status, 'url': url, 'erro': erro})
        pd.DataFrame(log).to_csv(PASTA / 'log_download.csv', sep=';', index=False, encoding='utf-8-sig')
        print(f'{i}/{len(datas)} | {dia:%Y-%m-%d} | {status}', flush=True)

# %% 7. Exportar séries diárias e relatório de disponibilidade
consolidado = pd.DataFrame(registros)
consolidado['valor_tratado'] = consolidado.valor.mask(consolidado.valor.lt(LIMIAR_CHUVA_MM), 0.)
consolidado['fonte'] = 'CHIRPS_v2.0'
consolidado['unidade'] = 'mm/dia'
consolidado.to_csv(PASTA / 'CHIRPS_diario_estacoes.csv', sep=';', index=False, encoding='utf-8-sig')
resumo = []
for codigo, grupo in consolidado.groupby('codigo_estacao', sort=True):
    destino = SERIES / f'CHIRPS_diario_{codigo}.csv'
    grupo.sort_values('data_hora').to_csv(destino, sep=';', index=False, encoding='utf-8-sig')
    resumo.append({'codigo_estacao': codigo, 'dias_solicitados': len(datas),
                   'dias_validos': int(grupo.valor.notna().sum()),
                   'dias_ausentes': int(grupo.valor.isna().sum()), 'arquivo': str(destino)})
relatorio = pd.DataFrame(resumo)
relatorio.to_csv(PASTA / 'resumo_disponibilidade.csv', sep=';', index=False, encoding='utf-8-sig')
print(relatorio.to_string(index=False))
print('CSV para o script da aula: usar data=data_hora, valor=valor, sep=;')
print('valor_tratado é opcional: valores abaixo do limiar são zero; bruto preservado em valor.')
if not consolidado.valor.notna().any():
    raise RuntimeError('Nenhum valor válido extraído. Conferir log_download.csv e conexão.')

# Exportar também a série temporal como tabela de atributos no GeoPackage.
# No QGIS: abrir a camada bacia e a tabela precipitacao_diaria, ligadas por bacia_id.
media_bacia_df = pd.DataFrame(medias_bacia)
media_bacia_df['unidade'] = 'mm/dia'
media_bacia_df['fonte'] = 'CHIRPS_v2.0'
media_bacia_df.to_csv(PASTA / 'CHIRPS_diario_media_bacia.csv', sep=';', index=False, encoding='utf-8-sig')
gpkg_media = PASTA / 'bacia_precipitacao_CHIRPS.gpkg'
camada_bacia = gpd.GeoDataFrame({'bacia_id': [1], 'dias_solicitados': [len(datas)],
    'dias_validos': [int(media_bacia_df.valor.notna().sum())],
    'P_media_periodo_mm_dia': [media_bacia_df.valor.mean()],
    'inicio': [DATA_INICIO], 'fim': [DATA_FIM]}, geometry=[bacia], crs=4326)
camada_bacia.to_file(gpkg_media, layer='bacia', driver='GPKG', index=False)
with sqlite3.connect(gpkg_media) as con:
    media_bacia_df.to_sql('precipitacao_diaria', con, if_exists='replace', index=False)
    con.execute("DELETE FROM gpkg_contents WHERE table_name='precipitacao_diaria'")
    con.execute("INSERT INTO gpkg_contents (table_name, data_type, identifier, description, last_change) "
                "VALUES ('precipitacao_diaria', 'attributes', 'precipitacao_diaria', "
                "'Precipitacao media diaria ponderada pela area de intersecao dos pixels com a bacia; mm/dia', "
                "strftime('%Y-%m-%dT%H:%M:%fZ','now'))")
print('Média diária da bacia:', PASTA / 'CHIRPS_diario_media_bacia.csv')
print('GeoPackage com polígono e tabela diária:', gpkg_media)

# %% 8. Exemplo de integração com PRODUTOS_LOCAIS no script 07
# Usar o serie_id EXATO da chuva ANA correspondente e o arquivo do mesmo código.
# PRODUTOS_LOCAIS = {
#     'SERIE_ID_ANA_CHUVA': {'CHIRPS': {
#         'arquivo': str(SERIES / 'CHIRPS_diario_CODIGO.csv'),
#         'data': 'data_hora', 'valor': 'valor', 'sep': ';',
#         'campo_codigo': 'codigo_estacao', 'codigo': 'CODIGO'}}}
# Para limiar de 1 mm, configurar valor='valor_tratado' explicitamente.
