# -*- coding: utf-8 -*-
"""Precipitação diária ANA para bacia_03 e faixa externa configurável.
Executar após o script 04, no ambiente epagri2026.
Saída: mesmo estudo diário das vazões / dados / chuva / código da estação.
Preserva respostas originais e coordenadas; seleciona consistência exatamente 2.
Exporta estações selecionadas em GeoPackage e Shapefile (longitude/latitude,
EPSG:4326), na pasta precipitacao_ANA_downloads / execução.

USO: executar no Spyder/epagri2026 após o script 04. Ajustar DATA_INICIO,
DATA_FIM e, se necessário, ARQUIVO_ESTACOES_PLU na configuração abaixo.
A distância e a inclusão de externas são recuperadas do estudo anterior.
API atual: fornecer ANA_TOKEN ou ANA_IDENTIFICADOR + ANA_SENHA como
variáveis de ambiente antes de executar (credenciais disponibilizadas pela ANA).
Sem credenciais, auto tenta o serviço XML utilizado no notebook enviado.
A ANA anunciou migração do serviço antigo até 30/06/2026; ele pode falhar.
O código registra indisponibilidade, sem apresentá-la como download realizado.

Fontes consultadas em 03/10/2026:
https://www.ana.gov.br/hidrowebservice/swagger-ui/index.html
https://www.ana.gov.br/telemetria1ws/Dados.aspx
https://www.gov.br/ana/pt-br/assuntos/monitoramento-e-eventos-criticos/monitoramento-hidrologico/orientacoes-manuais/manuais/manual-hidrowebservice_publica.pdf

"""
from pathlib import Path
from datetime import datetime
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
import calendar
import math
import json
import os
import re
import time
import unicodedata
import xml.etree.ElementTree as ET
import pandas as pd
import geopandas as gpd

# ================= CONFIGURAÇÃO =================
BASE = Path(r'C:\Users\afrod\OneDrive\Documents\HydroUAI_git\hydrouai-epagri-2026\02_Dados\BD01\BD01')
BACIA = 'bacia_03'
PASTA_ESTUDOS = None  # None: última execução do script 04 para esta bacia.
ARQUIVO_ESTACOES_PLU = None  # Ex.: BASE / 'SIG' / 'Pluviometricas.shp'
CAMADA_ESTACOES = None  # GeoPackage: camada desejada, ou None para todas.
# None: recuperar a distância do diagnóstico usado no script 04.
DISTANCIA_BORDA_KM = None
# None: recuperar a escolha de internas/externas do script 04.
INCLUIR_EXTERNAS_PROXIMAS = None
CRS_METRICO = 31982  # SC; projeção com unidades em metros.
DATA_INICIO = '1940-01-01'
DATA_FIM = '2025-12-31'
NIVEL_CONSISTENCIA = 2
# auto: API atual quando há credenciais; senão serviço XML do notebook.
MODO_API = 'auto'  # 'atual', 'legado' ou 'auto'
URL_LEGADO = 'https://telemetriaws1.ana.gov.br/ServiceANA.asmx'
URL_ATUAL = 'https://www.ana.gov.br/hidrowebservice/EstacoesTelemetricas'
# API atual: definir ANA_IDENTIFICADOR e ANA_SENHA no ambiente, ou ANA_TOKEN.
# Não gravar credenciais nos relatórios ou no código.
TIMEOUT_SEGUNDOS = 120
INTERVALO_REQUISICOES_SEGUNDOS = 0.5
TENTATIVAS = 3
SENTINELAS = [-9999, -999, -99999]
# Colunas da camada podem ser fixadas; None usa reconhecimento automático.
CAMPO_CODIGO = None
CAMPO_TIPO = None
CAMPO_NOME = None
# IDs numéricos de tipo de estação variam entre cadastros: declarar se necessário.
VALORES_TIPO_PLU = ['Pluviométrica', 'Pluviometrica', 'Pluviometrico', 'Plu', '2']


def norm(value):
    return re.sub('[^a-z0-9]', '', unicodedata.normalize('NFKD', str(value)).encode('ascii', 'ignore').decode().lower())


def code(value):
    if pd.isna(value):
        return ''
    match = re.match(r'^(\d{1,8})(?:\.0)?(?:$|[\s\-_/])', str(value).strip())
    return match.group(1).zfill(8) if match else ''


def get_bytes(url, params=None, headers=None):
    if params:
        url += '?' + urlencode(params)
    for attempt in range(TENTATIVAS):
        try:
            request = Request(url, headers=headers or {})
            with urlopen(request, timeout=TIMEOUT_SEGUNDOS) as response:
                payload = response.read()
            time.sleep(INTERVALO_REQUISICOES_SEGUNDOS)
            return payload
        except HTTPError as error:
            if error.code not in [429, 500, 502, 503, 504] or attempt == TENTATIVAS - 1:
                raise RuntimeError(f'HTTP {error.code}: falha ao consultar serviço ANA.') from None
        except (URLError, TimeoutError):
            if attempt == TENTATIVAS - 1:
                raise RuntimeError('Serviço ANA sem resposta dentro do tempo limite.') from None
        time.sleep(min(2 ** attempt * 2, 10))


def xml_rows(payload, needed):
    """Ignora prefixos de namespace, preserva todos os campos de cada registro."""
    root = ET.fromstring(payload)
    def local(tag): return tag.rsplit('}', 1)[-1]
    rows = []
    for element in root.iter():
        if local(element.tag) in ['Fault', 'faultstring']:
            raise ValueError('Serviço ANA retornou erro XML/SOAP.')
        children = list(element)
        fields = {local(child.tag): child.text.strip() if child.text else '' for child in children if not list(child)}
        if needed.intersection(fields):
            rows.append(fields)
    return rows


def parse_date(value):
    text = str(value).strip()
    if re.match(r'^\d{4}-\d{2}-\d{2}', text):
        return pd.to_datetime(text, errors='coerce')
    return pd.to_datetime(text, dayfirst=True, errors='coerce')


def converter_diarios(records, station, start, end):
    """Série mensal Chuva01..31; nunca transformar dia inválido em observação."""
    rows, issues = [], []
    for record in records:
        f = {norm(k): v for k, v in record.items()}
        lv = pd.to_numeric(f.get('nivelconsistencia', None), errors='coerce')
        if pd.isna(lv) or lv != NIVEL_CONSISTENCIA:
            continue
        returned_code = code(f.get('estacaocodigo', f.get('codigoestacao', station)))
        if returned_code != station:
            raise ValueError(f'Resposta contém código {returned_code}, esperado {station}.')
        base = parse_date(f.get('datahora', f.get('datahoramedicao', f.get('datamedicao', f.get('data', '')))))
        if pd.isna(base):
            issues.append('Registro mensal com data inválida.')
            continue
        found = False
        for day in range(1, calendar.monthrange(base.year, base.month)[1] + 1):
            keys = [f'chuva{day:02d}', f'chuva{day}']
            key = next((k for k in keys if k in f), None)
            if key is None:
                continue
            found = True
            date = pd.Timestamp(base.year, base.month, day)
            if not (start <= date <= end):
                continue
            raw = str(f[key]).strip()
            if ',' in raw:
                raw = raw.replace('.', '').replace(',', '.')
            value = pd.to_numeric(raw, errors='coerce')
            if pd.notna(value) and value in SENTINELAS:
                value = float('nan')
            rows.append({'data_hora': date.strftime('%Y-%m-%d'), 'valor': value,
                         'codigo_estacao': station, 'fonte': 'ANA', 'variavel': 'Precipitação (mm) [consistencia=2]',
                         'tipo': 'chuva', 'resolucao': 'diaria', 'nivel_consistencia': int(lv),
                         'status_ANA': f.get(f'chuva{day:02d}status', f.get(f'chuva{day:02d}situacao', ''))})
        if not found:
            issues.append('Registro sem campos Chuva01..31: formato de retorno não reconhecido.')
    cols = ['data_hora', 'valor', 'codigo_estacao', 'fonte', 'variavel', 'tipo', 'resolucao', 'nivel_consistencia', 'status_ANA']
    result = pd.DataFrame(rows, columns=cols)
    if not result.empty:
        # Só eliminar duplicatas integrais; conflitos na mesma data permanecem.
        result = result.drop_duplicates().sort_values('data_hora')
    return result, issues


def pasta_estudos():
    if PASTA_ESTUDOS is not None:
        root = Path(PASTA_ESTUDOS)
    else:
        candidates = sorted((p for p in (BASE / 'Estudos_bacia_03').glob('*')
                             if p.is_dir() and (p / 'criterios.json').exists()
                             and (p / '01_diarios_ANA_consistencia_2').is_dir()), reverse=True)
        root = next((p for p in candidates if json.loads((p / 'criterios.json').read_text(encoding='utf-8')).get('bacia') == BACIA), None)
        if root is None:
            raise FileNotFoundError('Execute o script 04 primeiro, ou defina PASTA_ESTUDOS com a pasta criada por ele.')
    criteria = json.loads((root / 'criterios.json').read_text(encoding='utf-8'))
    if criteria.get('bacia') != BACIA:
        raise ValueError('PASTA_ESTUDOS não corresponde à bacia escolhida.')
    if not (root / 'bacia.gpkg').exists():
        raise FileNotFoundError('bacia.gpkg ausente na pasta do estudo. Recuperar o arquivo do diagnóstico.')
    cfg_file = Path(criteria['diagnostico_origem']) / 'configuracao.json'
    cfg = json.loads(cfg_file.read_text(encoding='utf-8')) if cfg_file.exists() else {}
    distance = DISTANCIA_BORDA_KM if DISTANCIA_BORDA_KM is not None else cfg.get('distancia_borda_km')
    if distance is None:
        raise ValueError('Não foi possível recuperar a faixa externa. Defina DISTANCIA_BORDA_KM.')
    if not pd.notna(distance) or not math.isfinite(float(distance)) or distance < 0:
        raise ValueError('Distância da borda deve ser não negativa.')
    include = INCLUIR_EXTERNAS_PROXIMAS if INCLUIR_EXTERNAS_PROXIMAS is not None else criteria.get('incluir_externas_proximas', True)
    return root, float(distance), include


def estacoes_locais():
    paths = [Path(ARQUIVO_ESTACOES_PLU)] if ARQUIVO_ESTACOES_PLU is not None else sorted((BASE / 'SIG').rglob('*.shp')) + sorted((BASE / 'SIG').rglob('*.gpkg')) + sorted((BASE / 'SIG').rglob('*.geojson'))
    frames = []
    for path in paths:
        layers = [CAMADA_ESTACOES] if CAMADA_ESTACOES else list(gpd.list_layers(path)['name']) if path.suffix.lower() == '.gpkg' else [None]
        for layer in layers:
            g = gpd.read_file(path, **({'layer': layer} if layer else {}))
            if not g.geom_type.eq('Point').any():
                continue
            if g.crs is None:
                raise ValueError(f'Camada de estações sem CRS: {path}')
            cols = {norm(c): c for c in g}
            cc = CAMPO_CODIGO or next((cols[k] for k in ['codigo', 'codigoana', 'codigoestacao', 'estacaocodigo', 'codana', 'codestacao', 'stationid'] if k in cols), None)
            tc = CAMPO_TIPO or next((cols[k] for k in ['tipoestaca', 'tipoestacao', 'tipo', 'stationtype'] if k in cols), None)
            nc = CAMPO_NOME or next((cols[k] for k in ['nome', 'estacaonome', 'nomeestacao', 'name'] if k in cols), None)
            if not cc:
                continue
            # Camada explícita deve conter apenas pluviométricas quando não há tipo.
            layer_name = norm(path.stem + str(layer or ''))
            plu_name = any(k in layer_name for k in ['pluvi', 'plu', 'rain']) and 'fluvi' not in layer_name
            if tc:
                mask = g[tc].map(norm).isin([norm(v) for v in VALORES_TIPO_PLU]) | g[tc].map(norm).str.contains('pluvi')
                g = g[mask]
            elif ARQUIVO_ESTACOES_PLU is None and not plu_name:
                continue
            g = g[g.geom_type.eq('Point')].to_crs(4326)
            if g.empty:
                continue
            frames.append(gpd.GeoDataFrame({'codigo': g[cc].map(code), 'nome': g[nc].fillna('').astype(str) if nc else '',
                                           'fonte_posicao': str(path), 'longitude': g.geometry.x, 'latitude': g.geometry.y,
                                           'geometry': g.geometry}, crs=4326))
    return pd.concat(frames, ignore_index=True) if frames else None


def inventario_para_pontos(records):
    rows = []
    for record in records:
        f = {norm(k): v for k, v in record.items()}
        c = code(f.get('codigoestacao', f.get('codigo', f.get('estacaocodigo', ''))))
        def number(v): return pd.to_numeric(str(v).replace(',', '.'), errors='coerce')
        x, y = number(f.get('longitude')), number(f.get('latitude'))
        typ = norm(f.get('tipoestacao', f.get('tipo', '')))
        if typ and typ not in [norm(v) for v in VALORES_TIPO_PLU] and 'pluvi' not in typ:
            continue
        if not c or not pd.notna(x + y) or not (-180 <= x <= 180 and -90 <= y <= 90):
            continue
        rows.append({'codigo': c, 'nome': f.get('estacaonome', f.get('nome', '')), 'longitude': x, 'latitude': y, 'fonte_posicao': 'Inventário ANA'})
    if not rows:
        raise ValueError('Inventário não retornou estações pluviométricas com coordenadas reconhecidas.')
    df = pd.DataFrame(rows)
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=4326)


def selecionar_espacial(g, polygon, distance, include):
    g = gpd.GeoDataFrame(g, geometry='geometry', crs=4326).copy()
    g = g[g.codigo.ne('')].drop_duplicates(['codigo', 'longitude', 'latitude'])
    metric = g.to_crs(CRS_METRICO)
    g['distancia_borda_km'] = metric.distance(polygon.boundary) / 1000
    inside = metric.geometry.map(polygon.covers)
    g['relacao'] = 'externa_proxima'
    g.loc[inside, 'relacao'] = 'interna'
    return g[inside | ((g.distancia_borda_km <= distance) & include)].copy()


class ClienteANA:
    def __init__(self):
        self.token = os.environ.get('ANA_TOKEN', '').removeprefix('Bearer ').strip()
        self.user = os.environ.get('ANA_IDENTIFICADOR', '')
        self.password = os.environ.get('ANA_SENHA', '')
        self.generated = time.monotonic() if self.token else 0
        self.mode = MODO_API
        if self.mode == 'auto':
            self.mode = 'atual' if self.token or (self.user and self.password) else 'legado'
        if self.mode not in ['atual', 'legado']:
            raise ValueError('MODO_API deve ser auto, atual ou legado.')
        if self.mode == 'atual' and not (self.token or (self.user and self.password)):
            raise ValueError('API atual exige ANA_TOKEN ou ANA_IDENTIFICADOR + ANA_SENHA no ambiente.')
        if self.mode == 'legado':
            print('Usando serviço XML do notebook. A ANA anunciou migração até 30/06/2026; disponibilidade deve ser verificada.')

    @staticmethod
    def items(payload):
        response = json.loads(payload)
        if not isinstance(response, dict):
            raise ValueError('Resposta JSON da ANA fora do formato esperado.')
        if str(response.get('code', 200)) not in ['200', '201']:
            raise ValueError('API ANA retornou falha no resultado da consulta; consultar resposta original.')
        records = response.get('items', [])
        if records is None:
            return []
        if isinstance(records, dict):
            records = [records]
        if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
            raise ValueError('Campo items fora do formato esperado.')
        return records

    def headers(self):
        if not self.token or time.monotonic() - self.generated >= 55 * 60:
            if not self.user or not self.password:
                raise ValueError('Token expirado: fornecer novo ANA_TOKEN ou credenciais no ambiente.')
            payload = get_bytes(URL_ATUAL + '/OAUth/v1', headers={'Identificador': self.user, 'Senha': self.password})
            response = json.loads(payload)
            item = response.get('items', {})
            if isinstance(item, list):
                item = item[0] if item else {}
            self.token = item.get('tokenautenticacao', '') if isinstance(item, dict) else ''
            if not self.token:
                raise ValueError('Autenticação ANA não forneceu token. Conferir cadastro e credenciais.')
            self.generated = time.monotonic()
        return {'Authorization': 'Bearer ' + self.token}

    def inventario(self, raw_folder):
        if self.mode == 'legado':
            params = {k: '' for k in ['codEstDE', 'codEstATE', 'nmEst', 'nmRio', 'codSubBacia', 'codBacia', 'nmMunicipio', 'nmEstado', 'sgResp', 'sgOper', 'telemetrica']}
            params['tpEst'] = '2'
            payload = get_bytes(URL_LEGADO + '/HidroInventario', params)
            (raw_folder / 'inventario.xml').write_bytes(payload)
            records = xml_rows(payload, {'Codigo', 'EstacaoCodigo', 'CodigoEstacao', 'Latitude'})
        else:
            records = []
            # UF da bacia e vizinhas para não limitar a faixa externa pela fronteira estadual.
            for uf in ['SC', 'PR', 'RS']:
                payload = get_bytes(URL_ATUAL + '/HidroInventarioEstacoes/v1', {'Unidade Federativa': uf}, self.headers())
                (raw_folder / f'inventario_{uf}.json').write_bytes(payload)
                records.extend(self.items(payload))
        return inventario_para_pontos(records)

    def serie(self, station, start, end, raw_folder):
        if self.mode == 'legado':
            params = {'codEstacao': station, 'dataInicio': start.replace(day=1).strftime('%d/%m/%Y'),
                      'dataFim': end.replace(day=calendar.monthrange(end.year, end.month)[1]).strftime('%d/%m/%Y'), 'tipoDados': '2', 'nivelConsistencia': str(NIVEL_CONSISTENCIA)}
            payload = get_bytes(URL_LEGADO + '/HidroSerieHistorica', params)
            (raw_folder / f'{station}.xml').write_bytes(payload)
            return xml_rows(payload, {'DataHora', 'Chuva01', 'Chuva02', 'NivelConsistencia'})
        records = []
        # Ano calendário: cada consulta <=366 dias. Consultar meses completos;
        # o filtro exato do período solicitado é aplicado após expandir Chuva01..31.
        for year in range(start.year, end.year + 1):
            a = pd.Timestamp(year, start.month, 1) if year == start.year else pd.Timestamp(year, 1, 1)
            b = pd.Timestamp(year, end.month, calendar.monthrange(year, end.month)[1]) if year == end.year else pd.Timestamp(year, 12, 31)
            params = {'Código da Estação': int(station), 'Tipo Filtro Data': 'DATA_LEITURA',
                      'Data Inicial (yyyy-MM-dd)': a.strftime('%Y-%m-%d'), 'Data Final (yyyy-MM-dd)': b.strftime('%Y-%m-%d')}
            payload = get_bytes(URL_ATUAL + '/HidroSerieChuva/v1', params, self.headers())
            (raw_folder / f'{station}_{year}.json').write_bytes(payload)
            records.extend(self.items(payload))
        return records


def save(df, path):
    pd.DataFrame(df).to_csv(path, sep=';', index=False, encoding='utf-8-sig')


def exportar_estacoes(estacoes, pasta):
    """Uma feição por posição conhecida; códigos permanecem como texto.

    Manter juntos os arquivos .shp, .shx, .dbf, .prj e .cpg.
    Pode ser chamada sobre o GeoPackage existente, sem baixar séries novamente.
    """
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    pontos = estacoes.to_crs('EPSG:4326').copy()
    pontos['codigo'] = pontos['codigo'].astype(str)
    pontos.to_file(pasta / 'estacoes_pluviometricas.gpkg', driver='GPKG', index=False)
    # DBF aceita nomes de campos de até 10 caracteres. O nome completo e a
    # fonte da posição continuam disponíveis no CSV e no GeoPackage.
    shape = pontos[['codigo', 'nome', 'longitude', 'latitude', 'relacao',
                    'distancia_borda_km', 'geometry']].rename(
                        columns={'distancia_borda_km': 'dist_km'}).copy()
    shape['nome'] = shape['nome'].fillna('').astype(str).str.slice(0, 80)
    shape['longitude'] = shape.geometry.x
    shape['latitude'] = shape.geometry.y
    destino = pasta / 'estacoes_pluviometricas.shp'
    shape.to_file(destino, driver='ESRI Shapefile', encoding='UTF-8', index=False)
    return destino


def main():
    from pyproj import CRS
    projection = CRS.from_user_input(CRS_METRICO)
    if not projection.is_projected or any(abs(a.unit_conversion_factor - 1) > 1e-9 for a in projection.axis_info):
        raise ValueError('CRS_METRICO deve usar unidades em metros.')
    if NIVEL_CONSISTENCIA != 2:
        raise ValueError('Este estudo seleciona somente consistência 2.')
    start, end = pd.Timestamp(DATA_INICIO).normalize(), pd.Timestamp(DATA_FIM).normalize()
    if end < start:
        raise ValueError('DATA_FIM anterior à DATA_INICIO.')
    root, distance, include = pasta_estudos()
    daily_folder = root / '01_diarios_ANA_consistencia_2'
    if not daily_folder.is_dir():
        raise FileNotFoundError(f'Pasta diária não encontrada: {daily_folder}')
    boundary = gpd.read_file(root / 'bacia.gpkg')
    if boundary.crs is None:
        raise ValueError('Bacia sem CRS.')
    polygon = boundary.to_crs(CRS_METRICO).geometry.union_all()
    if not polygon.is_valid:
        raise ValueError('Geometria inválida: conferir diagnóstico antes de baixar.')
    run = daily_folder / 'precipitacao_ANA_downloads' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    raw_folder = run / 'respostas_originais'
    raw_folder.mkdir(parents=True, exist_ok=False)
    client = ClienteANA()
    stations = estacoes_locais()
    if stations is None:
        print('Nenhuma camada pluviométrica reconhecida em SIG; consultando inventário ANA.')
        stations = client.inventario(raw_folder)
    selected = selecionar_espacial(stations, polygon, distance, include)
    save(selected.drop(columns='geometry'), run / 'estacoes_selecionadas.csv')
    if selected.empty:
        raise ValueError('Nenhuma estação pluviométrica selecionada. Conferir camada, CRS e faixa externa.')
    shape_estacoes = exportar_estacoes(selected, run)
    print(f'Shapefile das estações: {shape_estacoes}', flush=True)
    logs, catalogs, frames, failures = [], [], [], []
    for station, group in selected.groupby('codigo', sort=True):
        print(f'Baixando precipitação: {station}', flush=True)
        try:
            records = client.serie(station, start, end, raw_folder)
            data, issues = converter_diarios(records, station, start, end)
            if issues:
                failures.extend({'codigo': station, 'detalhe': issue} for issue in sorted(set(issues)))
            if data.empty:
                logs.append({'codigo': station, 'status': 'sem_dados_consistencia_2', 'registros_recebidos': len(records), 'dias_validos': 0})
                continue
            # Arquivo novo por execução: não sobrescrever arquivos do estudo anterior.
            target = daily_folder / 'dados' / 'chuva' / station / f'ANA_P_diaria_c2_{run.name}.csv'
            target.parent.mkdir(parents=True, exist_ok=True)
            data['relacao_codigo'] = 'interna' if group.relacao.eq('interna').any() else 'externa_proxima'
            data['serie_id'] = f'ANA_P_{station}_c2_{run.name}'
            data['numero_posicoes_conhecidas'] = len(group)
            data['longitude'] = group.iloc[0].longitude if len(group) == 1 else float('nan')
            data['latitude'] = group.iloc[0].latitude if len(group) == 1 else float('nan')
            save(data, target)
            positions = group.drop(columns='geometry').to_dict('records')
            metadata = {'bacia': BACIA, 'codigo_estacao': station, 'fonte': 'ANA', 'unidade': 'mm',
                        'resolucao': 'diaria', 'nivel_consistencia': 2, 'modo_api': client.mode,
                        'periodo_solicitado': [DATA_INICIO, DATA_FIM], 'posicoes': positions,
                        'originais': str(raw_folder), 'nota': 'Coordenadas alternativas não têm datas de validade; conferir histórico. Negativos e conflitos de mesma data preservados.'}
            target.with_suffix('.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
            valid = data[data.valor.notna()]
            valid_dates = pd.to_datetime(valid.data_hora).unique()
            entry = {'codigo_estacao': station, 'tipo': 'chuva', 'fonte': 'ANA', 'variavel': 'Precipitação (mm) [consistencia=2]',
                     'resolucao': 'diaria', 'nivel_consistencia': 2, 'serie_id': data.serie_id.iloc[0],
                     'relacao_codigo': data.relacao_codigo.iloc[0], 'inicio': data.data_hora.min(), 'fim': data.data_hora.max(),
                     'registros': len(data), 'dias_validos': len(valid_dates),
                     'disponibilidade_periodo_solicitado_pct': round(100 * len(valid_dates) / len(pd.date_range(start, end)), 2),
                     'datas_com_multiplos_registros': int(data.loc[data.data_hora.duplicated(False), 'data_hora'].nunique()),
                     'negativos': int((data.valor < 0).sum()), 'arquivo_estudo': str(target.relative_to(daily_folder))}
            catalogs.append(entry)
            logs.append({'codigo': station, 'status': 'salvo' if len(valid_dates) else 'salvo_sem_valores_validos', 'registros_recebidos': len(records), 'dias_validos': len(valid_dates)})
            frames.append(data)
        except Exception as error:
            # Mensagens próprias não incluem cabeçalhos de autenticação.
            logs.append({'codigo': station, 'status': 'erro', 'registros_recebidos': None, 'dias_validos': 0})
            failures.append({'codigo': station, 'detalhe': f'{type(error).__name__}: {error}'})
            print(f'Falha na estação {station}: {type(error).__name__}', flush=True)
        # Salvar progresso após cada estação; uma falha não apaga downloads anteriores.
        save(logs, run / 'log_download.csv')
        save(catalogs, run / 'catalogo_precipitacao.csv')
        save(failures, run / 'pendencias.csv')
    if frames:
        save(pd.concat(frames, ignore_index=True), run / 'precipitacao_diaria_consolidada.csv')
    cfg = {'bacia': BACIA, 'pasta_diaria': str(daily_folder), 'distancia_borda_km': distance,
           'incluir_externas_proximas': include, 'data_inicio': DATA_INICIO, 'data_fim': DATA_FIM,
           'nivel_consistencia': NIVEL_CONSISTENCIA, 'modo_api': client.mode}
    (run / 'configuracao.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
    report = '<!doctype html><meta charset="utf-8"><title>Precipitação ANA bacia_03</title><h1>Precipitação diária ANA — bacia_03</h1>'
    report += f'<p>Consistência 2. Período: {DATA_INICIO} a {DATA_FIM}. Faixa externa: {distance:g} km. Modo API: {client.mode}.</p>'
    report += pd.DataFrame(logs).to_html(index=False, escape=True)
    report += '<h2>Séries baixadas</h2>' + pd.DataFrame(catalogs).to_html(index=False, escape=True)
    report += '<h2>Pendências</h2>' + pd.DataFrame(failures).to_html(index=False, escape=True)
    report += '<p>Downloads independentes dos arquivos de vazão. O catálogo e o consolidado anteriores permanecem como estavam; consultar catalogo_precipitacao.csv para integrar na análise. Não houve preenchimento ou correção de valores.</p>'
    (run / 'relatorio_download.html').write_text(report, encoding='utf-8')
    print(f'\nDados: {daily_folder / "dados" / "chuva"}\nRelatório: {run / "relatorio_download.html"}')
    if not catalogs:
        raise RuntimeError('Nenhuma série foi salva. Conferir log_download.csv e pendencias.csv; o serviço pode estar indisponível ou não haver dados de consistência 2.')


if __name__ == '__main__':
    main()
