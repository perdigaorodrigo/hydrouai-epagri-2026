# -*- coding: utf-8 -*-
"""Dados finais do curso: somente estações internas à bacia escolhida.
Executar após os scripts 04 e 05, no ambiente epagri2026.
Não faz downloads nem preenchimento; agregação diária opcional. Originais preservados.
Diários: ANA, consistência exatamente 2 (chuva, vazão e nível disponíveis).
Horários: chuva, nível e vazão disponíveis, conforme seleção anterior.
"""
# %% 1. Importar bibliotecas
from pathlib import Path
from datetime import datetime
import json
import re
import shutil
import unicodedata
import pandas as pd
import geopandas as gpd

# %% 2. Escolher bacia e configurar caminhos
# Alterar apenas BACIA para selecionar outro conjunto gerado pelo script 04.
BASE = Path(r'C:\Users\afrod\OneDrive\Documents\HydroUAI_git\hydrouai-epagri-2026\02_Dados\BD01\BD01')
BACIA = 'bacia_01'  # Ex.: 'bacia_03'.
PASTA_ESTUDOS = None  # None: localizar a última execução completa do script 04.
SAIDA = BASE / f'Dados_Finais_Curso_{BACIA}' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
# Sem datas de deslocamento, um código com posições dentro E fora não pode
# ser atribuído integralmente à bacia. False: excluir e registrar para revisão.
INCLUIR_CODIGOS_COM_POSICOES_DENTRO_E_FORA = False
# None: preservar o período de cada série. Ex.: ('2013-01-01', '2023-12-31').
PERIODO_CURSO = None
CONVERTER_HORARIOS_EM_DIARIOS = True  # True: gerar diários adicionais; horários preservados.
MIN_HORAS_VALIDAS_DIA = 24  # 24: só dias completos; reduzir permite dias parciais.
# Chuva: soma; nível/vazão: média. Usar o fuso original, sem deslocar horários.
# Séries derivadas não recebem consistência ANA 2; origem indicada no catálogo.


# %% 3. Funções de preparação
def norm(x):
    return re.sub('[^a-z0-9]', '', unicodedata.normalize('NFKD', str(x)).encode('ascii', 'ignore').decode().lower())


def codigo(x):
    if pd.isna(x): return ''
    return re.sub(r'\.0$', '', str(x).strip())


def ler(path):
    return pd.read_csv(path, sep=';', dtype={'codigo_estacao': str, 'codigo': str, 'serie_id': str})


def salvar(df, path):
    pd.DataFrame(df).to_csv(path, sep=';', index=False, encoding='utf-8-sig')


def raiz_estudos():
    if PASTA_ESTUDOS is not None:
        root = Path(PASTA_ESTUDOS)
    else:
        candidates = sorted((p for p in (BASE / f'Estudos_{BACIA}').glob('*') if p.is_dir()
                             and (p / 'criterios.json').exists()
                             and (p / '01_diarios_ANA_consistencia_2' / 'catalogo_series.csv').exists()
                             and (p / '02_horarios_precipitacao_nivel' / 'catalogo_series.csv').exists()), reverse=True)
        root = next((p for p in candidates if json.loads((p / 'criterios.json').read_text(encoding='utf-8')).get('bacia') == BACIA), None)
        if root is None:
            raise FileNotFoundError(f'Nenhum estudo completo de {BACIA}. Execute o script 04 ou ajuste PASTA_ESTUDOS.')
    criteria = json.loads((root / 'criterios.json').read_text(encoding='utf-8'))
    if criteria.get('bacia') != BACIA:
        raise ValueError('A pasta de estudos não corresponde à bacia selecionada.')
    return root, criteria


def nivel_consistencia(row):
    for col in ['nivel_consistencia', 'nivel_consistencia_selecionado']:
        value = pd.to_numeric(row.get(col), errors='coerce')
        if pd.notna(value): return value
    match = re.search(r'(?i)\[\s*consist[eê]ncia\s*=\s*(\d+(?:\.\d+)?)\s*\]', str(row.get('variavel', '')))
    return float(match.group(1)) if match else float('nan')


def carregar_catalogos(root, pendencias):
    entries = []
    for study, resolution in [('01_diarios_ANA_consistencia_2', 'diaria'), ('02_horarios_precipitacao_nivel', 'horaria')]:
        folder = root / study
        catalog = ler(folder / 'catalogo_series.csv')
        for _, row in catalog.iterrows():
            r = row.to_dict()
            r.update(origem_catalogo=str(folder / 'catalogo_series.csv'), arquivo_origem=str(folder / str(r['arquivo_estudo'])), grupo_final='diarios' if resolution == 'diaria' else 'horarios')
            entries.append(r)
    # Cada estação de chuva baixada pode ter vários downloads. Escolher o mais
    # recente com dias válidos, sem concatenar versões ou repetir observações.
    downloaded = {}
    downloads = root / '01_diarios_ANA_consistencia_2' / 'precipitacao_ANA_downloads'
    for path in sorted(downloads.glob('*/catalogo_precipitacao.csv')):
        try:
            catalog = ler(path)
        except pd.errors.EmptyDataError:
            continue
        for _, row in catalog.iterrows():
            r = row.to_dict()
            station = codigo(r.get('codigo_estacao'))
            if pd.to_numeric(r.get('dias_validos', 0), errors='coerce') <= 0:
                continue
            source = root / '01_diarios_ANA_consistencia_2' / str(r['arquivo_estudo'])
            if not source.exists():
                pendencias.append({'codigo_estacao': station, 'motivo': 'Arquivo de download ausente; versão ignorada.', 'arquivo': str(source)})
                continue
            r.update(origem_catalogo=str(path), arquivo_origem=str(source), grupo_final='diarios')
            downloaded[station] = r
    # Quando houver chuva ANA já no catálogo diário, preferir a versão baixada
    # recentemente para o mesmo código, evitando duplicar esse conjunto.
    entries = [r for r in entries if not (r['grupo_final'] == 'diarios'
               and norm(r.get('fonte')) == 'ana' and norm(r.get('tipo')) == 'chuva'
               and codigo(r.get('codigo_estacao')) in downloaded)]
    entries.extend(downloaded.values())
    if not downloaded:
        pendencias.append({'codigo_estacao': '', 'motivo': 'Nenhum download de chuva ANA com dias válidos foi localizado. Conferir execução do script 05.', 'arquivo': str(downloads)})
    return entries


def coletar_posicoes(root, criteria, entries):
    frames = []
    global_pos = Path(criteria.get('diagnostico_origem', '')) / 'posicoes_estacoes.csv'
    for path in [global_pos, root / 'estacoes_posicoes.csv']:
        if path.exists(): frames.append(ler(path))
    for entry in entries:
        metadata = Path(entry['arquivo_origem']).with_suffix('.json')
        if metadata.exists():
            obj = json.loads(metadata.read_text(encoding='utf-8'))
            rows = obj.get('posicoes', [])
            if rows: frames.append(pd.DataFrame(rows))
    for path in (root / '01_diarios_ANA_consistencia_2' / 'precipitacao_ANA_downloads').glob('*/estacoes_selecionadas.csv'):
        frames.append(ler(path))
    if not frames: raise ValueError('Nenhuma tabela de coordenadas localizada.')
    positions = pd.concat(frames, ignore_index=True)
    if not all(c in positions for c in ['codigo', 'longitude', 'latitude']):
        raise ValueError('Tabelas de posições precisam de codigo, longitude e latitude.')
    positions['codigo'] = positions.codigo.map(codigo)
    positions['longitude'] = pd.to_numeric(positions.longitude, errors='coerce')
    positions['latitude'] = pd.to_numeric(positions.latitude, errors='coerce')
    positions = positions.dropna(subset=['longitude', 'latitude'])
    positions = positions[positions.codigo.ne('') & positions.longitude.between(-180, 180) & positions.latitude.between(-90, 90)]
    positions = positions.drop_duplicates(['codigo', 'longitude', 'latitude']).copy()
    cols = [c for c in ['codigo', 'nome', 'longitude', 'latitude', 'fonte_posicao'] if c in positions]
    return positions[cols]


def selecionar_internas(positions, polygon):
    p = gpd.GeoDataFrame(positions.copy(), geometry=gpd.points_from_xy(positions.longitude, positions.latitude), crs=4326)
    p['dentro_bacia'] = p.geometry.map(polygon.covers)
    inside = set(p.loc[p.dentro_bacia, 'codigo'])
    outside = set(p.loc[~p.dentro_bacia, 'codigo'])
    ambiguous = inside & outside
    accepted = inside if INCLUIR_CODIGOS_COM_POSICOES_DENTRO_E_FORA else inside - ambiguous
    return p, accepted, ambiguous


def agregar_diario(data, tipo):
    """Uma observação por hora; duplicatas concordantes não contam duas vezes."""
    tempos = pd.to_datetime(data.data_hora, format='mixed', errors='raise')
    amostra = pd.DataFrame({'tempo': tempos, 'valor': pd.to_numeric(data.valor, errors='coerce')})
    # Para timestamps repetidos, divergências tornam a hora ausente.
    grupos = amostra.groupby('tempo').valor
    conflitos = grupos.nunique(dropna=True).gt(1)
    horas = grupos.first().sort_index()
    horas.loc[conflitos[conflitos].index] = float('nan')
    # Não converter resolução sub-horária silenciosamente.
    if horas.index.floor('h').duplicated().any():
        raise ValueError('Mais de um timestamp distinto por hora; conferir a resolução antes da agregação.')
    contagem = horas.resample('D').count()
    valores = horas.resample('D').sum(min_count=1) if tipo == 'chuva' else horas.resample('D').mean()
    valores = valores.where(contagem.ge(MIN_HORAS_VALIDAS_DIA))
    diaria = pd.DataFrame({'data_hora': valores.index.strftime('%Y-%m-%d'),
                          'valor': valores.to_numpy(), 'horas_validas': contagem.to_numpy(),
                          'dia_completo': contagem.eq(24).to_numpy()})
    conflitos_dia = conflitos[conflitos].resample('D').sum().reindex(valores.index, fill_value=0)
    diaria['horas_conflitantes'] = conflitos_dia.to_numpy()
    return diaria


def main():
    if not re.fullmatch(r'bacia_\d+', BACIA):
        raise ValueError("BACIA deve ter formato como 'bacia_02'.")
    if CONVERTER_HORARIOS_EM_DIARIOS and not 1 <= MIN_HORAS_VALIDAS_DIA <= 24:
        raise ValueError('MIN_HORAS_VALIDAS_DIA deve estar entre 1 e 24.')
    root, criteria = raiz_estudos()
    boundary = gpd.read_file(root / 'bacia.gpkg')
    if boundary.crs is None: raise ValueError('Bacia sem CRS.')
    polygon = boundary.to_crs(4326).geometry.union_all()
    if not polygon.is_valid: raise ValueError('Geometria inválida; revisar limites antes da seleção.')
    if PERIODO_CURSO and pd.Timestamp(PERIODO_CURSO[1]) < pd.Timestamp(PERIODO_CURSO[0]):
        raise ValueError('PERIODO_CURSO: fim anterior ao início.')
    pendencias, catalog_final, rejected, frames = [], [], [], {'diarios': [], 'horarios': []}
    entries = carregar_catalogos(root, pendencias)
    positions = coletar_posicoes(root, criteria, entries)
    points, accepted, ambiguous = selecionar_internas(positions, polygon)
    selected = []
    seen = set()
    for r in entries:
        station = codigo(r.get('codigo_estacao'))
        typ, resolution, source = norm(r.get('tipo')), norm(r.get('resolucao')), norm(r.get('fonte'))
        reason = ''
        if station in ambiguous and station not in accepted:
            reason = 'Código com posições dentro e fora; datas de deslocamento não disponíveis.'
        elif station not in accepted:
            reason = 'Estação externa ou sem coordenada interna reconhecida.'
        elif r['grupo_final'] == 'diarios' and not (source == 'ana' and resolution == 'diaria' and nivel_consistencia(r) == 2 and typ in ['chuva', 'vazao', 'nivel']):
            reason = 'Não atende a ANA diária, consistência 2 e variável hidrológica.'
        elif r['grupo_final'] == 'horarios' and not (resolution == 'horaria' and typ in ['chuva', 'nivel', 'vazao']):
            reason = 'Não atende a chuva/nível/vazão horários.'
        if reason:
            rejected.append({'codigo_estacao': station, 'tipo': typ, 'grupo': r['grupo_final'], 'motivo': reason, 'arquivo': r['arquivo_origem']})
            continue
        key = (r['grupo_final'], str(Path(r['arquivo_origem']).resolve()))
        if key in seen: continue
        seen.add(key)
        if not Path(r['arquivo_origem']).is_file():
            raise FileNotFoundError(f'Arquivo selecionado não encontrado: {r["arquivo_origem"]}')
        r['codigo_estacao'] = station
        selected.append(r)
    if not selected: raise ValueError('Nenhuma série interna atende aos critérios. Conferir coordenadas, catálogos e bacia.')
    SAIDA.mkdir(parents=True, exist_ok=False)
    shutil.copy2(root / 'bacia.gpkg', SAIDA / 'bacia.gpkg')
    for r in selected:
        station, typ, group = r['codigo_estacao'], norm(r['tipo']), r['grupo_final']
        data = ler(Path(r['arquivo_origem']))
        if not all(k in data for k in ['data_hora', 'valor']):
            raise ValueError(f'Dados sem data_hora/valor: {r["arquivo_origem"]}')
        original = len(data)
        times = pd.to_datetime(data.data_hora, errors='coerce')
        invalid = int(times.isna().sum())
        if invalid:
            pendencias.append({'codigo_estacao': station, 'motivo': f'{invalid} datas inválidas excluídas do arquivo final.', 'arquivo': r['arquivo_origem']})
        mask = times.notna()
        if PERIODO_CURSO:
            a, b = (pd.Timestamp(x).normalize() for x in PERIODO_CURSO)
            mask &= times.ge(a) & times.lt(b + pd.Timedelta('1d'))
        data = data.loc[mask].copy()
        if data.empty:
            rejected.append({'codigo_estacao': station, 'tipo': typ, 'grupo': group, 'motivo': 'Sem registros no período escolhido.', 'arquivo': r['arquivo_origem']})
            continue
        data['data_hora'] = times.loc[mask].dt.strftime('%Y-%m-%d') if group == 'diarios' else times.loc[mask].astype(str)
        data['valor'] = pd.to_numeric(data.valor, errors='coerce')
        data['codigo_estacao'], data['tipo'], data['fonte'], data['resolucao'] = station, typ, r['fonte'], r['resolucao']
        data['serie_id'] = str(r.get('serie_id', Path(r['arquivo_origem']).stem))
        data['nivel_consistencia'] = 2 if group == 'diarios' else float('nan')
        data['relacao_codigo'] = 'interna'
        sp = points[points.codigo.eq(station) & points.dentro_bacia].drop(columns='geometry')
        data['numero_posicoes_internas'] = len(sp)
        data['posicao_espacial_ambigua'] = station in ambiguous
        data['longitude'] = sp.iloc[0].longitude if len(sp) == 1 else float('nan')
        data['latitude'] = sp.iloc[0].latitude if len(sp) == 1 else float('nan')
        # Gerar nome legível também para arquivos antigos e downloads do script 05.
        nome = '_'.join([norm(r['fonte']).upper() or 'FONTE',
                         norm(station) or 'SEM_CODIGO', typ, norm(r['resolucao'])])
        if group == 'diarios': nome += '_c2'
        pasta_estacao = SAIDA / group / typ / station
        target = pasta_estacao / (nome + '.csv')
        numero = 2
        while target.exists() or target.with_suffix('.json').exists():
            target = pasta_estacao / f'{nome}_serie{numero:02d}.csv'
            numero += 1
        relative = target.relative_to(SAIDA)
        target.parent.mkdir(parents=True, exist_ok=True)
        salvar(data.sort_values('data_hora'), target)
        if Path(r['arquivo_origem']).with_suffix('.json').exists():
            shutil.copy2(Path(r['arquivo_origem']).with_suffix('.json'), target.with_name(target.stem + '_metadados_origem.json'))
        meta = {'bacia': BACIA, 'arquivo_origem': r['arquivo_origem'], 'arquivo_final': str(relative),
                'periodo_curso': PERIODO_CURSO, 'codigo_estacao': station, 'posicoes_internas': sp.to_dict('records'),
                'posicao_espacial_ambigua': station in ambiguous,
                'nota': 'Sem preenchimento, agregação ou fusão de arquivos/fontes. Posições alternativas não têm validade temporal atribuída.'}
        target.with_suffix('.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
        valid = data[data.valor.notna()]
        catalog_final.append({'grupo': group, 'tipo': typ, 'codigo_estacao': station, 'fonte': r['fonte'],
                              'serie_id': data.serie_id.iloc[0], 'nivel_consistencia': 2 if group == 'diarios' else None,
                              'inicio': data.data_hora.min(), 'fim': data.data_hora.max(), 'registros': len(data),
                              'registros_origem': original, 'valores_validos': len(valid),
                              'timestamps_duplicados': int(data.data_hora.duplicated(False).sum()),
                              'negativos': int((data.valor < 0).sum()), 'numero_posicoes_internas': len(sp),
                              'posicao_espacial_ambigua': station in ambiguous, 'arquivo_final': str(relative),
                              'arquivo_origem': r['arquivo_origem']})
        frames[group].append(data)
        if group == 'horarios' and CONVERTER_HORARIOS_EM_DIARIOS:
            derivada = agregar_diario(data, typ)
            sid_derivada = str(data.serie_id.iloc[0]) + '_diario_derivado'
            metodo = 'soma' if typ == 'chuva' else 'media'
            for col in ['codigo_estacao', 'tipo', 'fonte', 'longitude', 'latitude',
                        'numero_posicoes_internas', 'posicao_espacial_ambigua']:
                derivada[col] = data[col].iloc[0]
            derivada['resolucao'] = 'diaria'
            derivada['serie_id'] = sid_derivada
            derivada['nivel_consistencia'] = float('nan')
            derivada['relacao_codigo'] = 'interna'
            derivada['origem_resolucao'] = 'horaria_agregada'
            derivada['metodo_agregacao'] = metodo
            nome_d = f'{norm(r["fonte"]).upper()}_{norm(station)}_{typ}_diaria_derivada_horaria'
            destino_d = SAIDA / 'diarios' / typ / station / (nome_d + '.csv')
            numero = 2
            while destino_d.exists() or destino_d.with_suffix('.json').exists():
                destino_d = destino_d.with_name(f'{nome_d}_serie{numero:02d}.csv')
                numero += 1
            destino_d.parent.mkdir(parents=True, exist_ok=True)
            relativo_d = destino_d.relative_to(SAIDA)
            salvar(derivada, destino_d)
            meta_d = {**meta, 'arquivo_final': str(relativo_d), 'serie_id': sid_derivada,
                      'serie_id_horaria_origem': str(data.serie_id.iloc[0]),
                      'arquivo_horario_final': str(relative), 'resolucao': 'diaria',
                      'origem_resolucao': 'horaria_agregada', 'metodo_agregacao': metodo,
                      'min_horas_validas_dia': MIN_HORAS_VALIDAS_DIA,
                      'nota': 'Agregação por dia no fuso original. Duplicatas concordantes contam uma vez; horas conflitantes são ausentes. Dias insuficientes ficam NaN. Não houve preenchimento.'}
            destino_d.with_suffix('.json').write_text(json.dumps(meta_d,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
            validas = derivada.valor.notna()
            catalog_final.append({'grupo': 'diarios', 'tipo': typ, 'codigo_estacao': station,
                'fonte': r['fonte'], 'serie_id': sid_derivada, 'nivel_consistencia': None,
                'inicio': derivada.data_hora.min(), 'fim': derivada.data_hora.max(),
                'registros': len(derivada), 'registros_origem': len(data),
                'valores_validos': int(validas.sum()), 'timestamps_duplicados': 0,
                'negativos': int(derivada.valor.lt(0).sum()), 'numero_posicoes_internas': len(sp),
                'posicao_espacial_ambigua': station in ambiguous, 'arquivo_final': str(relativo_d),
                'arquivo_origem': str(target), 'origem_resolucao': 'horaria_agregada',
                'metodo_agregacao': metodo, 'min_horas_validas_dia': MIN_HORAS_VALIDAS_DIA,
                'dias_completos': int(derivada.dia_completo.sum())})
            frames['diarios'].append(derivada)
    if not catalog_final:
        raise ValueError('Nenhuma observação no período escolhido. Ajuste PERIODO_CURSO.')
    catalog = pd.DataFrame(catalog_final)
    if 'origem_resolucao' not in catalog:
        catalog['origem_resolucao'] = 'nativa'
    else:
        catalog['origem_resolucao'] = catalog.origem_resolucao.fillna('nativa')
    summary = catalog.groupby(['grupo', 'tipo']).agg(codigos=('codigo_estacao', 'nunique'), series=('serie_id', 'nunique'), registros=('registros', 'sum'), valores_validos=('valores_validos', 'sum')).reset_index()
    horarias = catalog[catalog.grupo.eq('horarios')]
    disponibilidade = []
    for code, grupo in horarias.groupby('codigo_estacao', sort=True):
        tipos = set(grupo.tipo)
        disponibilidade.append({'codigo_estacao': code, 'tem_chuva': 'chuva' in tipos,
            'tem_nivel': 'nivel' in tipos, 'tem_vazao': 'vazao' in tipos,
            'tem_tres_variaveis': {'chuva','nivel','vazao'}.issubset(tipos)})
    disponiveis = pd.DataFrame(disponibilidade, columns=['codigo_estacao','tem_chuva',
        'tem_nivel','tem_vazao','tem_tres_variaveis'])
    salvar(disponiveis, SAIDA / 'disponibilidade_variaveis_horarias.csv')
    selected_codes = set(catalog.codigo_estacao)
    internal_points = points[points.codigo.isin(selected_codes) & points.dentro_bacia].copy()
    salvar(internal_points.drop(columns='geometry'), SAIDA / 'coordenadas_estacoes.csv')
    internal_points.to_file(SAIDA / 'estacoes_internas.gpkg', driver='GPKG')
    salvar(catalog, SAIDA / 'catalogo_dados_finais.csv')
    salvar(summary, SAIDA / 'resumo_dados_finais.csv')
    salvar(pd.DataFrame(rejected, columns=['codigo_estacao', 'tipo', 'grupo', 'motivo', 'arquivo']), SAIDA / 'series_excluidas.csv')
    salvar(pd.DataFrame(pendencias, columns=['codigo_estacao', 'motivo', 'arquivo']), SAIDA / 'pendencias.csv')
    for group, tables in frames.items():
        directory = SAIDA / group
        directory.mkdir(exist_ok=True)
        cols = ['data_hora', 'valor', 'codigo_estacao', 'tipo', 'fonte', 'resolucao', 'serie_id', 'nivel_consistencia', 'longitude', 'latitude']
        combined = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=cols)
        salvar(combined, directory / 'dados_consolidados.csv')
    body = '<!doctype html><meta charset="utf-8"><title>Dados finais do curso</title><style>body{font:14px Arial;max-width:1300px;margin:30px auto}table{border-collapse:collapse;display:block;overflow:auto}td,th{border:1px solid #ccc;padding:7px}</style>' + f'<h1>Dados finais — {BACIA}</h1>'
    body += '<p>Somente estações internas. Diários ANA: consistência 2. Horários: chuva, nível e vazão disponíveis. Agregação diária opcional; sem preenchimento.</p>'
    body += f'<p>Gerar diários a partir de horários: {CONVERTER_HORARIOS_EM_DIARIOS}. Mínimo de horas válidas por dia: {MIN_HORAS_VALIDAS_DIA}. Chuva: soma; nível e vazão: média. Diários derivados têm origem identificada, sem atribuição de consistência ANA.</p>'
    body += '<h2>Resumo</h2>' + summary.to_html(index=False)
    body += '<h2>Variáveis horárias por estação</h2>' + disponiveis.to_html(index=False)
    body += '<p>Estações com uma ou duas variáveis são mantidas. Conferir a sobreposição temporal antes de utilizar as três variáveis conjuntamente.</p>'
    body += '<h2>Catálogo</h2>' + catalog.drop(columns='arquivo_origem').to_html(index=False, escape=True)
    body += '<h2>Exclusões</h2>' + pd.DataFrame(rejected).to_html(index=False, escape=True)
    body += '<h2>Pendências</h2>' + pd.DataFrame(pendencias).to_html(index=False, escape=True)
    (SAIDA / 'relatorio_dados_finais.html').write_text(body, encoding='utf-8')
    config = {'bacia': BACIA, 'pasta_estudos_origem': str(root), 'periodo_curso': PERIODO_CURSO,
              'converter_horarios_em_diarios': CONVERTER_HORARIOS_EM_DIARIOS,
              'min_horas_validas_dia': MIN_HORAS_VALIDAS_DIA,
              'incluir_codigos_com_posicoes_dentro_e_fora': INCLUIR_CODIGOS_COM_POSICOES_DENTRO_E_FORA,
              'selecao': 'Interior e borda do polígono (covers). Sem faixa externa.'}
    (SAIDA / 'configuracao.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    (SAIDA / 'LEIA_ME.txt').write_text(f'DADOS FINAIS PARA O CURSO — {BACIA}\n\nDiários: ANA, consistência 2, chuva/vazão/nível disponíveis.\nHorários: chuva, nível e vazão disponíveis. Somente códigos com coordenadas internas.\nAs pastas diarios e horarios contêm arquivos por variável e código, além de\num consolidado em formato longo. Fonte e serie_id distinguem séries.\nNão somar arquivos/fontes sobrepostos nem confundir duplicatas com observações\nindependentes. Duplicatas, lacunas e valores negativos foram preservados para\ntratamento no curso. Não houve preenchimento. Agregação diária opcional identificada na origem.\nChuva do script 05: último download com dias válidos para cada estação.\nCoordenadas alternativas permanecem no catálogo e nos metadados.\nPor padrão, códigos com posições internas e externas foram excluídos para revisão.\nOs percentuais e os períodos devem ser avaliados antes de escolher treinamento\ne validação. Consultar catálogo, relatório, exclusões e pendências.\nObservar as condições de uso das bases ANA e EPAGRI.\n', encoding='utf-8')
    with (SAIDA / 'LEIA_ME.txt').open('a', encoding='utf-8') as leia:
        leia.write(f'\nDiários derivados dos horários: {CONVERTER_HORARIOS_EM_DIARIOS}.\nChuva: soma; nível/vazão: média. Mínimo de horas válidas: {MIN_HORAS_VALIDAS_DIA}.\nSéries derivadas têm origem_resolucao=horaria_agregada e não recebem consistência ANA 2.\nO dia utiliza o fuso original das séries.\n')
    print(summary.to_string(index=False))
    print(f'\nPasta final: {SAIDA}\nAbrir relatorio_dados_finais.html.')


# %% 4. Executar preparação dos dados finais
if __name__ == '__main__':
    main()
