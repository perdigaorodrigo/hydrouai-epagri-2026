# -*- coding: utf-8 -*-
"""Separar estudos da bacia escolhida a partir do diagnóstico geral EPAGRI.
Executar no ambiente epagri2026, depois de 03_diagnostico_bacias_epagri.py.
Não altera os dados de origem, não agrega e não preenche séries.
"""
# %% 1. Importar bibliotecas
from pathlib import Path
from datetime import datetime
import json
import re
import shutil
import unicodedata
import pandas as pd

# %% 2. Escolher a bacia e configurar os diretórios
# Após executar a célula 1, alterar apenas BACIA para escolher outra bacia.
BASE = Path(r'C:\Users\afrod\OneDrive\Documents\HydroUAI_git\hydrouai-epagri-2026\02_Dados\BD01\BD01')
BACIA = 'bacia_01'  # Ex.: 'bacia_03'; controla leitura, saída e relatório.
# None: localizar a última execução com catálogo e produtos da bacia.
# Também pode informar a pasta exata: Path(r'C:\...\20261003_...').
PASTA_DIAGNOSTICO = None
# Preservar a seleção espacial do diagnóstico, incluindo externas da faixa X km.
INCLUIR_EXTERNAS_PROXIMAS = True
SAIDA = BASE / f'Estudos_{BACIA}' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')


# %% 3. Definir as funções de seleção e exportação
def norm(value):
    text = unicodedata.normalize('NFKD', str(value)).encode('ascii', 'ignore').decode()
    return re.sub(r'[^a-z0-9]', '', text.lower())


def selecionar(catalogo):
    """Critérios de seleção explícitos; consistência ausente não é aceita."""
    c = catalogo.copy()
    obrigatorias = ['serie_id', 'fonte', 'tipo', 'resolucao', 'variavel', 'codigo_estacao', 'relacao_codigo']
    missing = [k for k in obrigatorias if k not in c]
    if missing:
        raise ValueError('Catálogo incompatível. Executar script 03. Faltam: ' + ', '.join(missing))
    if c.serie_id.duplicated().any():
        raise ValueError('Catálogo contém IDs repetidos. Conferir diagnóstico antes de separar os dados.')
    level = c.variavel.fillna('').str.extract(r'(?i)\[\s*consist[eê]ncia\s*=\s*(\d+(?:\.\d+)?)\s*\]', expand=False)
    c['nivel_consistencia_selecionado'] = pd.to_numeric(level, errors='coerce')
    # Nomes canônicos para os diretórios; os arquivos originais são copiados intactos.
    c['tipo'] = c.tipo.map(norm)
    eligible = pd.Series(True, index=c.index)
    if not INCLUIR_EXTERNAS_PROXIMAS:
        eligible &= c.relacao_codigo.eq('interna')
    diaria = eligible & c.fonte.map(norm).eq('ana') & c.resolucao.map(norm).eq('diaria') & c.nivel_consistencia_selecionado.eq(2)
    horaria = eligible & c.resolucao.map(norm).eq('horaria') & c.tipo.map(norm).isin(['chuva', 'nivel', 'vazao'])
    return c.loc[diaria].copy(), c.loc[horaria].copy(), c.loc[~(diaria | horaria)].copy()


def encontrar_diagnostico():
    if PASTA_DIAGNOSTICO is not None:
        root = Path(PASTA_DIAGNOSTICO)
        if not (root / BACIA / 'series.csv').exists():
            raise FileNotFoundError(f'Não encontrado: {root / BACIA / "series.csv"}')
        return root
    directory = BASE / 'Diagnostico_Geral_Epagri'
    candidates = sorted((p for p in directory.glob('*') if p.is_dir()
                         and (p / 'catalogo_series.csv').exists()
                         and (p / BACIA / 'series.csv').exists()
                         and (p / 'relatorio_geral.html').exists()), reverse=True)
    if not candidates:
        raise FileNotFoundError(f'Nenhum diagnóstico completo com {BACIA} em {directory}. Execute o script 03 ou ajuste PASTA_DIAGNOSTICO.')
    return candidates[0]


def salvar(df, path):
    df.to_csv(path, sep=';', index=False, encoding='utf-8-sig')


def copiar_grupo(catalog, study, root, posicoes):
    folder = SAIDA / study
    folder.mkdir()
    catalog = catalog.copy()
    catalog['arquivo_estudo'] = ''
    catalog['nome_arquivo_origem'] = ''
    pos = posicoes[posicoes.codigo.isin(catalog.codigo_estacao)].copy()
    salvar(pos, folder / 'coordenadas_estacoes.csv')
    frames = []
    for i, r in catalog.iterrows():
        candidate = r.get('arquivo_por_bacia', '')
        if pd.notna(candidate) and str(candidate).strip():
            source = root / BACIA / str(candidate)
        else:
            source = root / str(r.arquivo_separado)
        if not source.exists():
            raise FileNotFoundError(f'Arquivo da série não encontrado: {source}. Manter o diagnóstico completo.')
        # Nome legível; IDs técnicos continuam no catálogo para rastreabilidade.
        nome = '_'.join([norm(r.fonte).upper() or 'FONTE',
                         norm(r.codigo_estacao) or 'SEM_CODIGO',
                         norm(r.tipo), norm(r.resolucao)])
        if pd.notna(r.nivel_consistencia_selecionado):
            nome += f'_c{int(r.nivel_consistencia_selecionado)}'
        destino_estacao = folder / 'dados' / r.tipo / r.codigo_estacao
        target = destino_estacao / (nome + source.suffix)
        numero = 2
        while target.exists() or target.with_suffix('.json').exists():
            target = destino_estacao / f'{nome}_serie{numero:02d}{source.suffix}'
            numero += 1
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if source.with_suffix('.json').exists():
            shutil.copy2(source.with_suffix('.json'), target.with_suffix('.json'))
        catalog.loc[i, 'arquivo_estudo'] = str(target.relative_to(folder))
        catalog.loc[i, 'nome_arquivo_origem'] = source.name
        frame = pd.read_csv(source, sep=';', dtype={'codigo_estacao': str})
        frame['serie_id'] = r.serie_id
        frame['tipo'] = r.tipo
        frame['nivel_consistencia'] = r.nivel_consistencia_selecionado
        frame['relacao_codigo'] = r.relacao_codigo
        station_positions = pos[pos.codigo.eq(r.codigo_estacao)]
        frame['numero_posicoes_conhecidas'] = len(station_positions)
        # Uma posição única pode ser inserida na tabela; alternativas não
        # multiplicam observações nem recebem datas de validade inventadas.
        frame['longitude'] = station_positions.iloc[0].longitude if len(station_positions) == 1 else float('nan')
        frame['latitude'] = station_positions.iloc[0].latitude if len(station_positions) == 1 else float('nan')
        frames.append(frame)
    columns = ['data_hora', 'valor', 'codigo_estacao', 'fonte', 'variavel', 'resolucao', 'serie_id', 'tipo', 'nivel_consistencia', 'relacao_codigo', 'numero_posicoes_conhecidas', 'longitude', 'latitude']
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    salvar(combined, folder / 'dados_consolidados.csv')
    salvar(catalog, folder / 'catalogo_series.csv')
    summary = catalog.groupby(['codigo_estacao', 'fonte', 'tipo', 'resolucao', 'relacao_codigo'], dropna=False).agg(series=('serie_id', 'nunique')).reset_index()
    salvar(summary, folder / 'resumo_estacoes.csv')
    return dict(estudo=study, series=len(catalog), codigos=catalog.codigo_estacao.nunique(), registros=len(combined), chuva_codigos=catalog.loc[catalog.tipo.eq('chuva'), 'codigo_estacao'].nunique(), nivel_codigos=catalog.loc[catalog.tipo.eq('nivel'), 'codigo_estacao'].nunique(), vazao_codigos=catalog.loc[catalog.tipo.eq('vazao'), 'codigo_estacao'].nunique())


def main():
    if not re.fullmatch(r'bacia_\d+', BACIA):
        raise ValueError("BACIA deve ter formato como 'bacia_02'.")
    root = encontrar_diagnostico()
    print(f'Diagnóstico selecionado: {root}')
    catalog = pd.read_csv(root / BACIA / 'series.csv', sep=';', dtype={'codigo_estacao': str, 'serie_id': str})
    positions = pd.read_csv(root / BACIA / 'estacoes_posicoes.csv', sep=';', dtype={'codigo': str})
    daily, hourly, excluded = selecionar(catalog)
    # Conferir todos os caminhos antes de iniciar a exportação.
    for _, r in pd.concat([daily, hourly]).iterrows():
        relative = r.get('arquivo_por_bacia', '')
        source = root / BACIA / str(relative) if pd.notna(relative) and str(relative).strip() else root / str(r.arquivo_separado)
        if not source.is_file():
            raise FileNotFoundError(f'Série não encontrada: {source}')
    SAIDA.mkdir(parents=True, exist_ok=False)
    summaries = [copiar_grupo(daily, '01_diarios_ANA_consistencia_2', root, positions),
                 copiar_grupo(hourly, '02_horarios_precipitacao_nivel', root, positions)]
    # Nome da pasta horária mantido para compatibilidade com os scripts anteriores;
    # dentro de dados agora há chuva/, nivel/ e vazao/, conforme disponíveis.
    disponibilidade = []
    for codigo, grupo in hourly.groupby('codigo_estacao', sort=True):
        tipos = set(grupo.tipo)
        disponibilidade.append({'codigo_estacao': codigo,
            'tem_chuva': 'chuva' in tipos, 'tem_nivel': 'nivel' in tipos,
            'tem_vazao': 'vazao' in tipos,
            'tem_tres_variaveis': {'chuva', 'nivel', 'vazao'}.issubset(tipos)})
    disponiveis = pd.DataFrame(disponibilidade, columns=['codigo_estacao', 'tem_chuva',
        'tem_nivel', 'tem_vazao', 'tem_tres_variaveis'])
    salvar(disponiveis, SAIDA / 'disponibilidade_variaveis_horarias.csv')
    summary = pd.DataFrame(summaries)
    salvar(summary, SAIDA / 'resumo_estudos.csv')
    salvar(excluded, SAIDA / 'series_nao_selecionadas.csv')
    for filename in ['bacia.gpkg', 'estacoes.gpkg', 'mapa.png', 'estacoes_posicoes.csv']:
        source = root / BACIA / filename
        if source.exists():
            shutil.copy2(source, SAIDA / filename)
    notes = {'bacia': BACIA, 'diagnostico_origem': str(root), 'incluir_externas_proximas': INCLUIR_EXTERNAS_PROXIMAS,
             'diarios': 'Fonte ANA, resolução diária, nível de consistência exatamente 2.',
             'horarios': 'Todas as fontes, resolução horária, tipos chuva, nível e vazão (quando disponíveis).',
             'observacoes': ['Resolução nativa preservada. Não há conversão de horário para diário.',
                             'Arquivos/fontes sobrepostos não são fundidos. IDs de série são preservados.',
                             'Sem atribuição automática de posição histórica: consultar coordenadas_estacoes.csv e JSONs.',
                             'Percentuais do catálogo usam períodos próprios; comparar em período comum na próxima análise.']}
    (SAIDA / 'criterios.json').write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding='utf-8')
    body = f'<!doctype html><meta charset="utf-8"><title>Estudos {BACIA}</title>' + '<style>body{font:15px Arial;max-width:1200px;margin:30px auto}table{border-collapse:collapse}th,td{border:1px solid #ccc;padding:8px}img{max-width:100%}</style>' + f'<h1>Estudos — {BACIA}</h1>'
    body += '<p>Diários: ANA, consistência 2. Horários: precipitação, nível e vazão disponíveis, sem restrição de fonte.</p>'
    body += '<p>Estações externas próximas incluídas: ' + str(INCLUIR_EXTERNAS_PROXIMAS) + '.</p>'
    body += summary.to_html(index=False)
    body += '<h2>Variáveis horárias por estação</h2>' + disponiveis.to_html(index=False)
    body += '<p>Estações com apenas uma ou duas variáveis também são mantidas. Ter as três variáveis não implica disponibilidade simultânea; conferir os períodos de cada série.</p>'
    for label, data in [('Diários ANA — consistência 2', daily), ('Horários — precipitação, nível e vazão', hourly)]:
        body += '<h2>' + label + '</h2>'
        columns = [c for c in ['codigo_estacao', 'fonte', 'tipo', 'variavel', 'resolucao', 'relacao_codigo', 'inicio', 'fim', 'disponibilidade_diaria_pct', 'maior_falha_dias', 'numero_posicoes_codigo'] if c in data]
        body += data[columns].to_html(index=False, escape=True, na_rep='—') if len(data) else '<p>Nenhuma série atende aos critérios; conferir o diagnóstico.</p>'
    if (SAIDA / 'mapa.png').exists():
        body += '<h2>Seleção espacial do diagnóstico</h2><img src="mapa.png">'
    body += '<p>Dados consolidados em formato longo. Série + data/hora identifica o registro candidato; duplicatas permanecem para diagnóstico. Coordenadas alternativas estão nas tabelas de posições. Não houve preenchimento, agregação ou recorte temporal.</p>'
    (SAIDA / 'relatorio_selecao.html').write_text(body, encoding='utf-8')
    print(summary.to_string(index=False))
    print(f'\nResultados: {SAIDA}\nAbrir relatorio_selecao.html no navegador.')


# %% 4. Executar a separação para a bacia escolhida
if __name__ == '__main__':
    main()
