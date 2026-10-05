# -*- coding: utf-8 -*-
"""Comparar Q diária e horária já prontas, quatro máximas e chuva antecedente.
Spyder: executar por células. Não recalcular as séries diárias.
"""
# %% 1. Bibliotecas
from pathlib import Path
import csv
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# %% 2. Arquivos, estação de vazão e estações de chuva
DIRETORIO_DADOS = Path(r'C:\Users\afrod\OneDrive\Documents\HydroUAI_git\hydrouai-epagri-2026\02_Dados\Dia 1\Dados_Preenchidos')
ARQUIVO_DIARIO = DIRETORIO_DADOS / 'series_diarias_preenchidas.csv'
ARQUIVO_HORARIO = DIRETORIO_DADOS / 'series_horarias_preenchidas.csv'
DIRETORIO_RESULTADOS = DIRETORIO_DADOS / 'Comparacao_Maximas_Q'
COLUNA_VAZAO = '2463_Q'
COLUNAS_CHUVA = ['2463_P', '84100000_P']
UNIDADE_Q = 'm³/s'  # Confirmar a unidade da fonte.
INICIO = None
FIM = None
# Eventos separados: a segunda maior amostra pode pertencer à mesma cheia.
SEPARACAO_EVENTOS_DIAS = 7  # Distância mínima entre os picos selecionados.
DIAS_ANTES_HIDROGRAMA = 5
DIAS_DEPOIS_HIDROGRAMA = 3
DIAS_CHUVA_ANTECEDENTE = 5  # 5 dias e 120 horas.
REFERENCIA_HORA = 'inicio'  # início ou fim do intervalo de chuva, conforme CSV.
EXIBIR_FIGURAS = True
SALVAR_FIGURAS = True
plt.rcParams.update({'figure.dpi': 110, 'font.size': 10, 'axes.grid': True,
                     'grid.alpha': .2, 'axes.spines.top': False, 'axes.spines.right': False})

# %% 3. Funções de apoio

def ler(arquivo, freq):
    with arquivo.open(encoding='utf-8-sig', newline='') as f:
        nomes = next(csv.reader(f, delimiter=';'))
    if len(nomes) != len(set(nomes)):
        raise ValueError('Colunas duplicadas no arquivo: ' + str(arquivo))
    df = pd.read_csv(arquivo, sep=';')
    datas = pd.to_datetime(df.pop('data_hora'), errors='raise')
    df.index = pd.DatetimeIndex(datas, name='data_hora')
    if df.index.tz is not None:
        raise ValueError('Usar CSVs da aula 1 com a mesma hora local, sem offset.')
    if df.index.has_duplicates or not df.index.equals(df.index.floor(freq)):
        raise ValueError('Datas repetidas ou fora da grade: ' + str(arquivo))
    df = df.sort_index()
    for c in df:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '.', regex=False), errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan)
    if INICIO is not None:
        df = df.loc[pd.Timestamp(INICIO):]
    if FIM is not None:
        limite = pd.Timestamp(FIM)
        if len(str(FIM)) == 10:
            limite += pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        df = df.loc[:limite]
    if df.empty:
        raise ValueError('Período sem dados.')
    return df.reindex(pd.date_range(df.index.min(), df.index.max(), freq=freq, name='data_hora'))


def figura(fig, nome):
    fig.tight_layout()
    if SALVAR_FIGURAS:
        fig.savefig(DIRETORIO_RESULTADOS / f'{nome}.png', dpi=180, bbox_inches='tight')
    if EXIBIR_FIGURAS:
        plt.show()
    else:
        plt.close(fig)


def maiores_eventos(y, n=4):
    # Seleção por magnitude, seguida de exclusão dos picos próximos.
    escolhidos = []
    for data, valor in y.dropna().sort_values(ascending=False, kind='stable').items():
        if all(abs(data - d) >= pd.Timedelta(days=SEPARACAO_EVENTOS_DIAS) for d, _ in escolhidos):
            escolhidos.append((data, float(valor)))
            if len(escolhidos) == n:
                break
    return pd.DataFrame([{'ordem': i+1, 'data_pico': d, 'Q_max': q}
                         for i, (d, q) in enumerate(escolhidos)])


def hidro(evento, qd, qh, ordem):
    dia = evento['data_pico'].normalize()
    inicio = dia - pd.Timedelta(days=max(DIAS_ANTES_HIDROGRAMA, DIAS_CHUVA_ANTECEDENTE))
    fim = dia + pd.Timedelta(days=DIAS_DEPOIS_HIDROGRAMA + 1)
    hd = qd.loc[(qd.index >= inicio) & (qd.index < fim)]
    hh = qh.loc[(qh.index >= inicio) & (qh.index < fim)]
    # Mesmo eixo de tempo nos três painéis: acumulados horários, diários e Q.
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                             gridspec_kw={'height_ratios': [1, 1, 1.6]})
    cores = plt.get_cmap('tab10')
    gh = pd.date_range(inicio, fim, freq='h', inclusive='left')
    gd = pd.date_range(inicio, fim, freq='D', inclusive='left')
    # Ambos os acumulados começam no mesmo instante, 00:00 do dia inicial.
    # Exibir chuva até depois do pico para acompanhar o desenvolvimento da cheia.
    for i, coluna in enumerate(COLUNAS_CHUVA):
        ph = horarios[coluna].copy()
        if REFERENCIA_HORA == 'fim':
            ph.index -= pd.Timedelta(hours=1)
        ph = ph.reindex(gh)
        pdia = diarios[coluna].reindex(gd)
        for ax, p, passo, arquivo in [(axes[0], ph, pd.Timedelta(hours=1), 'horaria'),
                                      (axes[1], pdia, pd.Timedelta(days=1), 'diaria')]:
            acumulada = p.cumsum(skipna=False)
            # Mostrar os totais somente ao FIM do intervalo: o total diário de D
            # só está completo às 00:00 de D+1, e não no início de D.
            tempos = p.index + passo
            ax.step([inicio] + list(tempos), [0.] + list(acumulada), where='post',
                    color=cores(i % 10), lw=1.4, label=coluna)
            pd.DataFrame({'P_intervalo_mm': p.to_numpy(),
                          'P_acumulada_mm': acumulada.to_numpy()}, index=tempos).to_csv(
                DIRETORIO_RESULTADOS / f'evento_{ordem}_{coluna}_{arquivo}_janela_hidrograma.csv',
                sep=';', index_label='fim_intervalo')
    axes[0].set_ylabel('P acumulada\nhorária (mm)')
    axes[1].set_ylabel('P acumulada\ndiária (mm)')
    axes[0].set_title(f'Evento {ordem} — acumulados desde {inicio:%d/%m/%Y %H:%M}')
    axes[0].legend(loc='best')
    axes[1].legend(loc='best')
    axes[2].plot(hh.index, hh, color='#287c94', lw=1, label='Vazão horária')
    axes[2].plot(hd.index, hd, 'o-', color='#d55e00', lw=1.3, ms=4, label='Vazão diária (média do dia)')
    axes[2].set_ylabel(f'Vazão ({UNIDADE_Q})')
    axes[2].set_xlabel('Data e hora — alinhadas entre os painéis')
    axes[2].legend(loc='best')
    pico_h = pd.Timestamp(evento['data_pico_horario_no_dia'])
    for ax in axes:
        ax.axvline(pico_h, color='#333333', ls='--', lw=.9, label='Pico horário')
        ax.axvspan(dia, dia + pd.Timedelta(days=1), color='gray', alpha=.1)
    axes[2].legend(loc='best')
    axes[2].set_xlim(inicio, fim)
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%d/%m\n%H:%M'))
    fig.suptitle(f'{COLUNA_VAZAO} — chuva e resposta da vazão | pico diário {dia:%d/%m/%Y}')
    figura(fig, f'03_hidrograma_chuva_evento_{ordem}')
    hd.rename('Q_diaria').to_csv(DIRETORIO_RESULTADOS / f'evento_{ordem}_Q_diaria.csv', sep=';')
    hh.rename('Q_horaria').to_csv(DIRETORIO_RESULTADOS / f'evento_{ordem}_Q_horaria.csv', sep=';')


def chuva_antecedente(diario, horario, evento, ordem):
    dia = evento['data_pico'].normalize()
    pico_h = pd.Timestamp(evento['data_pico_horario_no_dia'])
    linhas = []
    for coluna in COLUNAS_CHUVA:
        # D-5 a D-1: cinco dias completos, excluindo o dia do pico diário.
        gd = pd.date_range(dia - pd.Timedelta(days=DIAS_CHUVA_ANTECEDENTE), periods=DIAS_CHUVA_ANTECEDENTE, freq='D')
        # 120 intervalos completos que terminam até o instante do pico horário.
        # Referência início: [pico-120h, pico); referência fim: (pico-120h, pico].
        primeiro = pico_h - pd.Timedelta(hours=24 * DIAS_CHUVA_ANTECEDENTE)
        if REFERENCIA_HORA == 'fim':
            primeiro += pd.Timedelta(hours=1)
        gh = pd.date_range(primeiro, periods=24 * DIAS_CHUVA_ANTECEDENTE, freq='h')
        pdia = diario[coluna].reindex(gd)
        phora = horario[coluna].reindex(gh)
        for p, resolucao in [(pdia, 'diaria'), (phora, 'horaria')]:
            # Ausência torna o acumulado posterior desconhecido; não assumir chuva zero.
            acumulada = p.cumsum(skipna=False)
            tabela = pd.DataFrame({'P_intervalo_mm': p, 'P_acumulada_mm': acumulada})
            tabela.to_csv(DIRETORIO_RESULTADOS / f'evento_{ordem}_{coluna}_{resolucao}_antecedente.csv', sep=';', index_label='data_hora')
            linhas.append({'evento': ordem, 'chuva': coluna, 'resolucao': resolucao,
                           'referencia': dia if resolucao == 'diaria' else pico_h,
                           'inicio_janela': p.index.min(), 'fim_janela': p.index.max(),
                           'n_esperado': len(p), 'n_valido': int(p.notna().sum()),
                           'P_total_mm': p.sum(min_count=len(p))})
    return linhas

# %% 4. Abrir as séries prontas — não agregar novamente
if REFERENCIA_HORA not in ['inicio', 'fim'] or SEPARACAO_EVENTOS_DIAS <= 0:
    raise ValueError('Conferir referência da hora e separação dos eventos.')
DIRETORIO_RESULTADOS.mkdir(parents=True, exist_ok=True)
diarios = ler(ARQUIVO_DIARIO, 'D')
horarios = ler(ARQUIVO_HORARIO, 'h')
for c in [COLUNA_VAZAO] + COLUNAS_CHUVA:
    if c not in diarios or c not in horarios:
        raise ValueError(f'{c}: precisa existir nos dois arquivos.')
qd, qh = diarios[COLUNA_VAZAO], horarios[COLUNA_VAZAO]
# Comparar apenas dias com Q diária e pelo menos uma Q horária válida.
# Não recalcular Q diária; esta operação apenas identifica dias coincidentes.
base_h = qh.copy()
if REFERENCIA_HORA == 'fim':
    base_h.index -= pd.Timedelta(hours=1)
contagem = base_h.resample('D').count()
dias_comuns = qd.dropna().index.intersection(contagem[contagem > 0].index)
qd_comum = qd.reindex(dias_comuns)
qh_comum = qh.loc[base_h.index.normalize().isin(dias_comuns)]
if qd_comum.empty:
    raise ValueError('Sem dias coincidentes com dados de vazão.')
print('Dias coincidentes:', len(dias_comuns), '|', dias_comuns.min(), 'a', dias_comuns.max())

# %% 5. Plotar a vazão horária individualmente
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(qh.index, qh, color='#287c94', lw=.8)
ax.set(title=f'{COLUNA_VAZAO} — horária', ylabel=f'Vazão ({UNIDADE_Q})', xlabel='Data')
figura(fig, '01_Q_horaria')

# %% 6. Plotar a vazão diária individualmente
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(qd.index, qd, color='#d55e00', lw=.9)
ax.set(title=f'{COLUNA_VAZAO} — diária', ylabel=f'Vazão ({UNIDADE_Q})', xlabel='Data')
figura(fig, '01_Q_diaria')

# %% 7. As quatro maiores vazões em cada resolução — eventos separados
maximas_diarias = maiores_eventos(qd_comum)
maximas_horarias = maiores_eventos(qh_comum)
maximas_diarias.to_csv(DIRETORIO_RESULTADOS / '02_quatro_maximas_diarias.csv', sep=';', index=False)
maximas_horarias.to_csv(DIRETORIO_RESULTADOS / '02_quatro_maximas_horarias.csv', sep=';', index=False)
print('DIÁRIAS:\n', maximas_diarias.to_string(index=False))
print('HORÁRIAS:\n', maximas_horarias.to_string(index=False))
# Rankings independentes podem identificar eventos diferentes nas duas resoluções.

# %% 8. Comparar cada máximo diário com o pico horário no mesmo dia
comparacoes = []
for _, evento in maximas_diarias.iterrows():
    dia = evento.data_pico.normalize()
    hh = qh.loc[base_h.index.normalize() == dia].dropna()
    pico_h = hh.idxmax()
    comparacoes.append({'evento': int(evento.ordem), 'data_pico': dia,
                       'Q_diaria': evento.Q_max, 'Q_pico_horario_no_dia': hh.max(),
                       'data_pico_horario_no_dia': pico_h, 'horas_validas_no_dia': len(hh),
                       'dia_horario_completo': len(hh) == 24,
                       'diferenca_pico_horario_menos_diaria': hh.max() - evento.Q_max,
                       'razao_pico_horario_diaria': hh.max() / evento.Q_max if evento.Q_max != 0 else np.nan})
eventos = pd.DataFrame(comparacoes)
eventos.to_csv(DIRETORIO_RESULTADOS / '03_comparacao_quatro_eventos.csv', sep=';', index=False)
print(eventos.to_string(index=False))

# %% 9. Chuva acumulada e hidrogramas juntos — dois maiores eventos
# Evento 1 é o maior Q DIÁRIO. Pico horário associado é buscado no mesmo dia.
for _, evento in eventos.head(2).iterrows():
    hidro(evento, qd, qh, int(evento.evento))

# %% 10. Chuva antecedente dos dois maiores eventos — 5 dias e 120 horas
# As janelas têm igual duração, porém referências diferentes: dia diário e pico horário.
# Por isso, seus totais não necessariamente coincidem. Zeros são conservados.
chuva_resumo = []
for _, evento in eventos.head(2).iterrows():
    chuva_resumo.extend(chuva_antecedente(diarios, horarios, evento, int(evento.evento)))
pd.DataFrame(chuva_resumo).to_csv(DIRETORIO_RESULTADOS / '04_resumo_chuva_antecedente.csv', sep=';', index=False)
print(pd.DataFrame(chuva_resumo).to_string(index=False))
print('Resultados:', DIRETORIO_RESULTADOS)
# Valores preenchidos e observados são analisados juntos, conforme CSVs finais.
