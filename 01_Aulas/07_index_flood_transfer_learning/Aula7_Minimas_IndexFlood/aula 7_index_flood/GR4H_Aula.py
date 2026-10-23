# %% 1. CONFIGURACOES — comece aqui (Ctrl+Enter executa a celula)
"""GR4H didatico para Spyder. Execute as celulas em ordem na mesma console.

As tabelas ficam no Explorador de Variaveis. A calibracao e a etapa demorada:
depois dela, reexecute apenas as celulas de graficos para explorar os resultados.

Versão simplificada para aula. Dependências: numpy, pandas, scipy,
matplotlib e hydrogr (implementação originalmente testada na versão 1.2.2).
No modo qt, é necessário um backend Qt disponível no ambiente do Spyder.
Chuva e ETP: mm por hora; vazão: unidade escolhida abaixo.
Execute as células 1 a 13 em ordem na primeira execução.
"""

# ============================================================================
# CONFIGURACOES DE ENTRADA — EDITE ESTE BLOCO E EXECUTE O SCRIPT
# Coloque dados_GR4H.csv na mesma pasta deste script.
# ============================================================================
from pathlib import Path

PASTA = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
ARQUIVO_ENTRADA = PASTA / 'dados_GR4H.csv'
PASTA_RESULTADOS = PASTA / 'resultados_GR4H'

# 'qt': janela interativa ligada a console do Spyder (zoom, pan, selecao temporal).
# 'inline': figuras estaticas no painel Plots do Spyder; widgets nao funcionam.
MODO_GRAFICO = 'qt'
SALVAR_RESULTADOS = True
# None/None mostra toda a serie; para detalhe, informe datas ISO e rode a celula 9.
INICIO_VISUALIZACAO = None      # Ex.: '2023-01-01'
FIM_VISUALIZACAO = None         # Ex.: '2023-03-01'

# Analises adicionais (celulas 12 e 13).
EVENTOS_POR_PERIODO = 3         # Maiores eventos nao sobrepostos de cada periodo
HORAS_ANTES_EVENTO = 48
HORAS_DEPOIS_EVENTO = 72
# Opcional: intervalos manuais; se vazio, usa selecao automatica de eventos.
# Exemplo: [('2023-01-01 00:00', '2023-01-07 23:00')]
INTERVALOS_HIDROGRAMAS = []

AREA_KM2 = 539.0
UNIDADE_VAZAO = 'm3/s'         # 'm3/s' ou 'mm/h'; chuva e ETP em mm por hora
SEPARADOR_CSV = None           # None: detectar; ou ';' ou ','
SEPARADOR_DECIMAL = '.'        # Use ',' se o CSV tiver decimal brasileiro
DIA_ANTES_DO_MES = False       # True para datas como 31/12/2020

# Informe os nomes das quatro colunas do CSV.
COLUNA_DATA = 'data_hora'
COLUNA_CHUVA = 'P_mm_h'
COLUNA_EVAPO = 'ETP_mm_h'
COLUNA_VAZAO = 'Q_m3_s'         # Q_mm_h e redundante; nao converter Q_m3_s novamente

WARMUP_HORAS = 8760            # Primeiro ano excluido do ajuste e das metricas
FRACAO_CALIBRACAO = 0.70       # Fracao apos warm-up; restante vai para validacao
INICIO_VALIDACAO = None        # Ex.: '2022-01-01T00:00:00'; prevalece sobre fracao
FUNCAO_OBJETIVO = 'NSE'        # 'NSE' ou 'KGE'

SEMENTE_ALEATORIA = 42
NUMERO_REINICIOS = 3
MAXIMO_ITERACOES = 150
TAMANHO_POPULACAO = 12         # Multiplicador por parametro na evolucao diferencial
DISTANCIA_ENTRE_PICOS_H = 24
JANELA_PICOS_H = 12            # Busca o pico simulado em +/- este numero de horas

# Limites de busca, na ordem X1 (mm), X2 (mm/h), X3 (mm), X4 (h).
# Sao limites exploratorios; ajuste com justificativa hidrologica.
BOUNDS = [
    (1., 6000.),   # X1: teste de sensibilidade ao limite superior
    (-20., 20.),   # X2
    (1., 1000.),   # X3
    (.5, 120.)    # X4
]
# ============================================================================
# BIBLIOTECAS E IMPLEMENTACAO
# ============================================================================

from pathlib import Path
import json
import unicodedata
import numpy as np
import pandas as pd
import matplotlib

# %% 2. AMBIENTE GRAFICO E BIBLIOTECAS
# Se o Spyder estiver usando Agg de uma execucao anterior, reinicie a console.
try:
    from IPython import get_ipython
    console = get_ipython()
except ImportError:
    console = None
if console is not None:
    console.run_line_magic('matplotlib', MODO_GRAFICO)
elif MODO_GRAFICO == 'qt':
    matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import SpanSelector, Button
from matplotlib.patches import Patch
from scipy.optimize import differential_evolution
from scipy.signal import find_peaks
from hydrogr import ModelGr4h

# %% 3. FUNCOES DE APOIO — execute uma vez antes de ler os dados
# NSE: 1 e perfeito, 0 equivale a referencia da media observada.
# KGE original: combina correlacao, variabilidade e vies de media.
# PBIAS aqui: positivo = superestimativa. RMSE tem unidade de m3/s.
# Os estados sao reiniciados para cada candidato e preservados na simulacao final.
def norm(s):
    return ''.join(c for c in unicodedata.normalize('NFKD', s.lower()) if not unicodedata.combining(c)).strip()

def read_data(path, columns, sep=None, decimal='.', dayfirst=False):
    """Ler as quatro colunas informadas nas configurações."""
    raw = pd.read_csv(path, sep=sep, engine='python', decimal=decimal)
    # Aceita diferenças de maiúsculas, acentos e espaços nos nomes.
    nomes = {norm(coluna): coluna for coluna in raw.columns}
    selected = [nomes[norm(coluna)] for coluna in columns]
    dates = pd.to_datetime(raw[selected[0]], errors='raise', dayfirst=dayfirst)
    idx = pd.DatetimeIndex(dates)
    if idx.hasnans or idx.has_duplicates or not idx.is_monotonic_increasing:
        raise ValueError('Datas invalidas, duplicadas ou fora de ordem; corrija o CSV.')
    if len(idx) < 3 or not np.all((idx[1:] - idx[:-1]) == pd.Timedelta(hours=1)):
        raise ValueError('GR4H exige passo regular de 1 hora, sem lacunas. Dados diarios Xavier exigem GR4J ou dados horarios reais; nao dividir/repetir dias.')
    df = raw[selected[1:]].apply(pd.to_numeric, errors='raise').astype(float)
    df.columns = ['precipitation', 'evapotranspiration', 'qobs']
    df.index = idx
    if not np.isfinite(df.iloc[:, :2]).all().all() or (df.iloc[:, :2] < 0).any().any():
        raise ValueError('Chuva e ETP devem ser finitas e >= 0; sem preenchimento automatico.')
    if np.isinf(df.qobs).any() or (df.qobs.dropna() < 0).any():
        raise ValueError('Vazoes negativas/infinitas: use celulas vazias para observacoes ausentes.')
    return df

def simulate(forcing, params):
    # Nova instancia a cada avaliacao: estados independentes entre candidatos.
    model = ModelGr4h(dict(zip(['X1', 'X2', 'X3', 'X4'], map(float, params))))
    return model.run(forcing)['flow'].to_numpy()

def metrics(obs, sim):
    good = np.isfinite(obs) & np.isfinite(sim)
    o, s = np.asarray(obs)[good], np.asarray(sim)[good]
    out = dict(n=int(len(o)), NSE=np.nan, KGE=np.nan, PBIAS=np.nan,
               RMSE=np.nan, r=np.nan, alpha=np.nan, beta=np.nan)
    if len(o) < 2:
        return out
    out['RMSE'] = float(np.sqrt(np.mean((s-o)**2)))
    if np.sum(o) > 0:
        out['PBIAS'] = float(100*np.sum(s-o)/np.sum(o))
    if np.var(o) > 0:
        out['NSE'] = float(1-np.sum((s-o)**2)/np.sum((o-o.mean())**2))
    if o.mean() > 0 and o.std() > 0 and s.std() > 0:
        r = np.corrcoef(o, s)[0, 1]
        alpha, beta = s.std()/o.std(), s.mean()/o.mean()
        out.update(r=float(r), alpha=float(alpha), beta=float(beta),
                   KGE=float(1-np.sqrt((r-1)**2+(alpha-1)**2+(beta-1)**2)))
    return out

def peak_table(index, o, s, period, distance=24, window=12):
    # Picos observados acima do percentil 95; maximo simulado na janela local.
    # Eventos com qualquer observacao ausente na janela sao descartados.
    threshold = np.nanquantile(o, .95)
    candidates, _ = find_peaks(np.where(np.isfinite(o), o, -np.inf), height=threshold, distance=distance)
    rows = []
    for k in candidates:
        if k-window < 0 or k+window >= len(o):
            continue
        lo, hi = k-window, k+window+1
        if not np.isfinite(o[lo:hi]).all() or o[k] <= 0:
            continue
        j = lo + int(np.argmax(s[lo:hi]))
        rows.append(dict(periodo=period, data_obs=str(index[k]), data_sim=str(index[j]),
                         qobs_m3s=float(o[k]), qsim_m3s=float(s[j]),
                         erro_pico_pct=float(100*(s[j]-o[k])/o[k]), atraso_h=float((index[j]-index[k]).total_seconds()/3600)))
    return rows


# %% 4. LER O CSV E CONFERIR AS UNIDADES
dados = read_data(ARQUIVO_ENTRADA,
                 [COLUNA_DATA, COLUNA_CHUVA, COLUNA_EVAPO, COLUNA_VAZAO],
                 SEPARADOR_CSV, SEPARADOR_DECIMAL, DIA_ANTES_DO_MES)
fator_vazao = AREA_KM2 / 3.6
dados['qobs_m3s'] = dados.qobs * (fator_vazao if UNIDADE_VAZAO == 'mm/h' else 1.)
forcantes = dados[['precipitation', 'evapotranspiration']].copy()
print(f'CSV: {ARQUIVO_ENTRADA}')
print(f'{len(dados):,} horas; {dados.index[0]} ate {dados.index[-1]}')
print('Chuva e ETP: mm por hora. Vazao de trabalho: m3/s.')
print(dados.head().to_string())


# %% 5. DEFINIR WARM-UP, CALIBRACAO E VALIDACAO
# Warm-up: estabiliza reservatorios; sua vazao nao entra no ajuste nem nas metricas.
# Calibracao: observacoes usadas para escolher X1-X4.
# Validacao: parametros fixos, sem reajuste com observacoes desse periodo.
n_horas = len(dados)
fim_warmup = WARMUP_HORAS
corte = (int(dados.index.searchsorted(pd.Timestamp(INICIO_VALIDACAO)))
         if INICIO_VALIDACAO else
         fim_warmup + int((n_horas-fim_warmup)*FRACAO_CALIBRACAO))
if fim_warmup >= corte or corte >= n_horas or min(corte-fim_warmup, n_horas-corte) < 48:
    raise ValueError('Apos warm-up, reserve pelo menos 48 horas em cada periodo.')
fatias = {'Warm-up': slice(0, fim_warmup),
          'Calibracao': slice(fim_warmup, corte),
          'Validacao': slice(corte, n_horas)}
cores_periodos = {'Warm-up': '#999999', 'Calibracao': '#e6a532', 'Validacao': '#58a66c'}
dados['periodo'] = (['Warm-up']*fim_warmup + ['Calibracao']*(corte-fim_warmup)
                    + ['Validacao']*(n_horas-corte))
resumo_periodos = pd.DataFrame([
    {'periodo': nome, 'inicio': dados.index[sl][0] if len(dados.iloc[sl]) else pd.NaT,
     'fim': dados.index[sl][-1] if len(dados.iloc[sl]) else pd.NaT,
     'horas': len(dados.iloc[sl]),
     'q_validas': int(dados.iloc[sl].qobs_m3s.notna().sum())}
    for nome, sl in fatias.items()])
for nome in ('Calibracao', 'Validacao'):
    o = dados.iloc[fatias[nome]].qobs_m3s
    if o.notna().sum() < 48 or o.std() == 0 or o.mean() <= 0:
        raise ValueError(f'{nome}: precisa de >=48 vazoes, variabilidade e media positiva.')
print('\nPERIODOS (datas inclusivas):\n', resumo_periodos.to_string(index=False))


# %% 6. VISUALIZAR A DIVISAO ANTES DE CALIBRAR
def marcar_periodos(ax, legenda=False):
    for nome, sl in fatias.items():
        if sl.stop > sl.start:
            ax.axvspan(dados.index[sl.start],
                       dados.index[sl.stop-1] + pd.Timedelta(hours=1),
                       color=cores_periodos[nome], alpha=.16,
                       label=nome if legenda else '_nolegend_', zorder=0)
    for limite in (fim_warmup, corte):
        if 0 < limite < n_horas:
            ax.axvline(dados.index[limite], color='0.4', ls='--', lw=.8)

def formatar_datas(ax):
    loc = mdates.AutoDateLocator(minticks=3, maxticks=7)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

fig_periodos, eixos_periodos = plt.subplots(2, 1, figsize=(13, 6), sharex=True,
                                           num='GR4H — entradas e periodos', clear=True)
eixos_periodos[0].plot(dados.index, dados.precipitation, color='#2878b5', lw=.7)
eixos_periodos[0].set(ylabel='Chuva (mm/h)', title='Entradas e divisao temporal — confira antes de calibrar')
eixos_periodos[1].plot(dados.index, dados.qobs_m3s, color='black', lw=.7, label='Observada')
eixos_periodos[1].set(ylabel='Vazao (m3/s)')
for ax in eixos_periodos:
    marcar_periodos(ax)
    ax.grid(alpha=.2)
eixos_periodos[0].legend(handles=[Patch(color=cor, alpha=.3, label=nome)
                                for nome, cor in cores_periodos.items()], loc='upper right')
formatar_datas(eixos_periodos[1])
fig_periodos.tight_layout()
plt.show(block=False)


# %% 7. CALIBRAR — ESTA E A CELULA DEMORADA
# Para explorar graficos depois, NAO e necessario rodar esta celula novamente.
# O otimizador recebe somente forcantes ate o fim da calibracao.
obs_cal = dados.qobs_m3s.to_numpy()[fim_warmup:corte]
forcantes_cal = forcantes.iloc[:corte]
def objetivo_calibracao(parametros):
    s = simulate(forcantes_cal, parametros)[fim_warmup:] * fator_vazao
    if not np.isfinite(s).all() or (s < 0).any():
        return 1e12
    valor = metrics(obs_cal, s)[FUNCAO_OBJETIVO]
    return 1-valor if np.isfinite(valor) else 1e12

ajustes = []
registros_otimizacao = []
for reinicio in range(NUMERO_REINICIOS):
    print(f'Iniciando busca {reinicio+1}/{NUMERO_REINICIOS}...', flush=True)
    ajuste = differential_evolution(objetivo_calibracao, BOUNDS,
                                   seed=SEMENTE_ALEATORIA+reinicio,
                                   maxiter=MAXIMO_ITERACOES, popsize=TAMANHO_POPULACAO,
                                   tol=1e-6, polish=True, workers=1)
    ajustes.append(ajuste)
    registros_otimizacao.append(dict(reinicio=reinicio+1, seed=SEMENTE_ALEATORIA+reinicio,
                                     objetivo=float(ajuste.fun), convergiu=bool(ajuste.success),
                                     mensagem=str(ajuste.message), nfev=int(ajuste.nfev),
                                     **dict(zip(['X1','X2','X3','X4'], ajuste.x))))
    print(f'{FUNCAO_OBJETIVO}={1-ajuste.fun:.5f}; {ajuste.message}', flush=True)
melhor_ajuste = min(ajustes, key=lambda a: a.fun)
if not np.isfinite(melhor_ajuste.fun) or melhor_ajuste.fun >= 1e12:
    raise ValueError('Nao foi encontrada simulacao valida.')
parametros_calibrados = pd.DataFrame({
    'parametro': ['X1','X2','X3','X4'], 'valor': melhor_ajuste.x,
    'unidade': ['mm','mm/h','mm','h'],
    'significado': ['Capacidade de producao', 'Trocas de agua',
                    'Capacidade de propagacao', 'Tempo do hidrograma unitario'],
    'limite_inferior': [b[0] for b in BOUNDS], 'limite_superior': [b[1] for b in BOUNDS]})
tabela_otimizacao = pd.DataFrame(registros_otimizacao)
print('\nPARAMETROS:\n', parametros_calibrados.to_string(index=False))
print('\nConfira convergencia na tabela_otimizacao do Explorador de Variaveis.')


# %% 8. SIMULAR A SERIE TODA E MOSTRAR METRICAS
# Uma unica simulacao continua: a validacao herda os estados da calibracao.
dados['qsim_mm_h'] = simulate(forcantes, melhor_ajuste.x)
dados['qsim_m3s'] = dados.qsim_mm_h*fator_vazao
if not np.isfinite(dados.qsim_m3s).all() or (dados.qsim_m3s < 0).any():
    raise ValueError('Simulacao final invalida.')
linhas_metricas, eventos_picos = [], []
for nome in ('Calibracao', 'Validacao'):
    trecho = dados.iloc[fatias[nome]]
    o, s = trecho.qobs_m3s.to_numpy(), trecho.qsim_m3s.to_numpy()
    eventos = peak_table(trecho.index, o, s, nome, DISTANCIA_ENTRE_PICOS_H, JANELA_PICOS_H)
    eventos_picos.extend(eventos)
    valores = metrics(o, s)
    valores.update(periodo=nome, n_picos=len(eventos),
                   MAPE_picos_pct=np.mean([abs(e['erro_pico_pct']) for e in eventos]) if eventos else np.nan,
                   MAE_atraso_h=np.mean([abs(e['atraso_h']) for e in eventos]) if eventos else np.nan)
    linhas_metricas.append(valores)
tabela_metricas = pd.DataFrame(linhas_metricas).set_index('periodo')
tabela_picos = pd.DataFrame(eventos_picos, columns=['periodo','data_obs','data_sim',
    'qobs_m3s','qsim_m3s','erro_pico_pct','atraso_h'])
print('\nMETRICAS (warm-up excluido):\n', tabela_metricas.round(4).to_string())
print('NSE/KGE: melhor perto de 1. PBIAS: melhor perto de 0; positivo = superestima.')
print('RMSE: m3/s. Picos: maxima local observada >=P95 e maximo simulado em janela.')
fig_metricas, ax_metricas = plt.subplots(figsize=(12, 3.8), num='GR4H — metricas', clear=True)
ax_metricas.axis('off')
colunas_exibidas = ['NSE','KGE','PBIAS','RMSE','n','n_picos','MAPE_picos_pct','MAE_atraso_h']
tabela_visual = ax_metricas.table(cellText=tabela_metricas[colunas_exibidas].round(3).values,
                                  rowLabels=tabela_metricas.index,
                                  colLabels=colunas_exibidas, cellLoc='center', loc='center')
tabela_visual.auto_set_font_size(False)
tabela_visual.set_fontsize(9)
tabela_visual.scale(1, 1.8)
ax_metricas.set_title('Calibracao x validacao — warm-up excluido\nPBIAS em %; RMSE em m3/s; atraso em horas', pad=20)
fig_metricas.subplots_adjust(left=.14, right=.99, top=.78)
plt.show(block=False)


# %% 9. EXPLORAR OBSERVADO X SIMULADO — REEXECUTE LIVREMENTE
# No grafico inferior, clique e arraste um intervalo: o superior mostra o detalhe.
# Barra Matplotlib: lupa = zoom; cruz = mover; casa = voltar.
# Botoes laterais escondem/mostram as linhas; 'Serie toda' restaura a extensao.
# Objetos ficam vivos no console; nao use plt.close() se quiser manusea-los.
# Ao reexecutar esta célula, desconecte somente os controles desta figura.
fig_serie = plt.figure('GR4H — explorar series')
for controle in getattr(fig_serie, '_gr4h_controles', []):
    controle.disconnect_events()
fig_serie, (ax_serie, ax_contexto) = plt.subplots(2, 1, figsize=(14, 7),
    gridspec_kw={'height_ratios':[4, 1]}, num='GR4H — explorar series', clear=True)
fig_serie.subplots_adjust(left=.08, right=.82, bottom=.09, top=.92, hspace=.34)
linha_obs, = ax_serie.plot(dados.index, dados.qobs_m3s, color='black', lw=.9, label='Observada')
linha_sim, = ax_serie.plot(dados.index, dados.qsim_m3s, color='#2374c9', lw=.9, alpha=.8, label='Simulada')
ax_serie.set(ylabel='Vazao (m3/s)', title='Observada x simulada — selecione um trecho no painel inferior')
ax_contexto.plot(dados.index, dados.qobs_m3s, color='0.3', lw=.5)
ax_contexto.set(ylabel='m3/s', xlabel='Arraste aqui para selecionar o intervalo do painel superior')
for ax in (ax_serie, ax_contexto):
    marcar_periodos(ax)
    formatar_datas(ax)
    ax.grid(alpha=.2)
ax_serie.legend(handles=[linha_obs, linha_sim] + [Patch(color=c, alpha=.3, label=k)
                    for k, c in cores_periodos.items()], ncol=3, fontsize=9)

def atualizar_intervalo(inicio, fim):
    """Tambem pode ser chamado no console: atualizar_intervalo('2023-01-01','2023-02-01')."""
    inicio, fim = pd.Timestamp(inicio), pd.Timestamp(fim)
    if inicio >= fim:
        raise ValueError('Inicio precisa ser anterior ao fim.')
    trecho = dados.loc[inicio:fim]
    if trecho.empty:
        raise ValueError('Intervalo sem dados.')
    ax_serie.set_xlim(inicio, fim)
    visiveis = [trecho[col].to_numpy() for col, linha in
                [('qobs_m3s',linha_obs),('qsim_m3s',linha_sim)] if linha.get_visible()]
    valores = np.concatenate(visiveis) if visiveis else np.array([])
    valores = valores[np.isfinite(valores)]
    if len(valores):
        minimo, maximo = valores.min(), valores.max()
        margem = max((maximo-minimo)*.08, .01)
        ax_serie.set_ylim(max(0, minimo-margem), maximo+margem)
    fig_serie.canvas.draw_idle()

def selecionar_trecho(xmin, xmax):
    if xmax > xmin:
        atualizar_intervalo(pd.Timestamp(mdates.num2date(xmin)).tz_localize(None),
                            pd.Timestamp(mdates.num2date(xmax)).tz_localize(None))

seletor_temporal = SpanSelector(ax_contexto, selecionar_trecho, 'horizontal',
                               useblit=False, interactive=True,
                               props=dict(alpha=.3, facecolor='#2374c9'))
# Botões para mostrar ou ocultar cada série.
ax_botao_obs = fig_serie.add_axes([.84, .70, .14, .06])
ax_botao_sim = fig_serie.add_axes([.84, .61, .14, .06])
botao_obs = Button(ax_botao_obs, 'Observada: ON', useblit=False)
botao_sim = Button(ax_botao_sim, 'Simulada: ON', useblit=False)
def alternar_serie(rotulo):
    linha, botao = {'Observada':(linha_obs, botao_obs),
                    'Simulada':(linha_sim, botao_sim)}[rotulo]
    linha.set_visible(not linha.get_visible())
    botao.label.set_text(f"{rotulo}: {'ON' if linha.get_visible() else 'OFF'}")
    fig_serie.canvas.draw_idle()
botao_obs.on_clicked(lambda event: alternar_serie('Observada'))
botao_sim.on_clicked(lambda event: alternar_serie('Simulada'))
ax_reset = fig_serie.add_axes([.84, .48, .14, .06])
botao_reset = Button(ax_reset, 'Serie toda', useblit=False)
def restaurar_serie(event=None):
    atualizar_intervalo(dados.index[0], dados.index[-1])
    seletor_temporal.clear()
botao_reset.on_clicked(restaurar_serie)
fig_serie._gr4h_controles = [seletor_temporal, botao_obs, botao_sim, botao_reset]
atualizar_intervalo(INICIO_VISUALIZACAO or dados.index[0], FIM_VISUALIZACAO or dados.index[-1])
plt.show(block=False)
print('Grafico pronto. Para outro recorte, altere as datas da celula 1 e rode esta celula.')


# %% 10. DIAGNOSTICOS — DISPERSAO E CURVA DE PERMANENCIA
fig_diagnostico, eixos_diag = plt.subplots(2, 2, figsize=(12, 8),
                                          num='GR4H — diagnosticos', clear=True)
for i, nome in enumerate(('Calibracao','Validacao')):
    trecho = dados.iloc[fatias[nome]].dropna(subset=['qobs_m3s'])
    o, s = trecho.qobs_m3s.to_numpy(), trecho.qsim_m3s.to_numpy()
    ax = eixos_diag[i,0]
    ax.scatter(o, s, s=4, alpha=.3)
    limite = max(o.max(), s.max())
    ax.plot([0, limite], [0, limite], 'k--', label='Igualdade')
    ax.set(title=nome, xlabel='Observada (m3/s)', ylabel='Simulada (m3/s)')
    ax.legend()
    ax = eixos_diag[i,1]
    for valores, rotulo in ((o,'Observada'),(s,'Simulada')):
        ordenados = np.sort(valores)[::-1]
        ax.plot(100*np.arange(1,len(ordenados)+1)/(len(ordenados)+1), ordenados, label=rotulo)
    ax.set(title=f'Permanencia — {nome}', xlabel='Permanencia (%)', ylabel='Vazao (m3/s)', yscale='symlog')
    ax.legend()
for ax in eixos_diag.flat:
    ax.grid(alpha=.2)
fig_diagnostico.tight_layout()
plt.show(block=False)


# %% 11. SALVAR RESULTADOS — OS GRAFICOS CONTINUAM ABERTOS
if SALVAR_RESULTADOS:
    destino = Path(PASTA_RESULTADOS)
    destino.mkdir(parents=True, exist_ok=True)
    dados.to_csv(destino/'series.csv', index_label='data_hora')
    resumo_periodos.to_csv(destino/'periodos.csv', index=False)
    tabela_metricas.to_csv(destino/'metricas.csv')
    parametros_calibrados.to_csv(destino/'parametros.csv', index=False)
    tabela_picos.to_csv(destino/'picos.csv', index=False)
    tabela_otimizacao.to_csv(destino/'otimizacao.csv', index=False)
    # Somente configuracoes em MAIUSCULAS, nao tabelas nem objetos graficos.
    configuracao = {k:v for k,v in list(globals().items()) if k.isupper() and not k.startswith('_')}
    (destino/'configuracao.json').write_text(json.dumps(configuracao, indent=2, default=str), encoding='utf-8')
    for figura, nome in ((fig_periodos,'periodos'),(fig_metricas,'metricas'),
                         (fig_serie,'serie_interativa'),(fig_diagnostico,'diagnosticos')):
        figura.savefig(destino/f'{nome}.png', dpi=160)
    print(f'CSVs e imagens salvos em: {destino.resolve()}')
    print('O PNG da serie guarda o recorte atual; a interatividade fica na janela Qt.')
else:
    print('SALVAR_RESULTADOS=False: resultados mantidos no Explorador de Variaveis.')


# %% 12. HIDROGRAMAS DE EVENTOS — NSE, ERRO DE PICO E TEMPO DO PICO
# Rode depois da celula 8. Configuracoes: EVENTOS_POR_PERIODO e janelas na celula 1.
# Automático: maiores eventos por período, sem sobrepor janelas.
# Manual: informe as datas em INTERVALOS_HIDROGRAMAS, na célula 1.
# NSE usa toda a janela. Picos são os máximos dentro dessa mesma janela.
# Erro positivo = superestimativa. Atraso positivo = pico simulado depois.
janelas_eventos = []
if INTERVALOS_HIDROGRAMAS:
    for inicio, fim in INTERVALOS_HIDROGRAMAS:
        inicio, fim = pd.Timestamp(inicio), pd.Timestamp(fim)
        if inicio >= fim:
            raise ValueError('Intervalo manual: inicio deve ser anterior ao fim.')
        if inicio < dados.index[fim_warmup] or fim > dados.index[-1]:
            raise ValueError('Intervalo manual deve estar nos dados e fora do warm-up.')
        trecho = dados.loc[inicio:fim]
        if len(trecho) < 3:
            raise ValueError('Intervalo manual precisa conter pelo menos 3 horas.')
        janelas_eventos.append((trecho.index[0], trecho.index[-1]))
else:
    for nome in ('Calibracao', 'Validacao'):
        sl = fatias[nome]
        candidatos = tabela_picos.loc[tabela_picos.periodo == nome].sort_values('qobs_m3s', ascending=False)
        escolhidas = []
        for _, evento in candidatos.iterrows():
            centro = pd.Timestamp(evento.data_obs)
            inicio = centro-pd.Timedelta(hours=HORAS_ANTES_EVENTO)
            fim = centro+pd.Timedelta(hours=HORAS_DEPOIS_EVENTO)
            if inicio < dados.index[sl.start] or fim > dados.index[sl.stop-1]:
                continue
            if any(inicio <= b and fim >= a for a, b in escolhidas):
                continue
            if dados.loc[inicio:fim, 'qobs_m3s'].isna().any():
                continue
            escolhidas.append((inicio, fim))
            if len(escolhidas) >= EVENTOS_POR_PERIODO:
                break
        janelas_eventos.extend(escolhidas)

resultados_eventos = []
figuras_eventos = []
for numero, (inicio, fim) in enumerate(sorted(janelas_eventos), start=1):
    trecho = dados.loc[inicio:fim].copy()
    # Mesmas horas com observacao disponivel para metricas e identificacao de picos.
    pares = trecho.dropna(subset=['qobs_m3s', 'qsim_m3s'])
    if len(pares) < 3:
        print(f'Janela {numero}: menos de 3 pares validos; ignorada.')
        continue
    qp_obs, qp_sim = pares.qobs_m3s.max(), pares.qsim_m3s.max()
    tp_obs, tp_sim = pares.qobs_m3s.idxmax(), pares.qsim_m3s.idxmax()
    erro_pico = qp_sim-qp_obs
    erro_pico_pct = 100*erro_pico/qp_obs if qp_obs > 0 else np.nan
    atraso = (tp_sim-tp_obs).total_seconds()/3600
    nse_evento = metrics(pares.qobs_m3s.to_numpy(), pares.qsim_m3s.to_numpy())['NSE']
    periodo = ' / '.join(trecho.periodo.unique())
    resultados_eventos.append(dict(evento=numero, periodo=periodo,
        inicio=inicio, fim=fim, n_horas=len(trecho), n_pares=len(pares), NSE=nse_evento,
        Qp_obs_m3s=qp_obs, Qp_sim_m3s=qp_sim, erro_pico_m3s=erro_pico,
        erro_pico_pct=erro_pico_pct, tempo_pico_obs=tp_obs, tempo_pico_sim=tp_sim,
        atraso_pico_h=atraso, erro_absoluto_tempo_h=abs(atraso)))
    fig_evento, (ax_chuva, ax_evento) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
        gridspec_kw={'height_ratios':[1, 3]}, num=f'GR4H — evento {numero}', clear=True)
    figuras_eventos.append(fig_evento)
    ax_chuva.bar(trecho.index, trecho.precipitation, width=.8/24, color='#478fc1')
    ax_chuva.set(ylabel='Chuva\n(mm/h)')
    ax_chuva.invert_yaxis()
    ax_evento.plot(trecho.index, trecho.qobs_m3s, color='black', label='Observada')
    ax_evento.plot(trecho.index, trecho.qsim_m3s, color='#2374c9', label='Simulada')
    ax_evento.scatter([tp_obs], [qp_obs], color='black', marker='o', s=55, zorder=5, label='Pico observado')
    ax_evento.scatter([tp_sim], [qp_sim], color='#2374c9', marker='^', s=65, zorder=5, label='Pico simulado')
    ax_evento.axvline(tp_obs, color='black', ls=':', alpha=.6)
    ax_evento.axvline(tp_sim, color='#2374c9', ls=':', alpha=.6)
    ax_evento.set(ylabel='Vazao (m3/s)', xlabel='Data e hora', xlim=(inicio, fim))
    for ax in (ax_chuva, ax_evento):
        marcar_periodos(ax)
        ax.grid(alpha=.2)
    formatar_datas(ax_evento)
    ax_evento.legend(loc='best', ncol=2)
    fig_evento.suptitle(f'Evento {numero} — {periodo}\nNSE da janela = {nse_evento:.3f} | '
                        f'Erro de pico = {erro_pico_pct:+.1f}% | Atraso = {atraso:+.1f} h')
    fig_evento.tight_layout(rect=(0,0,1,.91))
    plt.show(block=False)

colunas_eventos = ['evento','periodo','inicio','fim','n_horas','n_pares','NSE',
    'Qp_obs_m3s','Qp_sim_m3s','erro_pico_m3s','erro_pico_pct','tempo_pico_obs',
    'tempo_pico_sim','atraso_pico_h','erro_absoluto_tempo_h']
tabela_hidrogramas = pd.DataFrame(resultados_eventos, columns=colunas_eventos)
if tabela_hidrogramas.empty:
    print('Nenhuma janela elegivel. Reduza a janela ou informe INTERVALOS_HIDROGRAMAS na celula 1.')
else:
    print('\nANALISE POR HIDROGRAMA (nao resume todos os eventos):\n',
          tabela_hidrogramas.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print('NSE indefinido em serie observada constante; erro percentual indefinido se pico observado=0.')
if SALVAR_RESULTADOS:
    destino = Path(PASTA_RESULTADOS)
    destino.mkdir(parents=True, exist_ok=True)
    tabela_hidrogramas.to_csv(destino/'hidrogramas_eventos.csv', index=False)
    for numero, figura in enumerate(figuras_eventos, start=1):
        figura.savefig(destino/f'hidrograma_evento_{numero:02d}.png', dpi=160)


# %% 13. PERMANENCIA COM Q90 E Q95 — CALIBRACAO E VALIDACAO
# Q90/Q95 são superadas em 90%/95% das horas: quantis 0,10 e 0,05.
# Use as mesmas horas observadas e simuladas. Curva com posição de Weibull.
resultados_permanencia = []
curvas_permanencia = []
fig_permanencia_q, eixos_perm = plt.subplots(1, 2, figsize=(14, 6),
    num='GR4H — permanencia Q90 Q95', clear=True)
for ax, nome in zip(eixos_perm, ('Calibracao', 'Validacao')):
    trecho = dados.iloc[fatias[nome]].dropna(subset=['qobs_m3s','qsim_m3s'])
    n_pares = len(trecho)
    permanencia = 100*np.arange(1,n_pares+1)/(n_pares+1)
    qrefs = {}
    for coluna, rotulo, cor in [('qobs_m3s','Observada','black'),
                                ('qsim_m3s','Simulada','#2374c9')]:
        valores = trecho[coluna].to_numpy()
        ordenados = np.sort(valores)[::-1]
        ax.plot(permanencia, ordenados, color=cor, label=rotulo, lw=1.3)
        qrefs[rotulo] = np.quantile(valores, [.10,.05], method='linear')
        curvas_permanencia.append(pd.DataFrame({'periodo':nome, 'serie':rotulo,
                                   'permanencia_pct':permanencia, 'vazao_m3s':ordenados}))
        ax.scatter([90,95], qrefs[rotulo], color=cor, marker='o' if rotulo=='Observada' else '^', zorder=5)
    for j, referencia in enumerate(('Q90','Q95')):
        qo, qs = qrefs['Observada'][j], qrefs['Simulada'][j]
        resultados_permanencia.append(dict(periodo=nome, referencia=referencia,
            observado_m3s=qo, simulado_m3s=qs, erro_m3s=qs-qo,
            erro_pct=100*(qs-qo)/qo if qo > 0 else np.nan, n_pares=n_pares))
    for p in (90,95):
        ax.axvline(p, color='0.5', lw=.8, ls=':')
    texto = '\n'.join(f'{ref}: obs {qrefs["Observada"][i]:.3f} | sim {qrefs["Simulada"][i]:.3f} m3/s'
                       for i,ref in enumerate(('Q90','Q95')))
    ax.text(.04,.06,texto, transform=ax.transAxes, fontsize=9,
            bbox=dict(facecolor='white', edgecolor='0.8', alpha=.9))
    ax.set(title=f'{nome} — {n_pares} horas pareadas', xlabel='Permanencia / excedencia (%)',
           ylabel='Vazao (m3/s)', xlim=(0,100), yscale='symlog')
    ax.grid(alpha=.25, which='both')
    ax.legend(loc='upper right')
fig_permanencia_q.suptitle('Curvas de permanencia horarias — warm-up excluido\n'
                          'Q90 e Q95: referencias de baixas vazoes; positivo no erro = superestimativa')
fig_permanencia_q.tight_layout(rect=(0,0,1,.89))
plt.show(block=False)
tabela_q90_q95 = pd.DataFrame(resultados_permanencia)
tabela_curvas_permanencia = pd.concat(curvas_permanencia, ignore_index=True)
print('\nQ90 E Q95 (m3/s):\n', tabela_q90_q95.round(4).to_string(index=False))
print('Estas referencias sao HORARIAS. Para comparar com estudo diario, agregue observado e simulado a medias diarias completas antes de calcular.')
if SALVAR_RESULTADOS:
    destino = Path(PASTA_RESULTADOS)
    destino.mkdir(parents=True, exist_ok=True)
    tabela_q90_q95.to_csv(destino/'q90_q95.csv', index=False)
    tabela_curvas_permanencia.to_csv(destino/'curvas_permanencia.csv', index=False)
    fig_permanencia_q.savefig(destino/'permanencia_q90_q95.png', dpi=160)
