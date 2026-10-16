# %% 00 | HYDROUAI - LSTM vs MLP | PREVISAO HORARIA
"""Comparacao didatica LSTM/MLP com a mesma janela horaria de P e/ou Q.
Base: series_horarias_preenchidas.csv (mesma pasta deste script).
Grid selecionado SOMENTE pela validacao; teste independente.
Dependencias: numpy, pandas, matplotlib, scikit-learn, torch.
"""
from pathlib import Path
from itertools import product
import copy
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# %% 01 | CONFIGURACOES - ALTERE AQUI
PASTA = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
ARQUIVO = PASTA / 'series_horarias_preenchidas.csv'
SAIDA = PASTA / 'resultados_LSTM_vs_MLP'
PASTA_MODELOS = SAIDA / 'modelos_completos'
COL_DATA = 'data_hora'
COL_P = ['2463_P', '84100000_P']
COL_Q = '2463_Q'
MODELOS = ['LSTM', 'MLP']
ENTRADAS = 'PQ'  # 'P' (so chuva) ou 'PQ' (chuva e vazao)
HORIZONTES_H = [1, 3, 6]
JANELAS_H_GRID = [24]  # inclui t e as H-1 horas anteriores
CAMADAS_GRID = [1]
UNIDADES_GRID = [32]
PCT_TREINO = 70
PCT_VALIDACAO = 15
PCT_TESTE = 15
EPOCAS_MAX = 100
BATCH_SIZE = 4320
LR_INICIAL = 0.001
WEIGHT_DECAY = 0.0
USAR_LR_ADAPTATIVO = True
PACIENCIA_LR = 5
FATOR_LR = 0.5
LR_MIN = 1e-5
USAR_EARLY_STOPPING = True
PACIENCIA_EARLY = 15
MELHORIA_MINIMA = 1e-6
# Na celula 10, mostrar somente o maior pico observado do teste.
N_EVENTOS = 1
JANELA_ANTES_PICO_H = 48
JANELA_DEPOIS_PICO_H = 48
# None = usar todos os pontos no grafico da serie completa da celula 08.
HORAS_PLOT_TESTE = 500
# Numero maximo de batches hachurados no grafico (None = todos).
MAX_BATCHES_HACHURADOS = None
SEMENTE = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if ENTRADAS not in ('P', 'PQ'):
    raise ValueError("ENTRADAS deve ser 'P' ou 'PQ'.")
if abs(PCT_TREINO + PCT_VALIDACAO + PCT_TESTE - 100) > 1e-8 or min(PCT_TREINO, PCT_VALIDACAO, PCT_TESTE) <= 0:
    raise ValueError('Percentuais positivos devem somar 100.')
print('Dispositivo:', DEVICE)

# %% 02 | LEITURA, FREQUENCIA E DIVISAO CRONOLOGICA
if not ARQUIVO.exists():
    raise FileNotFoundError(f'CSV nao encontrado: {ARQUIVO}')
df = pd.read_csv(ARQUIVO, sep=';', encoding='utf-8-sig')
colunas = [COL_DATA, *COL_P, COL_Q]
if not set(colunas).issubset(df.columns):
    raise ValueError(f'Colunas necessarias: {colunas}')
df[COL_DATA] = pd.to_datetime(df[COL_DATA], errors='raise')
if df[COL_DATA].duplicated().any():
    raise ValueError('Datas duplicadas.')
df = df.sort_values(COL_DATA).set_index(COL_DATA).asfreq('h')
for col in [*COL_P, COL_Q]:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    if (df[col].dropna() < 0).any():
        raise ValueError(f'Valores negativos em {col}.')
n = len(df)
i_treino = int(n * PCT_TREINO / 100)
i_val = int(n * (PCT_TREINO + PCT_VALIDACAO) / 100)
FIM_TREINO = df.index[i_treino - 1]
FIM_VALIDACAO = df.index[i_val - 1]
print('Periodo:', df.index.min(), 'ate', df.index.max())
print('Dados ausentes:', df[colunas[1:]].isna().sum().to_dict())
print('Fim treino:', FIM_TREINO, '| fim validacao:', FIM_VALIDACAO)
fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(df.index, df[COL_Q], color='0.25', lw=.7, label='Q observada')
for nome, a, b, cor in [('Treino', df.index[0], FIM_TREINO, 'tab:blue'),
                         ('Validacao', df.index[i_treino], FIM_VALIDACAO, 'tab:orange'),
                         ('Teste', df.index[i_val], df.index[-1], 'tab:green')]:
    ax.axvspan(a, b, color=cor, alpha=.17, label=nome)
ax.set(title='Divisao temporal - vazao observada', xlabel='Data', ylabel='Vazao')
ax.legend(); ax.grid(alpha=.2); plt.tight_layout(); plt.show()

# %% 03 | SEQUENCIAS TEMPORAIS - MESMAS ENTRADAS PARA LSTM E MLP
FEATURES = COL_P + ([COL_Q] if ENTRADAS == 'PQ' else [])
def montar_sequencias(dados, janela, horizonte):
    """X[i] = [t-H+1,...,t]; y[i] = Q(t+h); particao pela data-alvo."""
    valores = dados[FEATURES].to_numpy(dtype=np.float32)
    q = dados[COL_Q].to_numpy(dtype=np.float32)
    xs, ys, origens, alvos = [], [], [], []
    for i in range(janela - 1, len(dados) - horizonte):
        x = valores[i-janela+1:i+1]
        y = q[i+horizonte]
        if not np.isfinite(x).all() or not np.isfinite(y):
            continue
        xs.append(x); ys.append(y)
        origens.append(dados.index[i]); alvos.append(dados.index[i+horizonte])
    if not xs:
        raise ValueError('Nenhuma sequencia valida. Confira os dados e janelas.')
    return np.stack(xs), np.asarray(ys, dtype=np.float32), pd.DatetimeIndex(origens), pd.DatetimeIndex(alvos)

def mascaras_temporais(datas_alvo):
    return (np.asarray(datas_alvo <= FIM_TREINO),
            np.asarray((datas_alvo > FIM_TREINO) & (datas_alvo <= FIM_VALIDACAO)),
            np.asarray(datas_alvo > FIM_VALIDACAO))

# Tabela ilustrativa para o Spyder: linhas sao origens de previsao.
janela_exemplo = JANELAS_H_GRID[0]
horizonte_exemplo = HORIZONTES_H[0]
X_EXEMPLO, Y_EXEMPLO, DATAS_ORIGEM_EXEMPLO, DATAS_ALVO_EXEMPLO = montar_sequencias(df, janela_exemplo, horizonte_exemplo)
colunas_exemplo = [f'{col}_t-{janela_exemplo-1-k}h' for k in range(janela_exemplo) for col in FEATURES]
TABELA_ENTRADAS_DIDATICA = pd.DataFrame(X_EXEMPLO.reshape(len(X_EXEMPLO), -1), index=DATAS_ORIGEM_EXEMPLO, columns=colunas_exemplo)
TABELA_ENTRADAS_DIDATICA[f'ALVO_Q_t+{horizonte_exemplo}h'] = Y_EXEMPLO
print('Tabela didatica (primeiras 5 linhas):\n', TABELA_ENTRADAS_DIDATICA.head().to_string())
print('LSTM:', X_EXEMPLO.shape, '| MLP:', X_EXEMPLO.reshape(len(X_EXEMPLO), -1).shape)

# %% 04 | ARQUITETURAS LSTM E MLP (SEM DROPOUT)
class LSTMModel(nn.Module):
    def __init__(self, n_features, unidades, camadas):
        super().__init__()
        self.lstm = nn.LSTM(n_features, unidades, num_layers=camadas,
                            batch_first=True)
        self.saida = nn.Linear(unidades, 1)
    def forward(self, x):
        h, _ = self.lstm(x)
        return self.saida(h[:, -1, :])

class MLPModel(nn.Module):
    def __init__(self, n_features, janela, unidades, camadas):
        super().__init__()
        blocos = []
        entrada = n_features * janela
        for _ in range(camadas):
            blocos += [nn.Linear(entrada, unidades), nn.ReLU()]
            entrada = unidades
        blocos.append(nn.Linear(entrada, 1))
        self.net = nn.Sequential(*blocos)
    def forward(self, x):
        return self.net(x.flatten(start_dim=1))

def criar_modelo(cfg, n_features):
    if cfg['modelo'] == 'LSTM':
        return LSTMModel(n_features, cfg['unidades'], cfg['camadas']).to(DEVICE)
    return MLPModel(n_features, cfg['janela_h'], cfg['unidades'], cfg['camadas']).to(DEVICE)

# %% 05 | TREINAMENTO - CHAVES EARLY STOPPING E LR ADAPTATIVO
def fixar_semente():
    random.seed(SEMENTE); np.random.seed(SEMENTE); torch.manual_seed(SEMENTE)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEMENTE)

def treinar(modelo, xtr, ytr, xva, yva):
    loader = DataLoader(TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr[:, None])),
                        batch_size=BATCH_SIZE, shuffle=False)
    xv = torch.from_numpy(xva).to(DEVICE)
    yv = torch.from_numpy(yva[:, None]).to(DEVICE)
    opt = torch.optim.Adam(modelo.parameters(), lr=LR_INICIAL, weight_decay=WEIGHT_DECAY)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', patience=PACIENCIA_LR, factor=FATOR_LR, min_lr=LR_MIN)
        if USAR_LR_ADAPTATIVO else None)
    criterio = nn.MSELoss()
    melhor_perda, melhor_estado, melhor_epoca, espera = np.inf, None, 0, 0
    historico = []
    for epoca in range(1, EPOCAS_MAX + 1):
        modelo.train(); soma = 0.; n_amostras = 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); perda = criterio(modelo(xb), yb)
            perda.backward(); opt.step()
            soma += perda.item() * len(xb); n_amostras += len(xb)
        modelo.eval()
        with torch.no_grad():
            val = criterio(modelo(xv), yv).item()
        historico.append({'epoca': epoca, 'treino': soma/n_amostras,
                          'validacao': val, 'lr': opt.param_groups[0]['lr']})
        if val < melhor_perda - MELHORIA_MINIMA:
            melhor_perda, melhor_estado, melhor_epoca, espera = val, copy.deepcopy(modelo.state_dict()), epoca, 0
        else:
            espera += 1
        if scheduler is not None: scheduler.step(val)
        if epoca == 1 or epoca % 10 == 0:
            print(f'    epoca {epoca:3d}: treino={soma/n_amostras:.5f} val={val:.5f} LR={opt.param_groups[0]["lr"]:.5g}')
        if USAR_EARLY_STOPPING and espera >= PACIENCIA_EARLY:
            print('    early stopping na epoca', epoca); break
    modelo.load_state_dict(melhor_estado)
    return modelo, pd.DataFrame(historico), melhor_epoca

# %% 06 | NORMALIZACAO E METRICAS
# Os escalonadores sao ajustados EXCLUSIVAMENTE no conjunto de treino.
def preparar(cfg):
    X, y, datas_origem, datas_alvo = montar_sequencias(df, cfg['janela_h'], cfg['horizonte_h'])
    tr, va, te = mascaras_temporais(datas_alvo)
    if not (tr.any() and va.any() and te.any()):
        raise ValueError(f'Treino/validacao/teste vazios: {cfg}')
    n_features = X.shape[-1]
    sx = StandardScaler().fit(X[tr].reshape(-1, n_features))
    sy = StandardScaler().fit(y[tr, None])
    Xn = sx.transform(X.reshape(-1, n_features)).reshape(X.shape).astype(np.float32)
    yn = sy.transform(y[:, None]).ravel().astype(np.float32)
    return {'X': Xn, 'y': y, 'yn': yn, 'origem': datas_origem, 'datas': datas_alvo,
            'tr': tr, 'va': va, 'te': te, 'sx': sx, 'sy': sy, 'n_features': n_features}

def prever(modelo, X):
    """Inferencia em blocos para nao sobrecarregar a memoria do computador."""
    modelo.eval()
    saidas = []
    with torch.no_grad():
        for inicio in range(0, len(X), BATCH_SIZE):
            xb = torch.from_numpy(X[inicio:inicio+BATCH_SIZE]).to(DEVICE)
            saidas.append(modelo(xb).cpu().numpy().ravel())
    return np.concatenate(saidas)

def metricas(obs, pred):
    obs, pred = np.asarray(obs, dtype=float), np.asarray(pred, dtype=float)
    erro = pred-obs; den = np.sum((obs-obs.mean())**2)
    nse = 1-np.sum(erro**2)/den if den else np.nan
    rmse = np.sqrt(np.mean(erro**2))
    mae = np.mean(np.abs(erro))
    r = np.corrcoef(obs, pred)[0,1] if np.std(obs) and np.std(pred) else np.nan
    alpha = np.std(pred)/np.std(obs) if np.std(obs) else np.nan
    beta = np.mean(pred)/np.mean(obs) if np.mean(obs) else np.nan
    kge = 1-np.sqrt((r-1)**2+(alpha-1)**2+(beta-1)**2)
    pbias = 100*np.sum(obs-pred)/np.sum(obs) if np.sum(obs) else np.nan
    return dict(NSE=nse, KGE=kge, RMSE=rmse, MAE=mae, PBIAS=pbias)

# %% 06B | BATCHES HACHURADOS NA SERIE COMPLETA (ANTES DO TREINAMENTO)
# Os batches sao grupos de amostras consecutivas do conjunto de TREINO.
# A serie observada completa aparece ao fundo, incluindo validacao e teste.
# A marcacao usa as datas-alvo das amostras validas; o tamanho e BATCH_SIZE.
_cfg_visual = dict(janela_h=JANELAS_H_GRID[0], horizonte_h=HORIZONTES_H[0])
_Xvis, _yvis, _origvis, _datvis = montar_sequencias(
    df, _cfg_visual['janela_h'], _cfg_visual['horizonte_h'])
_trvis, _vavis, _tevis = mascaras_temporais(_datvis)
_datas_treino = _datvis[_trvis]
_n_batches = int(np.ceil(len(_datas_treino) / BATCH_SIZE))
_fig, _ax = plt.subplots(figsize=(15, 4.5))
_ax.plot(df.index, df[COL_Q], color='0.15', lw=0.65)
for _b in range(_n_batches if MAX_BATCHES_HACHURADOS is None
                else min(_n_batches, MAX_BATCHES_HACHURADOS)):
    _ini = _b * BATCH_SIZE
    _fim = min((_b + 1) * BATCH_SIZE, len(_datas_treino)) - 1
    _ax.axvspan(_datas_treino[_ini], _datas_treino[_fim],
                facecolor='tab:blue' if _b % 2 == 0 else 'tab:orange',
                alpha=0.10, edgecolor='0.45', hatch='///', linewidth=0.0)
_ax.axvline(FIM_TREINO, color='0.4', ls='--', lw=1)
_ax.axvline(FIM_VALIDACAO, color='0.4', ls='--', lw=1)
_ax.set(title=f'Serie completa | batches consecutivos no treino (N={BATCH_SIZE})',
        xlabel='Data', ylabel='Vazao')
_ax.grid(alpha=0.15)
plt.tight_layout()
plt.show()
print(f'Amostras de treino: {len(_datas_treino)} | batches: {_n_batches}')

# %% 07 | GRID SEARCH - SELECAO APENAS PELA VALIDACAO
SAIDA.mkdir(parents=True, exist_ok=True)
grade = [dict(modelo=m, horizonte_h=h, janela_h=j, camadas=c, unidades=u)
         for m,h,j,c,u in product(MODELOS,HORIZONTES_H,JANELAS_H_GRID,
                                  CAMADAS_GRID,UNIDADES_GRID)]
print('Treinamentos previstos:', len(grade))
resultados_grid, melhores = [], {}
for i, cfg in enumerate(grade, 1):
    print(f'\n[{i}/{len(grade)}] {cfg}')
    dados = preparar(cfg)
    fixar_semente()
    modelo = criar_modelo(cfg, dados['n_features'])
    modelo, hist, epoca_melhor = treinar(modelo, dados['X'][dados['tr']], dados['yn'][dados['tr']],
                                       dados['X'][dados['va']], dados['yn'][dados['va']])
    val_pred = dados['sy'].inverse_transform(prever(modelo, dados['X'][dados['va']])[:,None]).ravel()
    mse_val = float(np.mean((val_pred-dados['y'][dados['va']])**2))
    linha = {**cfg, 'MSE_validacao': mse_val, 'RMSE_validacao': np.sqrt(mse_val),
             'NSE_validacao': metricas(dados['y'][dados['va']],val_pred)['NSE'],
             'melhor_epoca': epoca_melhor, 'epocas_executadas': len(hist)}
    resultados_grid.append(linha)
    chave = (cfg['modelo'], cfg['horizonte_h'])
    if chave not in melhores or mse_val < melhores[chave]['MSE_validacao']:
        melhores[chave] = {**linha, 'estado': copy.deepcopy(modelo.cpu().state_dict()),
                           'historico': hist.copy()}
        print('    -> melhor configuracao na validacao')
    pd.DataFrame(resultados_grid).to_csv(SAIDA/'grid_validacao.csv',sep=';',index=False)
tabela_grid = pd.DataFrame(resultados_grid)
print('\nMELHORES CONFIGURACOES:')
print(pd.DataFrame([{k:v for k,v in melhor.items() if k not in ('estado','historico')}
                    for melhor in melhores.values()]).to_string(index=False))

# %% 08 | TESTE INDEPENDENTE - PREVISOES DETERMINISTICAS
resumo_teste, previsoes_teste = [], {}
for chave, melhor in melhores.items():
    cfg = {k:melhor[k] for k in ('modelo','horizonte_h','janela_h','camadas','unidades')}
    dados = preparar(cfg)
    modelo = criar_modelo(cfg, dados['n_features'])
    modelo.load_state_dict(melhor['estado'])
    modelo.to(DEVICE)
    pred = dados['sy'].inverse_transform(
        prever(modelo, dados['X'][dados['te']])[:,None]).ravel()
    pred = np.maximum(pred, 0.0)
    tabela = pd.DataFrame({'observado':dados['y'][dados['te']], 'previsto':pred},
                          index=dados['datas'][dados['te']])
    tabela.index.name = 'data_alvo'
    previsoes_teste[chave] = tabela
    resumo_teste.append({**cfg, **metricas(tabela.observado, tabela.previsto)})
    prefixo = f'{cfg["modelo"]}_h{cfg["horizonte_h"]}'
    tabela.to_csv(SAIDA/f'previsoes_{prefixo}.csv',sep=';')
    melhor['historico'].to_csv(SAIDA/f'convergencia_{prefixo}.csv',sep=';',index=False)
resumo = pd.DataFrame(resumo_teste)
resumo.to_csv(SAIDA/'comparacao_teste.csv',sep=';',index=False)
print('\nRESULTADOS NO TESTE:')
print(resumo[['modelo','horizonte_h','janela_h','camadas','unidades',
              'NSE','KGE','RMSE','PBIAS']].to_string(index=False))

# %% 09 | CONVERGENCIA E LEARNING RATE - MELHORES MODELOS
for (nome,h), melhor in melhores.items():
    hist = melhor['historico']; epoca = melhor['melhor_epoca']
    fig, ax = plt.subplots(2,1,figsize=(10,6),sharex=True,gridspec_kw={'height_ratios':[2,1]})
    ax[0].plot(hist.epoca,hist.treino,label='Treino')
    ax[0].plot(hist.epoca,hist.validacao,label='Validacao')
    ax[0].axvline(epoca,ls='--',color='tab:green',label=f'Melhor epoca: {epoca}')
    ax[0].set(ylabel='MSE (alvo normalizado)',title=f'{nome} +{h}h - convergencia')
    ax[0].legend(); ax[0].grid(alpha=.2)
    ax[1].step(hist.epoca,hist.lr,where='post')
    ax[1].set(xlabel='Epoca',ylabel='Learning rate'); ax[1].grid(alpha=.2)
    plt.tight_layout(); plt.savefig(SAIDA/f'convergencia_{nome}_h{h}.png',dpi=180); plt.show()

# %% 10 | EVENTOS DE CHEIA: DOIS MODELOS NO MESMO GRAFICO
# Escolhemos o maior pico observado no teste, como no grafico original.
# Mostramos LSTM e MLP lado a lado para cada horizonte. Para cada evento, avaliamos:
# - erro de pico (%): 100*(Qp_prev - Qp_obs)/Qp_obs (positivo = superestimativa)
# - erro de tempo (h): data pico previsto - data pico observado
# - NSE do evento: calculado somente dentro da janela exibida
# Cada linha da figura corresponde a um evento; cada coluna a um horizonte.

def selecionar_picos(serie, quantidade, antes_h, depois_h):
    """Seleciona picos altos, sem sobreposicao das janelas de avaliacao."""
    candidatos = serie.sort_values(ascending=False)
    escolhidos = []
    distancia_minima = pd.Timedelta(hours=antes_h + depois_h)
    for data, valor in candidatos.items():
        if all(abs(data - anterior) > distancia_minima for anterior in escolhidos):
            escolhidos.append(data)
            if len(escolhidos) == quantidade:
                break
    return sorted(escolhidos)

horizontes = sorted(set(h for _, h in previsoes_teste))
# Datas comuns aos modelos/horizontes: comparacao justa dos eventos.
indices = [set(tab.index) for tab in previsoes_teste.values()]
datas_comuns = pd.DatetimeIndex(sorted(set.intersection(*indices)))
if len(datas_comuns) == 0:
    raise ValueError('Nao ha datas comuns de teste entre os modelos/horizontes.')
obs_referencia = previsoes_teste[(MODELOS[0], horizontes[0])].loc[
    datas_comuns, 'observado']
picos_eventos = [obs_referencia.idxmax()]
linhas_eventos = []
fig, axes = plt.subplots(len(picos_eventos), len(horizontes),
                         figsize=(5.8*len(horizontes), 3.8*len(picos_eventos)),
                         squeeze=False)
cores = {'LSTM': 'tab:blue', 'MLP': 'tab:orange'}
for ie, data_pico in enumerate(picos_eventos):
    inicio = data_pico - pd.Timedelta(hours=JANELA_ANTES_PICO_H)
    fim = data_pico + pd.Timedelta(hours=JANELA_DEPOIS_PICO_H)
    for ih, h in enumerate(horizontes):
        ax = axes[ie, ih]
        ref = obs_referencia.loc[inicio:fim]
        ax.plot(ref.index, ref.values, color='black', lw=1.8, label='Observado')
        for nome in MODELOS:
            tab = previsoes_teste[(nome, h)].reindex(ref.index).dropna()
            if len(tab) < 2:
                continue
            # Para metricas, observacao e previsao usam exatamente as mesmas datas.
            pico_obs = float(tab.observado.max())
            pico_prev = float(tab.previsto.max())
            data_obs = tab.observado.idxmax()
            data_prev = tab.previsto.idxmax()
            erro_pico = 100*(pico_prev-pico_obs)/pico_obs if pico_obs != 0 else np.nan
            erro_tempo = (data_prev-data_obs).total_seconds()/3600
            nse_evento = metricas(tab.observado,tab.previsto)['NSE']
            linhas_eventos.append(dict(evento=ie+1, data_pico_referencia=data_pico,
                                       modelo=nome, horizonte_h=h, n_pontos=len(tab),
                                       erro_pico_pct=erro_pico,
                                       erro_tempo_pico_h=erro_tempo,
                                       NSE_evento=nse_evento))
            etiqueta = (f'{nome} | NSE={nse_evento:.2f} | '
                        f'EP={erro_pico:+.1f}% | ET={erro_tempo:+.0f}h')
            ax.plot(tab.index, tab.previsto, lw=1.1,
                    color=cores.get(nome), label=etiqueta)
        ax.set(title=f'Evento {ie+1} | +{h} h | pico obs.: {data_pico:%d/%m %Hh}',
               ylabel='Vazao', xlabel='Data')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, loc='best')
        ax.tick_params(axis='x', labelrotation=20)
plt.tight_layout()
plt.savefig(SAIDA/'eventos_comparacao_LSTM_MLP.png',dpi=180)
plt.show()
tabela_eventos = pd.DataFrame(linhas_eventos)
tabela_eventos.to_csv(SAIDA/'metricas_eventos.csv',sep=';',index=False)
print('\nMETRICAS POR EVENTO (EP positivo = pico superestimado; ET positivo = atrasado):')
print(tabela_eventos.to_string(index=False))

# %% 11 | NSE, KGE, PBIAS E FHV 2% POR HORIZONTE
# FHV 2% = 100 * soma(Qprev - Qobs) / soma(Qobs), considerando apenas
# os 2% maiores valores OBSERVADOS do periodo de teste.
# FHV > 0 indica superestimativa do volume nas maiores vazoes.
def fhv_2pct(obs, pred):
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    k = max(1, int(np.ceil(0.02*len(obs))))
    indices_top = np.argsort(obs)[-k:]
    denominador = np.sum(obs[indices_top])
    if denominador == 0:
        return np.nan
    return 100*np.sum(pred[indices_top]-obs[indices_top])/denominador

resumo['FHV_2pct'] = [fhv_2pct(previsoes_teste[(linha.modelo, linha.horizonte_h)].observado,
                              previsoes_teste[(linha.modelo, linha.horizonte_h)].previsto)
                      for linha in resumo.itertuples(index=False)]
resumo.to_csv(SAIDA/'comparacao_teste.csv',sep=';',index=False)
metricas_plot = [('NSE','NSE'),('KGE','KGE'),('PBIAS','PBIAS (%)'),
                 ('FHV_2pct','FHV 2% (%)')]
fig, axes = plt.subplots(2,2,figsize=(11,8),sharex=True)
for ax,(coluna,rotulo) in zip(axes.flat,metricas_plot):
    for nome, sub in resumo.groupby('modelo'):
        sub = sub.sort_values('horizonte_h')
        ax.plot(sub.horizonte_h, sub[coluna], 'o-', label=nome)
    if coluna in ('PBIAS','FHV_2pct'):
        ax.axhline(0,color='0.5',ls='--',lw=.9)
    ax.set(ylabel=rotulo, xlabel='Horizonte de previsao (h)')
    ax.grid(alpha=.2)
    ax.legend()
plt.suptitle('Metricas no teste independente | LSTM vs MLP')
plt.tight_layout()
plt.savefig(SAIDA/'metricas_por_horizonte.png',dpi=180)
plt.show()
print('\nNSE, KGE, PBIAS e FHV 2% (teste):')
print(resumo[['modelo','horizonte_h','NSE','KGE','PBIAS','FHV_2pct']].to_string(index=False))
print('Convencoes: PBIAS > 0 = subestimativa; FHV > 0 = superestimativa dos 2% maiores Qobs.')

# %% 12 | SALVAR MODELOS COMPLETOS (PESOS + CONFIGURACAO + NORMALIZACAO)
# Para reutilizar uma rede, nao basta salvar somente os pesos: tambem precisamos
# guardar a arquitetura, as colunas e as escalas aprendidas no TREINO.
# Um arquivo .pt por modelo/horizonte, sem serializar objetos Python arbitrarios.
PASTA_MODELOS.mkdir(parents=True, exist_ok=True)
for (nome,h), melhor in melhores.items():
    cfg = {k:melhor[k] for k in ('modelo','horizonte_h','janela_h','camadas','unidades')}
    dados = preparar(cfg)
    pacote = {
        'config': cfg,
        'features': list(FEATURES),
        'coluna_vazao': COL_Q,
        'state_dict': melhor['estado'],
        'sx_mean': dados['sx'].mean_.tolist(),
        'sx_scale': dados['sx'].scale_.tolist(),
        'sy_mean': dados['sy'].mean_.tolist(),
        'sy_scale': dados['sy'].scale_.tolist(),
        'fim_treino': str(FIM_TREINO),
        'fim_validacao': str(FIM_VALIDACAO),
    }
    caminho = PASTA_MODELOS / f'{nome}_h{h}.pt'
    torch.save(pacote, caminho)
    print('Modelo salvo:', caminho)
print('Arquivos completos em:', PASTA_MODELOS)

# %% 13 | REABRIR UM MODELO E PREVER O TESTE (CELULA INDEPENDENTE)
# DEMONSTRACAO: no Spyder, use "Remover todas as variaveis" e execute SOMENTE
# esta celula. Ela importa bibliotecas, le o CSV, reconstroi a rede e aplica
# as medias/desvios salvos. NAO precisa executar as celulas anteriores.
# Os arquivos .pt devem ter sido criados anteriormente na celula 12.

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn

# --- 1. Escolha qual modelo salvo deseja abrir ---
MODELO_ABRIR = 'LSTM'      # 'LSTM' ou 'MLP'
HORIZONTE_ABRIR = 1       # 1, 3 ou 6 horas (conforme modelos salvos)
PASTA_AULA = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
CSV_TESTE = PASTA_AULA / 'series_horarias_preenchidas.csv'
PASTA_SALVOS = PASTA_AULA / 'resultados_LSTM_vs_MLP' / 'modelos_completos'
ARQUIVO_MODELO = PASTA_SALVOS / f'{MODELO_ABRIR}_h{HORIZONTE_ABRIR}.pt'
DISPOSITIVO = torch.device('cpu')  # funciona mesmo sem GPU

# %% --- 2. Abrir o pacote com os pesos e os parametros de normalizacao ---
if not ARQUIVO_MODELO.exists():
    raise FileNotFoundError(f'Modelo nao encontrado: {ARQUIVO_MODELO}')
if not CSV_TESTE.exists():
    raise FileNotFoundError(f'Serie horaria nao encontrada: {CSV_TESTE}')
pacote = torch.load(ARQUIVO_MODELO, map_location='cpu', weights_only=True)
config = pacote['config']
variaveis = pacote['features']
coluna_q = pacote['coluna_vazao']
print('Modelo aberto:', ARQUIVO_MODELO.name)
print('Configuracao:', config)
print('Entradas:', variaveis)

# %%--- 3. Reconstruir a arquitetura (mesma arquitetura da celula 04) ---
class LSTMReaberta(nn.Module):
    def __init__(self, n_features, unidades, camadas):
        super().__init__()
        self.lstm = nn.LSTM(n_features, unidades, num_layers=camadas,
                            batch_first=True)
        self.saida = nn.Linear(unidades, 1)

    def forward(self, x):
        sequencia, _ = self.lstm(x)
        return self.saida(sequencia[:, -1, :])

class MLPReaberta(nn.Module):
    def __init__(self, n_features, janela, unidades, camadas):
        super().__init__()
        blocos = []
        entrada = n_features * janela
        for _ in range(camadas):
            blocos.extend([nn.Linear(entrada, unidades), nn.ReLU()])
            entrada = unidades
        blocos.append(nn.Linear(entrada, 1))
        self.net = nn.Sequential(*blocos)

    def forward(self, x):
        return self.net(x.flatten(start_dim=1))

if config['modelo'] == 'LSTM':
    rede = LSTMReaberta(len(variaveis), config['unidades'], config['camadas'])
elif config['modelo'] == 'MLP':
    rede = MLPReaberta(len(variaveis), config['janela_h'],
                       config['unidades'], config['camadas'])
else:
    raise ValueError('Arquitetura desconhecida no arquivo salvo.')
rede.load_state_dict(pacote['state_dict'])
rede.to(DISPOSITIVO).eval()
print('Pesos restaurados. Modelo pronto para previsao.')

# %% --- 4. Ler a serie original e reconstruir as janelas ---
serie = pd.read_csv(CSV_TESTE, sep=';', encoding='utf-8-sig')
serie['data_hora'] = pd.to_datetime(serie['data_hora'])
serie = serie.sort_values('data_hora').set_index('data_hora').asfreq('h')
for coluna in set(variaveis + [coluna_q]):
    if coluna not in serie:
        raise ValueError(f'Coluna ausente no CSV: {coluna}')
    serie[coluna] = pd.to_numeric(serie[coluna], errors='coerce')

janela = int(config['janela_h'])
horizonte = int(config['horizonte_h'])
valores = serie[variaveis].to_numpy(dtype=np.float32)
q = serie[coluna_q].to_numpy(dtype=np.float32)
X_lista, y_lista, datas_lista = [], [], []
for i in range(janela - 1, len(serie) - horizonte):
    x = valores[i - janela + 1:i + 1]
    y = q[i + horizonte]
    if np.isfinite(x).all() and np.isfinite(y):
        X_lista.append(x)
        y_lista.append(y)
        datas_lista.append(serie.index[i + horizonte])

if not X_lista:
    raise ValueError('Nao ha janelas validas para previsao.')
X = np.stack(X_lista)
y = np.asarray(y_lista)
datas = pd.DatetimeIndex(datas_lista)

# O inicio do TESTE e definido pelo limite de validacao salvo no .pt.
# Assim, nao dependemos das variaveis da sessao de treinamento.
mascara_teste = np.asarray(datas > pd.Timestamp(pacote['fim_validacao']))
X_teste = X[mascara_teste]
y_teste = y[mascara_teste]
datas_teste = datas[mascara_teste]
if len(X_teste) == 0:
    raise ValueError('O conjunto de teste ficou vazio.')

# %% --- 5. Aplicar a MESMA normalizacao aprendida no treino ---
media_x = np.asarray(pacote['sx_mean'], dtype=np.float32)
desvio_x = np.asarray(pacote['sx_scale'], dtype=np.float32)
X_teste_norm = ((X_teste - media_x) / desvio_x).astype(np.float32)

# %%--- 6. Executar o modelo salvo no TESTE, sem treinar ---
previsoes_norm = []
with torch.no_grad():
    for inicio in range(0, len(X_teste_norm), 1024):
        xb = torch.from_numpy(X_teste_norm[inicio:inicio + 1024]).to(DISPOSITIVO)
        previsoes_norm.append(rede(xb).cpu().numpy().ravel())
previsoes_norm = np.concatenate(previsoes_norm)
previsoes = (previsoes_norm * float(pacote['sy_scale'][0])
             + float(pacote['sy_mean'][0]))
previsoes = np.maximum(previsoes, 0.0)

    # %% --- 7. Avaliar e visualizar o TESTE completo ---
resultado_teste_reaberto = pd.DataFrame(
    {'observado': y_teste, 'previsto': previsoes}, index=datas_teste)
resultado_teste_reaberto.index.name = 'data_alvo'
denominador = np.sum((y_teste - y_teste.mean()) ** 2)
nse_teste = (1 - np.sum((previsoes - y_teste) ** 2) / denominador
             if denominador > 0 else np.nan)
print(f'Teste: {len(resultado_teste_reaberto)} previsoes')
print(f'NSE no teste: {nse_teste:.4f}')

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(resultado_teste_reaberto.index, resultado_teste_reaberto['observado'],
        color='black', lw=0.8, label='Observado')
ax.plot(resultado_teste_reaberto.index, resultado_teste_reaberto['previsto'],
        lw=0.8, label=f'{MODELO_ABRIR} +{HORIZONTE_ABRIR}h | NSE={nse_teste:.3f}')
ax.set(title='Modelo reaberto sem treinamento | Teste independente',
       xlabel='Data-alvo', ylabel='Vazao')
ax.grid(alpha=0.2)
ax.legend()
plt.tight_layout()
plt.show()
