# %% 00 | OBJETIVO E IMPORTAÇÕES
"""HydroUAI | MLP para previsão chuva-vazão horária.

Experimentos com grid search de:
- número de camadas ocultas e unidades por camada;
- conjuntos opcionais de lags e janelas hidrológicas;
- comparação P e P+Q; horizontes independentes.

A seleção é feita APENAS pela validação temporal. Teste reservado ao fim.
Abra no Spyder e execute as células em ordem (Ctrl+Enter).
CSV: series_horarias_preenchidas.csv, na mesma pasta.
pip install numpy pandas matplotlib scikit-learn torch
"""
from pathlib import Path
from itertools import product
import copy
import random
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# %% 01 | CONFIGURAÇÕES — ALTERE AQUI
PASTA = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
ARQUIVO = PASTA / 'series_horarias_preenchidas.csv'
SAIDA = PASTA / 'resultados_MLP_grid'
COL_DATA = 'data_hora'
COL_P = ['2463_P', '84100000_P']
COL_Q = '2463_Q'

HORIZONTES_H = [1,3,6]             # Exemplo: [1, 3, 6, 12, 24]
MODELOS = ['MLP_PQ']  # Somente chuva; chuva + vazão
# Porcentagens da série cronológica: precisam somar 100.
PCT_TREINO = 70
PCT_VALIDACAO = 15
PCT_TESTE = 15
if PCT_TREINO <= 0 or PCT_VALIDACAO <= 0 or PCT_TESTE <= 0 or abs(PCT_TREINO + PCT_VALIDACAO + PCT_TESTE - 100) > 1e-8:
    raise ValueError('As porcentagens devem ser positivas e somar 100.')

# Configurações apenas para os gráficos didáticos.
HORIZONTE_EXEMPLO = HORIZONTES_H[0]
MODELO_PLOT = MODELOS[0]
N_BATCHES_COLORIDOS = 30
# None = mostrar todo o teste; número inteiro = mostrar apenas as primeiras N horas.
HORAS_PLOT_TESTE = None

# Grid hidrológico: agrupamento da chuva e memória antecedente.
# AGRUPAMENTO_CHUVA_H = 1: valores horários individuais.
# AGRUPAMENTO_CHUVA_H = 2: somas em blocos consecutivos de 2 h.
# JANELA_CHUVA_H = 24: considerar as últimas 24 h até a origem t.
# JANELA_Q_H = 12: usar Q(t), Q(t-1), ..., Q(t-12).
# Cada lista abaixo contém as opções testadas pelo Grid Search.
AGRUPAMENTO_CHUVA_H_GRID = [1]
JANELA_CHUVA_H_GRID = [6]
JANELA_Q_H_GRID = [6]

# Exemplo: agrupamento 2 h e janela de 6 h gera:
# P(t-1)+P(t), P(t-3)+P(t-2), P(t-5)+P(t-4).
# JANELA_CHUVA_H deve ser múltipla de AGRUPAMENTO_CHUVA_H.
# Não são usadas médias de vazão nem tendências calculadas.

# Grid da arquitetura: N_CAMADAS × N_UNIDADES (mesma largura em todas
# as camadas ocultas de uma dada combinação).
GRID_N_CAMADAS = [1]
GRID_N_UNIDADES = [32]

EPOCAS_MAX = 100
BATCH_SIZE = 256
LR_INICIAL = 0.005
WEIGHT_DECAY = 0.0
USAR_LR_ADAPTATIVO = True
PACIENCIA_LR = 5
FATOR_LR = 0.5
LR_MIN = 1e-5
USAR_EARLY_STOPPING = True
PACIENCIA_EARLY = 12
MELHORIA_MINIMA = 1e-6
SEMENTE = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Dispositivo:', DEVICE)

# %% 02 | LEITURA E CONFERÊNCIA DOS DADOS
if not ARQUIVO.exists():
    raise FileNotFoundError(f'CSV não encontrado: {ARQUIVO}')
df = pd.read_csv(ARQUIVO, sep=';', encoding='utf-8-sig')
cols = [COL_DATA, *COL_P, COL_Q]
if not set(cols).issubset(df.columns):
    raise ValueError(f'Colunas necessárias: {cols}')
df[COL_DATA] = pd.to_datetime(df[COL_DATA], errors='raise')
if df[COL_DATA].duplicated().any():
    raise ValueError('Existem datas/horas duplicadas.')
df = df.sort_values(COL_DATA).set_index(COL_DATA).asfreq('h')
for c in [*COL_P, COL_Q]:
    df[c] = pd.to_numeric(df[c], errors='coerce')
    if (df[c].dropna() < 0).any():
        raise ValueError(f'Valores negativos em {c}; confira a base.')
print('Período:', df.index.min(), 'a', df.index.max())
print('Ausências:\n', df[[*COL_P, COL_Q]].isna().sum())
print(df.head())

# %% 03 | VISUALIZAR CHUVA E VAZÃO
trecho = df[[*COL_P, COL_Q]].iloc[:24 * 90]
fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
for posto in COL_P:
    ax[0].plot(trecho.index, trecho[posto], label=posto, alpha=0.8)
ax[0].set_ylabel('Precipitação')
ax[0].legend()
ax[1].plot(trecho.index, trecho[COL_Q], label=COL_Q)
ax[1].set_ylabel('Vazão')
ax[1].set_xlabel('Data')
plt.tight_layout()
plt.show()

# %% 03B | SEPARAÇÃO TEMPORAL: TREINO, VALIDAÇÃO E TESTE
# Dividimos a série em ordem cronológica; nenhuma amostra é sorteada entre períodos.
n_total = len(df)
i_treino = int(n_total * PCT_TREINO / 100)
i_validacao = int(n_total * (PCT_TREINO + PCT_VALIDACAO) / 100)
if not (0 < i_treino < i_validacao < n_total):
    raise ValueError('Poucos registros para as porcentagens escolhidas.')
FIM_TREINO = df.index[i_treino - 1]
FIM_VALIDACAO = df.index[i_validacao - 1]

fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(df.index, df[COL_Q], color='0.3', linewidth=0.7, label='Vazão observada')
periodos = [
    ('Treinamento', df.index[0], FIM_TREINO, 'tab:blue'),
    ('Validação', df.index[i_treino], FIM_VALIDACAO, 'tab:orange'),
    ('Teste', df.index[i_validacao], df.index[-1], 'tab:green'),
]
for nome, inicio, fim, cor in periodos:
    ax.axvspan(inicio, fim, color=cor, alpha=0.17, label=nome)
    print(f'{nome:12s}: {inicio} a {fim}')
ax.set(title='Série completa e divisão cronológica', xlabel='Data', ylabel='Vazão')
ax.legend(ncol=4)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.show()

# %% 04 | ATRIBUTOS: CHUVA AGRUPADA E VAZÕES ANTECEDENTES
def construir_atributos(dados, modelo, agrupamento_p, janela_p, janela_q):
    """Entradas conhecidas na origem t, sem qualquer dado futuro.

    A chuva é somada em blocos não sobrepostos de `agrupamento_p` horas.
    Para agrupamento 1, cada bloco contém uma hora individual.
    Somente em MLP_PQ, entram Q(t), Q(t-1), ..., Q(t-janela_q).
    """
    if modelo not in ('MLP_P', 'MLP_PQ'):
        raise ValueError(f'Modelo desconhecido: {modelo}')
    if modelo == 'MLP_PQ' and (janela_q is None or janela_q < 0):
        raise ValueError('JANELA_Q_H deve ser >= 0 para MLP_PQ.')
    if modelo in ('MLP_P', 'MLP_PQ'):
        if agrupamento_p < 1 or janela_p < 1 or janela_p % agrupamento_p:
            raise ValueError('A janela de chuva deve ser múltipla do agrupamento.')
    f = pd.DataFrame(index=dados.index)
    if modelo in ('MLP_P', 'MLP_PQ'):
        for posto in COL_P:
            for inicio in range(0, janela_p, agrupamento_p):
                # Bloco mais recente: P(t), P(t-1), ...;
                # bloco seguinte: P(t-agrupamento), ...
                bloco = sum(dados[posto].shift(lag)
                            for lag in range(inicio, inicio + agrupamento_p))
                if agrupamento_p == 1:
                    tempo = 't' if inicio == 0 else f't-{inicio}h'
                    nome = f'{posto}_{tempo}'
                else:
                    fim = inicio + agrupamento_p - 1
                    tempo_inicial = 't' if fim == 0 else f't-{fim}h'
                    tempo_final = 't' if inicio == 0 else f't-{inicio}h'
                    nome = f'{posto}_acum_{tempo_inicial}_a_{tempo_final}'
                
                f[nome] = bloco
    if modelo == 'MLP_PQ':
        for lag in range(janela_q + 1):
            f[f'Q_lag_{lag}h'] = dados[COL_Q].shift(lag)
    return f

def construir_amostras(dados, atributos, horizonte):
    # Origem t -> alvo Q(t+h); datas de partição são as datas do ALVO.
    y = dados[COL_Q].shift(-horizonte)
    valido = atributos.notna().all(axis=1) & y.notna()
    x = atributos.loc[valido].to_numpy(dtype=np.float64)
    alvo = y.loc[valido].to_numpy(dtype=np.float64)
    datas_alvo = atributos.index[valido] + pd.Timedelta(hours=horizonte)
    return x, alvo, datas_alvo

def mascaras_temporais(datas):
    # Os limites são definidos por posição na série original, não por amostra.
    # Assim, todos os horizontes usam as mesmas datas de separação.
    tr = np.asarray(datas <= FIM_TREINO)
    va = np.asarray((datas > FIM_TREINO) & (datas <= FIM_VALIDACAO))
    te = np.asarray(datas > FIM_VALIDACAO)
    return tr, va, te

# %% 04B | MOSTRAR A TABELA DE ENTRADAS ANTES DOS TENSORES
# Aqui os atributos são DataFrames do pandas, ainda em unidades originais.
# Escolhemos a primeira combinação do grid apenas para ilustrar.
MODELO_EXEMPLO = MODELOS[0]
AGRUPAMENTO_EXEMPLO = AGRUPAMENTO_CHUVA_H_GRID[0]
JANELA_CHUVA_EXEMPLO = JANELA_CHUVA_H_GRID[0]
JANELA_Q_EXEMPLO = JANELA_Q_H_GRID[0]

atributos_exemplo = construir_atributos(
    df, MODELO_EXEMPLO, AGRUPAMENTO_EXEMPLO,
    JANELA_CHUVA_EXEMPLO, JANELA_Q_EXEMPLO)
alvo_exemplo = df[COL_Q].shift(-HORIZONTE_EXEMPLO)

# Variável importante para a aula: colunas = chuva agrupada e Q defasada.
DADOS_ENTRADA_DIDATICOS = atributos_exemplo.copy()
TABELA_DIDATICA = DADOS_ENTRADA_DIDATICOS.copy()
TABELA_DIDATICA[f'ALVO_Q_t+{HORIZONTE_EXEMPLO}h'] = alvo_exemplo
TABELA_DIDATICA = TABELA_DIDATICA.dropna()

print('\nCOLUNAS DE ENTRADA (ANTES DE NORMALIZAR E CRIAR TENSORES):')
print(DADOS_ENTRADA_DIDATICOS.columns.tolist())
print('\nPRIMEIRAS 8 AMOSTRAS COMPLETAS:')
print(TABELA_DIDATICA.head(8).to_string())
print('\nUma amostra específica (linha 1):')
print(TABELA_DIDATICA.iloc[0].to_string())

# %% 05 | ARQUITETURA MLP, TREINAMENTO E MÉTRICAS
def fixar_semente():
    random.seed(SEMENTE)
    np.random.seed(SEMENTE)
    torch.manual_seed(SEMENTE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEMENTE)

def criar_mlp(n_entradas, n_camadas, n_unidades):
    blocos = []
    anterior = n_entradas
    for _ in range(n_camadas):
        blocos += [nn.Linear(anterior, n_unidades), nn.ReLU()]
        anterior = n_unidades
    blocos.append(nn.Linear(anterior, 1))
    return nn.Sequential(*blocos).to(DEVICE)

def metricas(obs, pred):
    obs, pred = np.asarray(obs), np.asarray(pred)
    erro = pred - obs
    den = np.sum((obs - obs.mean()) ** 2)
    nse = 1 - np.sum(erro**2) / den if den > 0 else np.nan
    rmse = np.sqrt(np.mean(erro**2))
    mae = np.mean(np.abs(erro))
    r = np.corrcoef(obs, pred)[0, 1] if np.std(obs) and np.std(pred) else np.nan
    alpha = np.std(pred) / np.std(obs) if np.std(obs) else np.nan
    beta = np.mean(pred) / np.mean(obs) if np.mean(obs) else np.nan
    kge = 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)
    pbias = 100 * np.sum(obs - pred) / np.sum(obs) if np.sum(obs) != 0 else np.nan
    # Convenção hidrológica: PBIAS > 0 indica subestimativa.
    return dict(NSE=nse, KGE=kge, RMSE=rmse, MAE=mae, PBIAS=pbias)

def treinar_mlp(xtr, ytr, xva, yva, n_camadas, n_unidades):
    rede = criar_mlp(xtr.shape[1], n_camadas, n_unidades)
    opt = torch.optim.Adam(rede.parameters(), lr=LR_INICIAL,
                           weight_decay=WEIGHT_DECAY)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=FATOR_LR, patience=PACIENCIA_LR,
        min_lr=LR_MIN
    ) if USAR_LR_ADAPTATIVO else None)
    criterio = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(xtr, dtype=torch.float32),
                      torch.tensor(ytr[:, None], dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=False
    )
    xv = torch.tensor(xva, dtype=torch.float32, device=DEVICE)
    yv = torch.tensor(yva[:, None], dtype=torch.float32, device=DEVICE)
    melhor, estado, espera = float('inf'), None, 0
    historico = []
    melhor_epoca = 0
    for epoca in range(1, EPOCAS_MAX + 1):
        rede.train()
        soma, n = 0., 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            perda = criterio(rede(xb), yb)
            perda.backward()
            opt.step()
            soma += perda.item() * len(xb)
            n += len(xb)
        rede.eval()
        with torch.no_grad():
            perda_val = criterio(rede(xv), yv).item()
        perda_tr = soma / n
        lr_atual = opt.param_groups[0]['lr']
        historico.append(dict(epoca=epoca, treino=perda_tr,
                              validacao=perda_val, lr=lr_atual))
        if perda_val < melhor - MELHORIA_MINIMA:
            melhor, estado, espera = perda_val, copy.deepcopy(rede.state_dict()), 0
            melhor_epoca = epoca
        else:
            espera += 1
        if scheduler is not None:
            scheduler.step(perda_val)
        if epoca == 1 or epoca % 10 == 0:
            print(f'  Época {epoca:03d} | treino={perda_tr:.5f} '
                  f'val={perda_val:.5f} | LR={lr_atual:.6g}')
        if USAR_EARLY_STOPPING and espera >= PACIENCIA_EARLY:
            print(f'  Early stopping na época {epoca}')
            break
    # Mesmo sem early stopping, usar os melhores pesos da validação.
    rede.load_state_dict(estado)
    return rede, pd.DataFrame(historico), melhor, melhor_epoca

def prever(rede, x):
    rede.eval()
    partes = []
    with torch.no_grad():
        for i in range(0, len(x), BATCH_SIZE):
            xb = torch.tensor(x[i:i+BATCH_SIZE], dtype=torch.float32, device=DEVICE)
            partes.append(rede(xb).cpu().numpy())
    return np.concatenate(partes).ravel()

# %% 06 | MONTAR GRID SEARCH E CONTAR TREINAMENTOS
# MLP_P não usa vazão: evita repetir janelas de Q equivalentes.
grade = []
for horizonte in HORIZONTES_H:
    for modelo in MODELOS:
        opcoes_chuva = (
            list(product(AGRUPAMENTO_CHUVA_H_GRID, JANELA_CHUVA_H_GRID))
            if modelo in ('MLP_P', 'MLP_PQ') else [(None, None)]
        )
        for agrupamento_p, janela_p in opcoes_chuva:
            if janela_p % agrupamento_p != 0:
                print('Ignorando combinação incompatível:', agrupamento_p, janela_p)
                continue
            opcoes_q = JANELA_Q_H_GRID if modelo == 'MLP_PQ' else [None]
            for janela_q in opcoes_q:
                for n_camadas, n_unidades in product(
                    GRID_N_CAMADAS, GRID_N_UNIDADES
                ):
                    grade.append(dict(
                        horizonte_h=horizonte, modelo=modelo,
                        agrupamento_p=agrupamento_p, janela_p=janela_p,
                        janela_q=janela_q, n_camadas=n_camadas,
                        n_unidades=n_unidades
                    ))
print(f'Total de treinamentos no grid: {len(grade)}')
print(pd.DataFrame(grade).head(10).to_string(index=False))

# %% 06B | VISUALIZAR OS BATCHES NA SÉRIE DE TREINAMENTO
# Representação didática: batches consecutivos, sem embaralhamento.

config_exemplo = grade[0]

atr = construir_atributos(
    df,
    config_exemplo['modelo'],
    config_exemplo['agrupamento_p'],
    config_exemplo['janela_p'],
    config_exemplo['janela_q']
)

xx, yy, datas_exemplo = construir_amostras(
    df, atr, config_exemplo['horizonte_h']
)

tr, va, te = mascaras_temporais(datas_exemplo)

datas_treino = pd.DatetimeIndex(datas_exemplo[tr])
q_treino = yy[tr]

# Quantidade total de batches
n_batches = int(np.ceil(len(q_treino) / BATCH_SIZE))

fig, ax = plt.subplots(figsize=(14, 5))

# Série completa de treinamento
ax.plot(
    datas_treino,
    q_treino,
    color='black',
    linewidth=0.8,
    label='Vazão observada'
)

# Destacar os primeiros batches com hachuras
for b in range(min(N_BATCHES_COLORIDOS, n_batches)):

    inicio = b * BATCH_SIZE
    fim = min((b + 1) * BATCH_SIZE, len(q_treino))

    ax.axvspan(
        datas_treino[inicio],
        datas_treino[fim - 1],
        facecolor=f'C{b % 10}',
        alpha=0.12,
        hatch='///',
        edgecolor=f'C{b % 10}',
        linewidth=0,
        label=f'Batch {b + 1}'
    )

ax.set(
    title=f'Divisão dos dados de treinamento em batches | Batch size = {BATCH_SIZE}',
    xlabel='Data',
    ylabel='Vazão'
)

ax.legend(ncol=4, fontsize=9)
ax.grid(alpha=0.2)

plt.tight_layout()
plt.show()

print(f'Amostras de treinamento: {len(q_treino)}')
print(f'Tamanho do batch: {BATCH_SIZE}')
print(f'Número total de batches: {n_batches}')

# %% 07 | GRID SEARCH: TREINAR E SELECIONAR PELA VALIDAÇÃO
# IMPORTANTE: não consultar o conjunto de teste durante a escolha da arquitetura.
fixar_semente()
SAIDA.mkdir(parents=True, exist_ok=True)
resultados_grid = []
melhores = {}
for i, cfg in enumerate(grade, 1):
    horizonte = cfg['horizonte_h']
    modelo = cfg['modelo']
    print(f'\n[{i}/{len(grade)}] {modelo} +{horizonte}h '
          f'| {cfg["n_camadas"]} camada(s) × {cfg["n_unidades"]} unidades')
    atributos = construir_atributos(df, modelo, cfg['agrupamento_p'],
                                    cfg['janela_p'], cfg['janela_q'])
    x, y, datas = construir_amostras(df, atributos, horizonte)
    tr, va, te = mascaras_temporais(datas)
    if not tr.any() or not va.any():
        raise ValueError(f'Treino/validação vazios: {cfg}')
    # Escalas ajustadas exclusivamente nas amostras de treinamento.
    sx = StandardScaler().fit(x[tr])
    sy = StandardScaler().fit(y[tr, None])
    xt, xv = sx.transform(x[tr]), sx.transform(x[va])
    yt = sy.transform(y[tr, None]).ravel()
    yv = sy.transform(y[va, None]).ravel()
    fixar_semente()
    rede, hist, mse_val_norm, melhor_epoca = treinar_mlp(
        xt, yt, xv, yv, cfg['n_camadas'], cfg['n_unidades'])
    # MSE em escala original para comparar configurações com escalas distintas.
    pred_val = sy.inverse_transform(prever(rede, xv).reshape(-1, 1)).ravel()
    mse_val = float(np.mean((pred_val - y[va])**2))
    linha = {**cfg, 'n_atributos': x.shape[1], 'n_treino': int(tr.sum()),
             'n_validacao': int(va.sum()), 'MSE_validacao': mse_val,
             'RMSE_validacao': float(np.sqrt(mse_val)),
             'NSE_validacao': metricas(y[va], pred_val)['NSE'],
             'epocas_executadas': len(hist)}
    resultados_grid.append(linha)
    chave = (modelo, horizonte)
    if chave not in melhores or mse_val < melhores[chave]['MSE_validacao']:
        melhores[chave] = {**linha, 'rede': copy.deepcopy(rede).cpu(),
                          'sx': sx, 'sy': sy, 'atributos': list(atributos.columns),
                          'historico': hist}
        print('  -> Nova melhor configuração na validação.')
    pd.DataFrame(resultados_grid).to_csv(
        SAIDA / 'grid_search_validacao.csv', sep=';', index=False)

tabela_grid = pd.DataFrame(resultados_grid)
print('\nMelhores resultados de validação:')
print(tabela_grid.sort_values('MSE_validacao').groupby(
    ['modelo', 'horizonte_h'], as_index=False).first()[
    ['modelo', 'horizonte_h', 'n_camadas', 'n_unidades',
     'RMSE_validacao', 'NSE_validacao']].to_string(index=False))

# %% 08 | TESTE FINAL — SOMENTE MELHOR CONFIGURAÇÃO DE CADA MODELO/HORIZONTE
# Avaliação única em teste após a seleção por validação.
resumo_teste = []
previsoes_teste = {}
for (modelo, horizonte), melhor in melhores.items():
    atributos = construir_atributos(
        df, modelo, melhor['agrupamento_p'],
        melhor['janela_p'], melhor['janela_q'])
    x, y, datas = construir_amostras(df, atributos, horizonte)
    tr, va, te = mascaras_temporais(datas)
    if not te.any():
        raise ValueError('Teste vazio: confira FIM_VALIDACAO.')
    xte = melhor['sx'].transform(x[te])
    rede = melhor['rede'].to(DEVICE)
    previsto = melhor['sy'].inverse_transform(
        prever(rede, xte).reshape(-1, 1)).ravel()
    previsto = np.maximum(previsto, 0.0)
    tabela = pd.DataFrame({
        'observado': y[te], 'previsto': previsto
    }, index=datas[te])
    tabela.index.name = 'data_alvo'
    previsoes_teste[(modelo, horizonte)] = tabela
    met = metricas(y[te], previsto)
    resumo_teste.append({
        'modelo': modelo, 'horizonte_h': horizonte,
        'n_camadas': melhor['n_camadas'], 'n_unidades': melhor['n_unidades'],
        'agrupamento_p': melhor['agrupamento_p'],
        'janela_p': melhor['janela_p'], 'janela_q': melhor['janela_q'],
        'MSE_validacao': melhor['MSE_validacao'], **met
    })
    prefixo = f'{modelo}_h{horizonte}'
    tabela.to_csv(SAIDA / f'previsoes_{prefixo}.csv', sep=';')
    melhor['historico'].to_csv(SAIDA / f'perdas_{prefixo}.csv',
                              sep=';', index=False)
    torch.save(rede.cpu().state_dict(), SAIDA / f'pesos_{prefixo}.pt')
    (SAIDA / f'config_{prefixo}.json').write_text(json.dumps(
        {k: (list(v) if isinstance(v, tuple) else v)
         for k, v in melhor.items()
         if k not in {'rede', 'sx', 'sy', 'historico'}},
        ensure_ascii=False, indent=2), encoding='utf-8')
resumo = pd.DataFrame(resumo_teste)
resumo.to_csv(SAIDA / 'comparacao_teste.csv', sep=';', index=False)
print('\nAVALIAÇÃO FINAL NO TESTE')
print(resumo[['modelo', 'horizonte_h', 'n_camadas', 'n_unidades',
              'NSE', 'PBIAS', 'KGE', 'RMSE']].to_string(index=False))

# %% 09 | CONVERGÊNCIA E LEARNING RATE DO MELHOR MODELO
# A linha vertical marca a época cujos pesos foram restaurados.
# A faixa final indica a espera desde a melhor época até a parada.
HORIZONTE_PLOT = HORIZONTES_H[0]
MODELO_PLOT = MODELOS[0]
melhor = melhores[(MODELO_PLOT, HORIZONTE_PLOT)]
hist = melhor['historico']
epoca_melhor = int(hist.loc[hist['validacao'].idxmin(), 'epoca'])
epoca_final = int(hist.epoca.iloc[-1])

fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                       gridspec_kw={'height_ratios': [2, 1]})
ax[0].plot(hist.epoca, hist.treino, label='Treinamento')
ax[0].plot(hist.epoca, hist.validacao, label='Validação')
ax[0].axvline(epoca_melhor, color='tab:green', linestyle='--',
              label=f'Melhor época / pesos salvos: {epoca_melhor}')
if USAR_EARLY_STOPPING:
    ax[0].axvspan(epoca_melhor, epoca_final, color='tab:red', alpha=0.12,
                  label=f'Espera após melhor época (paciência={PACIENCIA_EARLY})')
    ax[0].axvline(epoca_final, color='tab:red', linestyle=':',
                  label=f'Última época executada: {epoca_final}')
ax[0].set(ylabel='MSE (alvo padronizado)', title='Convergência e early stopping')
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.2)
ax[1].step(hist.epoca, hist.lr, where='post', label='Learning rate')
ax[1].set(xlabel='Época', ylabel='Learning rate')
ax[1].grid(alpha=0.2)
ax[1].legend()
plt.tight_layout()
plt.show()
print(f'Melhor época = {epoca_melhor}; última época executada = {epoca_final}.')
print('O modelo final usa os pesos da MELHOR época, não os da última.')

# %% 10 | NSE E PBIAS EM FUNÇÃO DO HORIZONTE (TESTE)
fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
for nome, dados_modelo in resumo.groupby('modelo'):
    dados_modelo = dados_modelo.sort_values('horizonte_h')
    ax[0].plot(dados_modelo.horizonte_h, dados_modelo.NSE,
               marker='o', label=nome)
    ax[1].plot(dados_modelo.horizonte_h, dados_modelo.PBIAS,
               marker='o', label=nome)
ax[0].set(ylabel='NSE', title='Desempenho no teste por horizonte')
ax[1].axhline(0, color='0.4', linewidth=1, linestyle='--')
ax[1].set(xlabel='Horizonte de previsão (h)', ylabel='PBIAS (%)')
for a in ax:
    a.grid(alpha=0.2)
    a.legend()
plt.tight_layout()
plt.show()
print('PBIAS positivo: subestimativa; PBIAS negativo: superestimativa.')

# %% 11 | PREVISÕES DO TESTE PARA TODOS OS HORIZONTES
# Usa a série completa de teste, salvo se HORAS_PLOT_TESTE for definido.
for modelo in MODELOS:
    horizontes_disponiveis = sorted(h for m, h in previsoes_teste if m == modelo)
    fig, ax = plt.subplots(len(horizontes_disponiveis), 1,
                           figsize=(13, 3.0*len(horizontes_disponiveis)),
                           sharex=True, squeeze=False)
    for eixo, h in zip(ax[:, 0], horizontes_disponiveis):
        tabela = previsoes_teste[(modelo, h)]
        if HORAS_PLOT_TESTE is not None:
            tabela = tabela.iloc[:HORAS_PLOT_TESTE]
        eixo.plot(tabela.index, tabela.observado, color='black',
                  linewidth=1.0, label='Observado')
        eixo.plot(tabela.index, tabela.previsto, linewidth=0.9,
                  label=f'Previsto +{h} h')
        eixo.set(ylabel='Vazão', title=f'{modelo} | horizonte +{h} h')
        eixo.grid(alpha=0.2)
        eixo.legend(loc='upper right')
    ax[-1, 0].set_xlabel('Data da vazão prevista (data-alvo)')
    plt.tight_layout()
    plt.show()

# %% 12 | ZOOM DIDÁTICO: MAIOR CHEIA DO TESTE
for h in sorted(h for m, h in previsoes_teste if m == MODELO_PLOT):
    tabela = previsoes_teste[(MODELO_PLOT, h)]
    pico = tabela.observado.idxmax()
    evento = tabela.loc[pico - pd.Timedelta(hours=72):
                        pico + pd.Timedelta(hours=72)]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(evento.index, evento.observado, label='Observado', lw=2)
    ax.plot(evento.index, evento.previsto, label=f'Previsto +{h} h')
    ax.set(title=f'Maior cheia no teste | +{h} h', ylabel='Vazão', xlabel='Data')
    ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()
