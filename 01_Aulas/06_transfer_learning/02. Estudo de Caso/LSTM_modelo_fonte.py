# %% 00 | HYDRO-UAI - LSTM REGIONAL (MODELO-FONTE MULTIBACIAS)
"""Uma unica LSTM treinada com varias bacias-fonte.

Formato esperado dos arquivos CSV (formato largo):
  diarios: data, 60220000_precipitation, 60220000_temperature,
           60220000_streamflow, 60654000_precipitation, ...
  estaticos: ID_bacia, AD (area em km2)

O modelo recebe precipitacao e temperatura; a vazao e o alvo em mm/dia.
Divisao cronologica por bacia; escalonadores ajustados somente com treino.
Uma unica rede e treinada com todas as sequencias das bacias-fonte.
O teste e avaliado individualmente por bacia, sem selecionar hiperparametros pelo teste.
"""

# %% 01 | IMPORTACOES E CONFIGURACOES
from pathlib import Path
import copy
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
ARQUIVO_DIARIO = BASE_DIR / '01. Dados' / '02. Fonte' / '02. Tratado' / 'Atributosdiarios_Fonte2.csv'
ARQUIVO_ESTATICO = BASE_DIR / '01. Dados' / '02. Fonte' / '02. Tratado' / 'Atributosestaticos_Fonte2.csv'
SAIDA = BASE_DIR / '02. Resultados' / '01.Fonte' / 'LSTM_regional'

# Ajuste conforme as bacias efetivamente presentes nos arquivos CSV.
BACIAS_FONTE = [
    '86480000', '60265000', '60850000', '86160000', '86100000', '60250000'
]

# Para comparar com o codigo original: entradas somente chuva e temperatura.
# Para usar API, altere para ['precipitation_mm', 'temperature', 'API_15'].
FEATURES = ['precipitation_mm', 'temperature']
JANELA_D = 180
HORIZONTE_D = 0
CAMADAS = 1
UNIDADES = 32
BATCH_SIZE = 128
EPOCAS_MAX = 100
LR_INICIAL = 0.005
WEIGHT_DECAY = 0.0
PACIENCIA_LR = 5
FATOR_LR = 0.2
LR_MIN = 1e-5
PACIENCIA_EARLY = 20
MELHORIA_MINIMA = 1e-6
PCT_TREINO = 70
PCT_VALIDACAO = 15
PCT_TESTE = 15
SEMENTE = 42
API_DIAS = 15
API_K = 0.9
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if PCT_TREINO + PCT_VALIDACAO + PCT_TESTE != 100:
    raise ValueError('Percentuais devem somar 100.')
if not BACIAS_FONTE or len(set(BACIAS_FONTE)) != len(BACIAS_FONTE):
    raise ValueError('A lista BACIAS_FONTE deve conter IDs unicos.')
SAIDA.mkdir(parents=True, exist_ok=True)
print('Dispositivo:', DEVICE)


def fixar_semente():
    random.seed(SEMENTE)
    np.random.seed(SEMENTE)
    torch.manual_seed(SEMENTE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEMENTE)


def ler_csv(caminho):
    if not caminho.exists():
        raise FileNotFoundError(f'Arquivo nao encontrado: {caminho}')
    df = pd.read_csv(caminho, sep=None, engine='python', encoding='utf-8-sig')
    df.columns = df.columns.astype(str).str.replace('\ufeff', '', regex=False).str.strip()
    return df


# %% 02 | CARREGAMENTO DE TODAS AS BACIAS

def carregar_bacias():
    diario = ler_csv(ARQUIVO_DIARIO)
    estatico = ler_csv(ARQUIVO_ESTATICO)
    if 'ID_bacia' not in estatico or 'AD' not in estatico:
        raise ValueError('Arquivo estatico precisa ter ID_bacia e AD.')
    col_data = 'data' if 'data' in diario else 'date'
    if col_data not in diario:
        raise ValueError('Arquivo diario precisa ter coluna data ou date.')
    datas = pd.to_datetime(diario[col_data], errors='coerce')
    if datas.isna().all():
        raise ValueError('Nao foi possivel interpretar as datas do arquivo diario.')
    estatico['ID_bacia'] = estatico['ID_bacia'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    estatico['AD'] = pd.to_numeric(estatico['AD'].astype(str).str.replace(',', '.', regex=False), errors='coerce')
    series = {}
    problemas = []
    for bacia in BACIAS_FONTE:
        nomes = {
            'precipitation_mm': f'{bacia}_precipitation',
            'temperature': f'{bacia}_temperature',
            'q_m3_s': f'{bacia}_streamflow',
        }
        faltantes = [c for c in nomes.values() if c not in diario.columns]
        if faltantes:
            problemas.append(f'{bacia}: colunas ausentes: {faltantes}')
            continue
        areas = estatico.loc[estatico['ID_bacia'] == bacia, 'AD'].dropna()
        areas = areas[areas > 0]
        if areas.empty:
            problemas.append(f'{bacia}: area AD ausente ou invalida')
            continue
        area = float(areas.iloc[0])
        df = pd.DataFrame({'date': datas})
        for destino, origem in nomes.items():
            df[destino] = pd.to_numeric(
                diario[origem].astype(str).str.replace(',', '.', regex=False),
                errors='coerce')
        df['q_mm_d'] = df['q_m3_s'] * 86.4 / area
        df = df.dropna(subset=['date']).sort_values('date').drop_duplicates('date', keep='last')
        df = df.set_index('date').asfreq('D')
        if 'API_15' in FEATURES:
            pesos = API_K ** np.arange(API_DIAS)
            df['API_15'] = df['precipitation_mm'].rolling(API_DIAS, min_periods=API_DIAS).apply(
                lambda x: np.dot(x[::-1], pesos), raw=True)
        df.attrs['area_km2'] = area
        series[bacia] = df
        print(f'{bacia}: {len(df)} dias | AD={area:.1f} km2')
    if problemas:
        print('\nBacias ignoradas:')
        for problema in problemas:
            print(' -', problema)
    if not series:
        raise ValueError('Nenhuma bacia carregada. Verifique IDs e colunas.')
    return series


series_bacias = carregar_bacias()


# %% 03 | SEQUENCIAS E DIVISAO CRONOLOGICA POR BACIA

def construir_conjuntos(series):
    partes = {p: {'X': [], 'y': [], 'bacia': [], 'data': []}
              for p in ('treino', 'validacao', 'teste')}
    periodos = []
    for bacia, df in series.items():
        n = len(df)
        i_tr = int(n * PCT_TREINO / 100)
        i_va = int(n * (PCT_TREINO + PCT_VALIDACAO) / 100)
        if i_tr < JANELA_D or i_va <= i_tr or i_va >= n:
            print(f'{bacia}: serie insuficiente para a divisao; ignorada')
            continue
        vals = df[FEATURES].to_numpy(dtype=np.float32)
        alvo = df['q_mm_d'].to_numpy(dtype=np.float32)
        contagem = {'treino': 0, 'validacao': 0, 'teste': 0}
        for i in range(JANELA_D - 1, n - HORIZONTE_D):
            pos_alvo = i + HORIZONTE_D
            x = vals[i-JANELA_D+1:i+1]
            y = alvo[pos_alvo]
            if not np.isfinite(x).all() or not np.isfinite(y):
                continue
            part = ('treino' if pos_alvo < i_tr else
                    'validacao' if pos_alvo < i_va else 'teste')
            partes[part]['X'].append(x)
            partes[part]['y'].append(y)
            partes[part]['bacia'].append(bacia)
            partes[part]['data'].append(df.index[pos_alvo])
            contagem[part] += 1
        periodos.append({'ID_bacia': bacia, 'fim_treino': str(df.index[i_tr-1].date()),
                         'fim_validacao': str(df.index[i_va-1].date()), **contagem})
    for part, valores in partes.items():
        if not valores['X']:
            raise ValueError(f'Nenhuma sequencia valida em {part}.')
        valores['X'] = np.stack(valores['X']).astype(np.float32)
        valores['y'] = np.asarray(valores['y'], dtype=np.float32)
        valores['bacia'] = np.asarray(valores['bacia'])
        valores['data'] = pd.DatetimeIndex(valores['data'])
        print(f'{part}: {len(valores["y"])} sequencias, '
              f'{len(set(valores["bacia"]))} bacias')
    return partes, pd.DataFrame(periodos)


partes, tabela_periodos = construir_conjuntos(series_bacias)
tabela_periodos.to_csv(SAIDA / 'periodos_e_amostras.csv', sep=';', index=False)


# %% 04 | NORMALIZACAO GLOBAL: SOMENTE TREINO
# O mesmo scaler e usado para TODAS as bacias (importante para modelo-fonte).
n_features = len(FEATURES)
sx = StandardScaler().fit(partes['treino']['X'].reshape(-1, n_features))
sy = StandardScaler().fit(partes['treino']['y'].reshape(-1, 1))
for part in partes.values():
    part['Xn'] = sx.transform(part['X'].reshape(-1, n_features)).reshape(part['X'].shape).astype(np.float32)
    part['yn'] = sy.transform(part['y'].reshape(-1, 1)).ravel().astype(np.float32)


# %% 05 | ARQUITETURA REGIONAL: PESOS COMPARTILHADOS
class LSTMRegional(nn.Module):
    def __init__(self, n_features, unidades, camadas):
        super().__init__()
        self.lstm = nn.LSTM(n_features, unidades, num_layers=camadas, batch_first=True)
        self.saida = nn.Linear(unidades, 1)

    def forward(self, x):
        sequencia, _ = self.lstm(x)
        return self.saida(sequencia[:, -1, :])


# %% 06 | TREINAMENTO CONJUNTO COM VALIDACAO REGIONAL
fixar_semente()
modelo = LSTMRegional(n_features, UNIDADES, CAMADAS).to(DEVICE)
tr = partes['treino']
va = partes['validacao']
loader = DataLoader(TensorDataset(torch.from_numpy(tr['Xn']),
                                  torch.from_numpy(tr['yn'][:, None])),
                    batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.from_numpy(va['Xn']),
                                      torch.from_numpy(va['yn'][:, None])),
                        batch_size=max(BATCH_SIZE, 512), shuffle=False)
criterio = nn.MSELoss()
opt = torch.optim.Adam(modelo.parameters(), lr=LR_INICIAL, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode='min', patience=PACIENCIA_LR, factor=FATOR_LR, min_lr=LR_MIN)
melhor_val = float('inf')
melhor_estado = None
melhor_epoca = 0
espera = 0
historico = []

for epoca in range(1, EPOCAS_MAX + 1):
    modelo.train()
    soma_tr, n_tr = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        loss = criterio(modelo(xb), yb)
        loss.backward()
        opt.step()
        soma_tr += loss.item() * len(xb)
        n_tr += len(xb)
    modelo.eval()
    soma_va, n_va = 0.0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = criterio(modelo(xb), yb)
            soma_va += loss.item() * len(xb)
            n_va += len(xb)
    perda_tr, perda_va = soma_tr / n_tr, soma_va / n_va
    lr_atual = opt.param_groups[0]['lr']
    historico.append({'epoca': epoca, 'MSE_treino': perda_tr,
                      'MSE_validacao': perda_va, 'learning_rate': lr_atual})
    if perda_va < melhor_val - MELHORIA_MINIMA:
        melhor_val = perda_va
        melhor_estado = copy.deepcopy(modelo.state_dict())
        melhor_epoca = epoca
        espera = 0
    else:
        espera += 1
    scheduler.step(perda_va)
    if epoca == 1 or epoca % 10 == 0:
        print(f'Epoca {epoca:3d} | treino={perda_tr:.5f} | '
              f'validacao={perda_va:.5f} | LR={lr_atual:.6g}')
    if espera >= PACIENCIA_EARLY:
        print(f'Early stopping na epoca {epoca}')
        break

if melhor_estado is None:
    raise RuntimeError('Nao foi possivel selecionar um modelo.')
modelo.load_state_dict(melhor_estado)
modelo.eval()
pd.DataFrame(historico).to_csv(SAIDA / 'historico_treinamento.csv', sep=';', index=False)
print('Melhor epoca:', melhor_epoca)


# %% 07 | AVALIACAO INDIVIDUAL POR BACIA NO TESTE

def prever(x_normalizado):
    saidas = []
    modelo.eval()
    with torch.no_grad():
        for i in range(0, len(x_normalizado), 1024):
            xb = torch.from_numpy(x_normalizado[i:i+1024]).to(DEVICE)
            saidas.append(modelo(xb).cpu().numpy().ravel())
    pred_norm = np.concatenate(saidas)
    return np.maximum(sy.inverse_transform(pred_norm[:, None]).ravel(), 0.0)


def metricas(obs, pred):
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    den = np.sum((obs - obs.mean()) ** 2)
    nse = 1 - np.sum((pred - obs) ** 2) / den if den > 0 else np.nan
    r = np.corrcoef(obs, pred)[0, 1] if np.std(obs) > 0 and np.std(pred) > 0 else np.nan
    alpha = np.std(pred) / np.std(obs) if np.std(obs) > 0 else np.nan
    beta = np.mean(pred) / np.mean(obs) if np.mean(obs) != 0 else np.nan
    kge = 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)
    pbias = 100 * np.sum(obs - pred) / np.sum(obs) if np.sum(obs) != 0 else np.nan
    return {'NSE': nse, 'KGE': kge,
            'RMSE': np.sqrt(np.mean((pred-obs)**2)), 'PBIAS': pbias}


teste = partes['teste']
pred = prever(teste['Xn'])
previsoes = pd.DataFrame({
    'ID_bacia': teste['bacia'], 'data': teste['data'],
    'Q_obs_mm_d': teste['y'], 'Q_prev_mm_d': pred
})
previsoes.to_csv(SAIDA / 'previsoes_teste_todas_bacias.csv', sep=';', index=False)
resultados = []
for bacia, sub in previsoes.groupby('ID_bacia', sort=True):
    resultados.append({'ID_bacia': bacia, 'N_dias': len(sub),
                       **metricas(sub['Q_obs_mm_d'], sub['Q_prev_mm_d'])})
    sub.to_csv(SAIDA / f'previsoes_teste_{bacia}.csv', sep=';', index=False)
resumo = pd.DataFrame(resultados).sort_values('ID_bacia')
resumo.to_csv(SAIDA / 'metricas_teste_por_bacia.csv', sep=';', index=False)
print('\nMETRICAS DE TESTE POR BACIA (mm/dia):')
print(resumo.to_string(index=False, float_format=lambda v: f'{v:.4f}'))


# %% 08 | SALVAR MODELO-FONTE PARA FUTURO TRANSFER LEARNING
pacote = {
    'tipo': 'LSTM_regional_fonte',
    'state_dict': {k: v.detach().cpu() for k, v in modelo.state_dict().items()},
    'config': {'modelo': 'LSTM', 'janela_d': JANELA_D,
               'horizonte_d': HORIZONTE_D, 'camadas': CAMADAS,
               'unidades': UNIDADES, 'n_features': n_features},
    'features': FEATURES,
    'bacias_fonte': sorted(series_bacias.keys()),
    'sx_mean': sx.mean_.tolist(),
    'sx_scale': sx.scale_.tolist(),
    'sy_mean': sy.mean_.tolist(),
    'sy_scale': sy.scale_.tolist(),
    'alvo': 'q_mm_d',
    'unidade_alvo': 'mm/dia',
    'api_dias': API_DIAS,
    'api_k': API_K,
    'melhor_epoca': melhor_epoca,
    'semente': SEMENTE,
}
caminho_modelo = SAIDA / 'LSTM_MODELO_FONTE.pt'
torch.save(pacote, caminho_modelo)
print('\nModelo-fonte salvo em:', caminho_modelo)

# %% 09 | GRAFICOS
hist = pd.DataFrame(historico)
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(hist['epoca'], hist['MSE_treino'], label='Treino')
axes[0].plot(hist['epoca'], hist['MSE_validacao'], label='Validacao')
axes[0].axvline(melhor_epoca, color='gray', ls='--', label='Melhor epoca')
axes[0].set_ylabel('MSE normalizado')
axes[0].legend()
axes[0].grid(alpha=.2)
axes[1].plot(hist['epoca'], hist['learning_rate'])
axes[1].set(xlabel='Epoca', ylabel='Learning rate')
axes[1].grid(alpha=.2)
plt.tight_layout()
plt.savefig(SAIDA / 'convergencia_regional.png', dpi=160)
plt.show()

fig, ax = plt.subplots(figsize=(10, max(4, len(resumo)*.38)))
ax.barh(resumo['ID_bacia'], resumo['NSE'])
ax.axvline(0, color='gray', lw=1)
ax.set(xlabel='NSE no teste', ylabel='Bacia-fonte',
       title='Desempenho da LSTM regional por bacia')
ax.grid(axis='x', alpha=.2)
plt.tight_layout()
plt.savefig(SAIDA / 'NSE_por_bacia.png', dpi=160)
plt.show()
