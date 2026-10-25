# %% 00 | HYDRO-UAI | TRANSFERENCIA LSTM: ZERO-SHOT E FINE-TUNING
"""Reutiliza pesos e escalonadores da LSTM regional multibacias.

IMPORTANTE SOBRE O PERIODO DE TESTE:
- O modelo-fonte fornecido salva periodos_e_amostras.csv (por bacia),
  MAS nao salva os periodos dentro do .pt.
- O modelo-alvo individual salva 'fim_validacao' no .pt, mas nao o fim do teste.
- Este programa descobre o teste do alvo a partir de fim_validacao e do ultimo
  dia do CSV, e confronta o intervalo com os testes das bacias-fonte.
- Por padrao, exige que TODAS as bacias-fonte tenham EXATAMENTE o mesmo teste.
  Se nao tiverem, interrompe e pede uma decisao metodologica, sem inferir datas.
- O teste nao participa do fine-tuning, early stopping ou normalizacao.
- A vazao observada do alvo e usada somente no ajuste FT e na avaliacao;
  zero-shot usa chuva/temperatura do alvo como entradas e vazao so para metricas.
"""

# %% 01 | IMPORTACOES E CONFIGURACOES
from pathlib import Path
from datetime import timedelta

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import copy
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

BASE_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
CSV_DIARIO_ALVO = BASE_DIR / '01. Dados' / '01. Alvo' / "02. Tratado" / 'Atributosdiarios_Alvo2.csv'
CSV_ESTATICO_ALVO = BASE_DIR / '01. Dados' / '01. Alvo' / "02. Tratado" / 'Atributosestaticos_Alvo2.csv'
ARQUIVO_MODELO_FONTE = (BASE_DIR / '02. Resultados' / '01.Fonte' /
                        'LSTM_regional' / 'LSTM_MODELO_FONTE.pt')
ARQUIVO_PERIODOS_FONTE = (BASE_DIR / '02. Resultados' / '01.Fonte' /
                          'LSTM_regional' / 'periodos_e_amostras.csv')
ARQUIVO_MODELO_ALVO = (BASE_DIR / '02. Resultados' / '02.Alvo' /
                       'resultados_diarios_LSTM' / 'modelos_completos' / 'LSTM_d0.pt')
SAIDA = BASE_DIR / '02. Resultados' / '03.Transferencia2'

BACIA_ALVO = '60220000'
EXECUTAR_ZERO_SHOT = True
EXECUTAR_FINE_TUNING = True
ANOS_FINE_TUNING = [1, 2, 5, 10, 'todos']
# 'recentes': ultimos anos antes do teste; 'iniciais': primeiros anos disponiveis
SELECAO_ANOS = 'recentes'
# Janela da disponibilidade do alvo para ajuste (None = usar serie anterior ao teste)
DATA_INICIO_DISPONIVEL = None   # ex.: '1990-01-01'
DATA_FIM_DISPONIVEL = None      # ex.: '2015-12-31'
# Datas opcionais SOMENTE para resolver divergencia, exigem validacao explicita
DATA_INICIO_TESTE_MANUAL = None
DATA_FIM_TESTE_MANUAL = None
# Exigir teste igual para TODAS as bacias fonte, e igual ao teste do modelo alvo
EXIGIR_TESTE_FONTE_IGUAL_ALVO = True
# Caso periodos fonte divirjam, nao permite override silencioso
AUTORIZAR_TESTE_MANUAL_DIVERGENTE = False

MODO_FINE_TUNING = 'FC_LSTM'  # 'FC', 'FC_LSTM', 'TODAS'
PCT_VALIDACAO_FT = 20
EPOCAS_FT = 100
BATCH_SIZE = 64
LR_FT = 1e-4
WEIGHT_DECAY = 1e-4
PACIENCIA_LR = 8
FATOR_LR = 0.5
LR_MIN = 1e-6
PACIENCIA_EARLY = 15
SEMENTE = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SAIDA.mkdir(parents=True, exist_ok=True)
if not EXECUTAR_ZERO_SHOT and not EXECUTAR_FINE_TUNING:
    raise ValueError('Habilite pelo menos um experimento.')


def ler_csv(caminho):
    if not caminho.exists():
        raise FileNotFoundError(f'Arquivo nao encontrado: {caminho}')
    df = pd.read_csv(caminho, sep=None, engine='python', encoding='utf-8-sig')
    df.columns = df.columns.astype(str).str.replace('\ufeff', '', regex=False).str.strip()
    return df


def numerico(s):
    return pd.to_numeric(s.astype(str).str.replace(',', '.', regex=False), errors='coerce')


def seed():
    random.seed(SEMENTE)
    np.random.seed(SEMENTE)
    torch.manual_seed(SEMENTE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEMENTE)


# %% 02 | CARREGAR MODELO-FONTE, ARQUITETURA E NORMALIZACAO
if not ARQUIVO_MODELO_FONTE.exists():
    raise FileNotFoundError(f'Treine e salve primeiro o modelo-fonte: {ARQUIVO_MODELO_FONTE}')
pacote = torch.load(ARQUIVO_MODELO_FONTE, map_location='cpu', weights_only=True)
cfg = pacote['config']
FEATURES = list(pacote['features'])
JANELA = int(cfg['janela_d'])
HORIZONTE = int(cfg['horizonte_d'])
SX_MEAN = np.asarray(pacote['sx_mean'], dtype=np.float32)
SX_SCALE = np.asarray(pacote['sx_scale'], dtype=np.float32)
SY_MEAN = float(pacote['sy_mean'][0])
SY_SCALE = float(pacote['sy_scale'][0])
if HORIZONTE != 0:
    raise ValueError('Este script foi preparado para horizonte zero. Ajuste antes de usar outro.')
if len(FEATURES) != len(SX_MEAN) or len(FEATURES) != len(SX_SCALE):
    raise ValueError('Inconsistencia entre FEATURES e normalizacao do modelo-fonte.')
if not np.isfinite(SX_SCALE).all() or (SX_SCALE <= 0).any() or SY_SCALE <= 0:
    raise ValueError('Escalonadores invalidos no arquivo fonte.')
if BACIA_ALVO in set(map(str, pacote.get('bacias_fonte', []))):
    raise ValueError(f'Bacia-alvo {BACIA_ALVO} consta no treino fonte! Remova-a das fontes e retreine para evitar vazamento.')


class LSTMRegional(nn.Module):
    def __init__(self, n_features, unidades, camadas):
        super().__init__()
        self.lstm = nn.LSTM(n_features, unidades, num_layers=camadas, batch_first=True)
        self.saida = nn.Linear(unidades, 1)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.saida(h[:, -1, :])


def modelo_da_fonte():
    rede = LSTMRegional(len(FEATURES), int(cfg['unidades']), int(cfg['camadas']))
    rede.load_state_dict(pacote['state_dict'], strict=True)
    return rede.to(DEVICE)


# %% 03 | CARREGAR BACIA-ALVO
origem = ler_csv(CSV_DIARIO_ALVO)
estaticos = ler_csv(CSV_ESTATICO_ALVO)
col_data = 'data' if 'data' in origem else 'date'
if col_data not in origem:
    raise ValueError('CSV diario precisa conter data ou date.')
colunas = {
    'precipitation_mm': f'{BACIA_ALVO}_precipitation',
    'temperature': f'{BACIA_ALVO}_temperature',
    'q_m3_s': f'{BACIA_ALVO}_streamflow'
}
if not set(colunas.values()).issubset(origem.columns):
    raise ValueError(f'Colunas ausentes: {sorted(set(colunas.values()) - set(origem.columns))}')
estaticos['ID_bacia'] = estaticos['ID_bacia'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
areas = numerico(estaticos.loc[estaticos['ID_bacia'] == BACIA_ALVO, 'AD']).dropna()
areas = areas[areas > 0]
if areas.empty:
    raise ValueError(f'Area AD nao encontrada para {BACIA_ALVO}.')
AD = float(areas.iloc[0])
df = pd.DataFrame({'date': pd.to_datetime(origem[col_data], errors='coerce')})
for destino, coluna in colunas.items():
    df[destino] = numerico(origem[coluna])
df = df.dropna(subset=['date']).sort_values('date').drop_duplicates('date', keep='last')
df = df.set_index('date').asfreq('D')
for col in colunas:
    df.loc[df[col].isin([-999, -9999, -99999]), col] = np.nan
df.loc[df['precipitation_mm'] < 0, 'precipitation_mm'] = np.nan
df.loc[df['q_m3_s'] < 0, 'q_m3_s'] = np.nan
df['q_mm_d'] = df['q_m3_s'] * 86.4 / AD
if 'API_15' in FEATURES:
    dias_api = int(pacote.get('api_dias', 15))
    k_api = float(pacote.get('api_k', .9))
    pesos = k_api ** np.arange(dias_api)
    df['API_15'] = df['precipitation_mm'].rolling(dias_api, min_periods=dias_api).apply(
        lambda x: np.dot(x[::-1], pesos), raw=True)
for feature in FEATURES:
    if feature not in df:
        raise ValueError(f'Entrada exigida pelo modelo-fonte nao encontrada: {feature}')


# %% 04 | DESCOBRIR E CONFERIR PERIODO DE TESTE
# Fonte: CSV salvo pelo codigo de treinamento multibacias original.
periodos_fonte = ler_csv(ARQUIVO_PERIODOS_FONTE)
necessarias = {'ID_bacia', 'fim_validacao'}
if not necessarias.issubset(periodos_fonte.columns):
    raise ValueError('periodos_e_amostras.csv nao possui ID_bacia e fim_validacao.')
periodos_fonte['ID_bacia'] = periodos_fonte['ID_bacia'].astype(str).str.replace(r'\.0$', '', regex=True)
fontes_treinadas = set(map(str, pacote.get('bacias_fonte', [])))
if fontes_treinadas:
    periodos_fonte = periodos_fonte[periodos_fonte['ID_bacia'].isin(fontes_treinadas)].copy()
    faltantes = fontes_treinadas - set(periodos_fonte['ID_bacia'])
    if faltantes:
        raise ValueError(f'Periodos ausentes para fontes treinadas: {sorted(faltantes)}')
periodos_fonte['inicio_teste'] = pd.to_datetime(periodos_fonte['fim_validacao']) + pd.Timedelta(days=1)
# Fim de teste das fontes NAO esta no CSV de periodos do modelo original.
# O fim e recuperado a partir da serie diaria comum, mas isso nao prova que
# todas as fontes tenham o mesmo ultimo dia VALIDO de teste.
# Exigir fim manual explicito ou conferir validade por bacia no CSV diario.
ultimo_dia_fonte = pd.to_datetime(origem[col_data], errors='coerce').max()
if pd.isna(ultimo_dia_fonte):
    raise ValueError('Sem datas validas no CSV diario.')
periodos_fonte['fim_teste_nominal'] = ultimo_dia_fonte.normalize()

if not ARQUIVO_MODELO_ALVO.exists():
    raise FileNotFoundError(
        'Modelo-alvo individual .pt nao encontrado. Para descobrir o teste original '
        'sem inventar datas, informe o checkpoint do modelo-alvo em ARQUIVO_MODELO_ALVO.')
pacote_alvo = torch.load(ARQUIVO_MODELO_ALVO, map_location='cpu', weights_only=True)
if 'fim_validacao' not in pacote_alvo:
    raise ValueError('Modelo-alvo nao contem fim_validacao. Nao e seguro inferir seu teste.')
inicio_alvo = pd.Timestamp(pacote_alvo['fim_validacao']).normalize() + pd.Timedelta(days=1)
fim_alvo = df.index.max().normalize()

inicios_fonte = sorted(periodos_fonte['inicio_teste'].dt.normalize().unique())
mesmo_inicio = len(inicios_fonte) == 1 and pd.Timestamp(inicios_fonte[0]) == inicio_alvo
mesmo_fim_nominal = (ultimo_dia_fonte.normalize() == fim_alvo)

print('\nPERIODO DE TESTE DO MODELO-ALVO:')
print(f'  {inicio_alvo.date()} a {fim_alvo.date()}')
print('INICIOS DE TESTE POR BACIA-FONTE:')
print(periodos_fonte[['ID_bacia', 'inicio_teste', 'fim_teste_nominal']].to_string(index=False))

print("INICIANDO PROXIMA ETAPA", flush=True)
print("TESTANDO VERIFICACAO DAS DATAS", flush=True)

if DATA_INICIO_TESTE_MANUAL is not None or DATA_FIM_TESTE_MANUAL is not None:
    if DATA_INICIO_TESTE_MANUAL is None or DATA_FIM_TESTE_MANUAL is None:
        raise ValueError('Preencha ambas as datas manuais do teste.')
    inicio_teste = pd.Timestamp(DATA_INICIO_TESTE_MANUAL)
    fim_teste = pd.Timestamp(DATA_FIM_TESTE_MANUAL)
    if (inicio_teste != inicio_alvo or fim_teste != fim_alvo or
            not mesmo_inicio or not mesmo_fim_nominal):
        if not AUTORIZAR_TESTE_MANUAL_DIVERGENTE:
            raise ValueError('Teste manual diverge dos registros; altere AUTORIZAR_TESTE_MANUAL_DIVERGENTE=True somente apos verificar o desenho experimental.')
else:
    inicio_teste, fim_teste = inicio_alvo, fim_alvo
    if EXIGIR_TESTE_FONTE_IGUAL_ALVO and (not mesmo_inicio or not mesmo_fim_nominal):
        raise ValueError(
            'TESTES NAO COINCIDEM: fonte e alvo possuem limites diferentes. '
            'Consulte periodos_e_amostras.csv e ajuste o treino fonte/alvo '
            'para datas de teste fixas; nao use o teste como dados de ajuste.')
if inicio_teste > fim_teste:
    raise ValueError('Inicio do teste posterior ao fim.')

# Para impedir sobreposicao entre as fontes e o teste do alvo, e necessario
# verificar o recorte real de treinamento da fonte, nao apenas o teste dela.
# Se os testes comecam juntos, a divisao cronologica fonte exclui esse teste.
if EXIGIR_TESTE_FONTE_IGUAL_ALVO and not mesmo_inicio:
    raise ValueError('Inicio do teste fonte e alvo divergem.')


# %% 05 | CONSTRUIR SEQUENCIAS DA BACIA-ALVO UMA UNICA VEZ
valores = df[FEATURES].to_numpy(dtype=np.float32)
alvos = df['q_mm_d'].to_numpy(dtype=np.float32)
registros = []
X_lista, y_lista, datas_lista = [], [], []
for i in range(JANELA - 1, len(df) - HORIZONTE):
    x = valores[i-JANELA+1:i+1]
    y = alvos[i+HORIZONTE]
    if np.isfinite(x).all() and np.isfinite(y):
        X_lista.append((x - SX_MEAN) / SX_SCALE)
        y_lista.append(float(y))
        datas_lista.append(df.index[i+HORIZONTE])
if not X_lista:
    raise ValueError('Nenhuma sequencia valida para a bacia-alvo.')
X = np.asarray(X_lista, dtype=np.float32)
y = np.asarray(y_lista, dtype=np.float32)
datas = pd.DatetimeIndex(datas_lista)
mascara_teste = np.asarray((datas >= inicio_teste) & (datas <= fim_teste))
if not mascara_teste.any():
    raise ValueError('Teste sem sequencias validas no alvo.')


def prever(rede, x):
    rede.eval()
    partes = []
    with torch.no_grad():
        for i in range(0, len(x), 1024):
            xb = torch.from_numpy(x[i:i+1024]).to(DEVICE)
            partes.append(rede(xb).cpu().numpy().ravel())
    return np.maximum(np.concatenate(partes) * SY_SCALE + SY_MEAN, 0.0)


def metricas(obs, pred):
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    den = np.sum((obs - obs.mean())**2)
    nse = 1 - np.sum((pred - obs)**2) / den if den > 0 else np.nan
    r = correlacao(obs, pred)
    alpha = np.std(pred)/np.std(obs) if np.std(obs) > 0 else np.nan
    beta = np.mean(pred)/np.mean(obs) if np.mean(obs) else np.nan
    kge = 1-np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)
    pbias = 100*np.sum(obs-pred)/np.sum(obs) if np.sum(obs) else np.nan
    return {'NSE': nse, 'KGE': kge, 'RMSE': np.sqrt(np.mean((pred-obs)**2)),
            'PBIAS': pbias, 'N_dias': len(obs)}

def correlacao(y_obs, y_sim):
    y_obs = np.asarray(y_obs, dtype=float)
    y_sim = np.asarray(y_sim, dtype=float)

    a = y_obs - np.mean(y_obs)
    b = y_sim - np.mean(y_sim)

    denominador = np.sqrt(np.sum(a**2) * np.sum(b**2))

    if denominador == 0:
        return np.nan

    return float(np.sum(a * b) / denominador)


# %% 06 | ZERO-SHOT: SEM ATUALIZAR PESOS E SEM USAR VAZAO PARA TREINO
resultados = []

def avaliar(rede, nome):
    obs = y[mascara_teste]
    pred = prever(rede, X[mascara_teste])
    tabela = pd.DataFrame({'data': datas[mascara_teste],
                           'Q_obs_mm_d': obs, 'Q_prev_mm_d': pred})
    tabela.to_csv(SAIDA / f'previsoes_{nome}.csv', sep=';', index=False)
    met = metricas(obs, pred)
    resultados.append({'experimento': nome, 'bacia_alvo': BACIA_ALVO,
                       'inicio_teste': str(inicio_teste.date()),
                       'fim_teste': str(fim_teste.date()), **met})
    print(nome, met)
    
    # Hidrograma interativo com Plotly
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=tabela['data'],
        y=obs,
        mode='lines',
        name='Observado',
        line=dict(color='black', width=1)
    ))

    fig.add_trace(go.Scatter(
        x=tabela['data'],
        y=pred,
        mode='lines',
        name=f'Simulado | NSE = {met["NSE"]:.3f}',
        line=dict(color='royalblue', width=1)
    ))

    fig.update_layout(
        title=f'Bacia {BACIA_ALVO} - {nome}',
        xaxis_title='Data',
        yaxis_title='Vazão (mm/dia)',
        template='plotly_white',
        hovermode='x unified'
    )

    fig.write_html(SAIDA / f'hidrograma_{nome}.html')

if EXECUTAR_ZERO_SHOT:
    avaliar(modelo_da_fonte(), 'zero_shot')


# %% 07 | FINE-TUNING COM N ANOS E TESTE FIXO
# Os anos sao selecionados por CALENDARIO, nao como 365*N amostras validas.
limite_inicio = pd.Timestamp(DATA_INICIO_DISPONIVEL) if DATA_INICIO_DISPONIVEL else df.index.min()
limite_fim = pd.Timestamp(DATA_FIM_DISPONIVEL) if DATA_FIM_DISPONIVEL else inicio_teste - pd.Timedelta(days=1)
limite_fim = min(limite_fim, inicio_teste - pd.Timedelta(days=1))
if limite_inicio >= inicio_teste or limite_fim < limite_inicio:
    raise ValueError('Periodo disponivel para FT invalido.')


def periodo_ft(anos):
    if anos == 'todos':
        return limite_inicio, limite_fim
    n = int(anos)
    if n <= 0:
        raise ValueError('Anos FT devem ser positivos.')
    if SELECAO_ANOS == 'recentes':
        fim = limite_fim
        inicio = max(limite_inicio, fim - pd.DateOffset(years=n) + pd.Timedelta(days=1))
    elif SELECAO_ANOS == 'iniciais':
        inicio = limite_inicio
        fim = min(limite_fim, inicio + pd.DateOffset(years=n) - pd.Timedelta(days=1))
    else:
        raise ValueError('SELECAO_ANOS deve ser recentes ou iniciais.')
    return inicio, fim


def configurar_ft(rede):
    for p in rede.parameters():
        p.requires_grad = False
    if MODO_FINE_TUNING == 'FC':
        for p in rede.saida.parameters():
            p.requires_grad = True
    elif MODO_FINE_TUNING == 'FC_LSTM':
        for p in rede.saida.parameters():
            p.requires_grad = True
        # PyTorch LSTM: ultima camada nomeada *_l{CAMADAS-1}
        for nome, p in rede.lstm.named_parameters():
            if nome.endswith(f'_l{int(cfg["camadas"])-1}'):
                p.requires_grad = True
    elif MODO_FINE_TUNING == 'TODAS':
        for p in rede.parameters():
            p.requires_grad = True
    else:
        raise ValueError('MODO_FINE_TUNING: FC, FC_LSTM ou TODAS.')
    return [p for p in rede.parameters() if p.requires_grad]


if EXECUTAR_FINE_TUNING:
    for anos in ANOS_FINE_TUNING:
        seed()
        inicio_ft, fim_ft = periodo_ft(anos)
        selecionado = np.asarray((datas >= inicio_ft) & (datas <= fim_ft))
        ids = np.flatnonzero(selecionado)
        if len(ids) < 20:
            print(f'FT {anos}: menos de 20 sequencias validas; ignorado')
            continue
        # Separacao CRONOLOGICA dentro do periodo selecionado; sem usar teste.
        n_tr = int(len(ids) * (100 - PCT_VALIDACAO_FT) / 100)
        if n_tr < 1 or n_tr >= len(ids):
            print(f'FT {anos}: treino ou validacao vazios; ignorado')
            continue
        ids_tr, ids_va = ids[:n_tr], ids[n_tr:]
        rede = modelo_da_fonte()
        parametros = configurar_ft(rede)
        opt = torch.optim.Adam(parametros, lr=LR_FT, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=FATOR_LR, patience=PACIENCIA_LR, min_lr=LR_MIN)
        criterio = nn.MSELoss()
        Xtr = torch.from_numpy(X[ids_tr])
        ytr = torch.from_numpy(((y[ids_tr]-SY_MEAN)/SY_SCALE)[:, None].astype(np.float32))
        Xva = torch.from_numpy(X[ids_va])
        yva = torch.from_numpy(((y[ids_va]-SY_MEAN)/SY_SCALE)[:, None].astype(np.float32))
        loader_tr = DataLoader(torch.utils.data.TensorDataset(Xtr, ytr),
                               batch_size=BATCH_SIZE, shuffle=True)
        loader_va = DataLoader(torch.utils.data.TensorDataset(Xva, yva),
                               batch_size=BATCH_SIZE, shuffle=False)
        melhor_val, melhor_estado, melhor_epoca, espera = np.inf, None, 0, 0
        historico = []
        for epoca in range(1, EPOCAS_FT+1):
            rede.train()
            soma_tr, qtd_tr = 0.0, 0
            for xb, yb in loader_tr:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                perda = criterio(rede(xb), yb)
                perda.backward()
                opt.step()
                soma_tr += perda.item()*len(xb); qtd_tr += len(xb)
            rede.eval()
            soma_va, qtd_va = 0.0, 0
            with torch.no_grad():
                for xb, yb in loader_va:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    perda = criterio(rede(xb), yb)
                    soma_va += perda.item()*len(xb); qtd_va += len(xb)
            tr_loss, val_loss = soma_tr/qtd_tr, soma_va/qtd_va
            historico.append({'epoca': epoca, 'MSE_treino': tr_loss,
                              'MSE_validacao': val_loss, 'lr': opt.param_groups[0]['lr']})
            if val_loss < melhor_val - 1e-6:
                melhor_val = val_loss
                melhor_estado = copy.deepcopy(rede.state_dict())
                melhor_epoca = epoca
                espera = 0
            else:
                espera += 1
            scheduler.step(val_loss)
            if epoca == 1 or epoca % 10 == 0:
                print(f'FT {anos} | epoca {epoca} | treino={tr_loss:.5f} '
                      f'val={val_loss:.5f} LR={opt.param_groups[0]["lr"]:.6g}')
            if espera >= PACIENCIA_EARLY:
                break
        if melhor_estado is None:
            raise RuntimeError(f'Nenhum estado FT valido para {anos}')
        rede.load_state_dict(melhor_estado)
        etiqueta = f'FT_{anos}_anos'
        pd.DataFrame(historico).to_csv(SAIDA / f'convergencia_{etiqueta}.csv', sep=';', index=False)
        checkpoint = {
            'tipo': 'LSTM_fine_tuning', 'state_dict': {k:v.detach().cpu() for k,v in rede.state_dict().items()},
            'config': cfg, 'features': FEATURES, 'sx_mean': SX_MEAN.tolist(),
            'sx_scale': SX_SCALE.tolist(), 'sy_mean': [SY_MEAN], 'sy_scale': [SY_SCALE],
            'bacia_alvo': BACIA_ALVO, 'anos_ft': str(anos), 'modo_ft': MODO_FINE_TUNING,
            'inicio_ft': str(inicio_ft.date()), 'fim_ft': str(fim_ft.date()),
            'fim_treino_ft': str(datas[ids_tr[-1]].date()),
            'inicio_validacao_ft': str(datas[ids_va[0]].date()),
            'inicio_teste': str(inicio_teste.date()), 'fim_teste': str(fim_teste.date()),
            'melhor_epoca': melhor_epoca,
        }
        torch.save(checkpoint, SAIDA / f'modelo_{etiqueta}.pt')
        avaliar(rede, etiqueta)


# %% 08 | COMPARACAO FINAL
resumo = pd.DataFrame(resultados)
resumo.to_csv(SAIDA / 'comparacao_transferencia.csv', sep=';', index=False)
print('\nCOMPARACAO FINAL:')
print(resumo.to_string(index=False))
print('\nResultados em:', SAIDA)
