# %% 00 | HYDROUAI - LSTM: CHUVA FUTURA E INCERTEZA
"""Base: series_horarias_preenchidas.csv, na mesma pasta do script.
Chuva futura observada e um experimento de previsibilidade, nao operacional.
MDN-GMM recalcula parametros em cada previsao sem atualizar pesos da LSTM.
Faixa central de 90% + PICP (%), MPIW e Brier Score nos hidrogramas.
O Brier avalia o evento Q > limiar, incluindo ocorrencias e nao ocorrencias.
"""
from pathlib import Path
from itertools import product
from math import erfc
import copy, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# %% 01 | CONFIGURACOES
PASTA = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
ARQUIVO = PASTA / 'series_horarias_preenchidas.csv'
SAIDA = PASTA / 'resultados_LSTM_chuva_incerteza'
COL_DATA, COL_P, COL_Q = 'data_hora', ['2463_P', '84100000_P'], '2463_Q'
ENTRADAS = 'P'  # 'P' ou 'PQ'
HORIZONTES_H, JANELAS_H_GRID = [6], [24]
CAMADAS_GRID, UNIDADES_GRID = [1], [32]
PCT_TREINO, PCT_VALIDACAO, PCT_TESTE = 70, 15, 15
EPOCAS_MAX, BATCH_SIZE, LR_INICIAL = 100, 4320, .001
WEIGHT_DECAY = 0.0
USAR_LR_ADAPTATIVO, PACIENCIA_LR, FATOR_LR, LR_MIN = True, 5, .5, 1e-5
USAR_EARLY_STOPPING, PACIENCIA_EARLY, MELHORIA_MINIMA = True, 15, 1e-6
# Etapa 1: chuva futura sem incerteza (comparar com a LSTM da aula anterior).
# Etapa 2: incerteza sem chuva futura.
# Etapa 3: incerteza combinada com chuva futura.
# chuva: 'nenhuma', 'discreta' (cada hora/estacao), 'acumulada' (total/estacao)
# metodo: 'deterministico', 'mc_dropout', 'mdn_gmm'
EXPERIMENTOS = [
    dict(nome='E0_base', etapa=1, chuva='nenhuma', metodo='deterministico'),
    dict(nome='E1_chuva_discreta', etapa=1, chuva='discreta', metodo='deterministico'),
    dict(nome='E1_chuva_acumulada', etapa=1, chuva='acumulada', metodo='deterministico'),
    dict(nome='E2_MC_sem_chuva', etapa=2, chuva='nenhuma', metodo='mc_dropout'),
    dict(nome='E2_GMM_sem_chuva', etapa=2, chuva='nenhuma', metodo='mdn_gmm'),
    dict(nome='E3_MC_discreta', etapa=3, chuva='discreta', metodo='mc_dropout'),
    dict(nome='E3_MC_acumulada', etapa=3, chuva='acumulada', metodo='mc_dropout'),
    dict(nome='E3_GMM_discreta', etapa=3, chuva='discreta', metodo='mdn_gmm'),
    dict(nome='E3_GMM_acumulada', etapa=3, chuva='acumulada', metodo='mdn_gmm'),
]
# Escolha quais experimentos serao treinados nesta execucao.
# Use None para treinar TODOS, ou uma lista com os nomes desejados.
# Exemplo: ['E0_base', 'E1_chuva_acumulada', 'E2_GMM_sem_chuva']
MODELOS_TREINAR = [

    'E0_base',
    'E2_GMM_sem_chuva',
]
# Para executar somente um experimento, use, por exemplo:
# MODELOS_TREINAR = ['E3_GMM_acumulada']
# MODELOS_TREINAR = None  # todos os experimentos

NOMES_DISPONIVEIS = [e['nome'] for e in EXPERIMENTOS]
if MODELOS_TREINAR is None:
    MODELOS_SELECIONADOS = EXPERIMENTOS.copy()
else:
    if isinstance(MODELOS_TREINAR, str):
        raise TypeError('MODELOS_TREINAR deve ser uma lista, nao uma string.')
    if len(MODELOS_TREINAR) != len(set(MODELOS_TREINAR)):
        raise ValueError('Ha nomes repetidos em MODELOS_TREINAR.')
    desconhecidos = set(MODELOS_TREINAR) - set(NOMES_DISPONIVEIS)
    if desconhecidos:
        raise ValueError(f'Modelos desconhecidos: {sorted(desconhecidos)}. Opcoes: {NOMES_DISPONIVEIS}')
    MODELOS_SELECIONADOS = [e for e in EXPERIMENTOS if e['nome'] in MODELOS_TREINAR]
if not MODELOS_SELECIONADOS:
    raise ValueError('Selecione pelo menos um experimento em MODELOS_TREINAR.')
print('Experimentos selecionados:', [e['nome'] for e in MODELOS_SELECIONADOS])

DROPOUT_P, N_MC = .20, 100
K_GAUSSIANAS, N_AMOSTRAS_GMM = 3, 100
N_EVENTOS, JANELA_ANTES_PICO_H, JANELA_DEPOIS_PICO_H = 1, 48, 48
HORAS_PLOT_TESTE, MAX_BATCHES_HACHURADOS = None, None
# Metricas probabilisticas: faixa central fixa de 90% (quantis 0.05 e 0.95).
UNIDADE_Q = 'm³/s'  # Ajuste se a coluna de vazao estiver em outra unidade.
LIMIAR_CHEIA = 50  # Informe uma vazao; None usa o percentil abaixo NO TREINO.
PERCENTIL_LIMIAR_CHEIA = 95  # Percentil estatistico: cerca de 5% acima no treino.
# Esse limiar automatico e didatico; nao representa uma cota oficial de alerta.
# Brier do MC: fracao de N_MC previsoes acima do limiar.
# Brier do GMM: probabilidade analitica da mistura, sem ruido de amostragem.
SALVAR_FIGURAS_DPI = 220
SEMENTE = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert ENTRADAS in ('P','PQ') and PCT_TREINO+PCT_VALIDACAO+PCT_TESTE == 100
assert len({e['nome'] for e in EXPERIMENTOS}) == len(EXPERIMENTOS)
for e in EXPERIMENTOS:
    assert e['chuva'] in ('nenhuma','discreta','acumulada')
    assert e['metodo'] in ('deterministico','mc_dropout','mdn_gmm')
SAIDA.mkdir(exist_ok=True, parents=True)
(SAIDA/'modelos_completos').mkdir(exist_ok=True)
print('Dispositivo:', DEVICE)

# %% 02 | LEITURA E DIVISAO CRONOLOGICA
if not ARQUIVO.exists():
    raise FileNotFoundError(f'CSV nao encontrado: {ARQUIVO}')
df = pd.read_csv(ARQUIVO, sep=';', encoding='utf-8-sig')
if not set([COL_DATA,*COL_P,COL_Q]).issubset(df.columns):
    raise ValueError('Colunas necessarias: '+str([COL_DATA,*COL_P,COL_Q]))
df[COL_DATA] = pd.to_datetime(df[COL_DATA], errors='raise')
if df[COL_DATA].duplicated().any():
    raise ValueError('Datas duplicadas')
df = df.sort_values(COL_DATA).set_index(COL_DATA).asfreq('h')
for c in [*COL_P,COL_Q]:
    df[c] = pd.to_numeric(df[c], errors='coerce')
    if (df[c].dropna()<0).any():
        raise ValueError('Valores negativos em '+c)
i_treino = int(len(df)*PCT_TREINO/100)
i_val = int(len(df)*(PCT_TREINO+PCT_VALIDACAO)/100)
FIM_TREINO, FIM_VALIDACAO = df.index[i_treino-1], df.index[i_val-1]
if LIMIAR_CHEIA is None:
    if not 0 < PERCENTIL_LIMIAR_CHEIA < 100:
        raise ValueError('PERCENTIL_LIMIAR_CHEIA deve estar entre 0 e 100.')
    q_treino = df.loc[:FIM_TREINO, COL_Q].to_numpy(float)
    q_treino = q_treino[np.isfinite(q_treino)]
    if not len(q_treino):
        raise ValueError('Nao ha vazoes de treino validas para definir o limiar.')
    LIMIAR_CHEIA_EFETIVO = float(np.percentile(q_treino, PERCENTIL_LIMIAR_CHEIA))
    ORIGEM_LIMIAR_CHEIA = f'percentil {PERCENTIL_LIMIAR_CHEIA:g} das vazoes do treino'
else:
    LIMIAR_CHEIA_EFETIVO = float(LIMIAR_CHEIA)
    ORIGEM_LIMIAR_CHEIA = 'valor definido em LIMIAR_CHEIA'
if not np.isfinite(LIMIAR_CHEIA_EFETIVO) or LIMIAR_CHEIA_EFETIVO < 0:
    raise ValueError('LIMIAR_CHEIA deve ser finito e nao negativo, na unidade de Q.')
print(f'Limiar do Brier: Q > {LIMIAR_CHEIA_EFETIVO:.3f} {UNIDADE_Q}'
      f' | {ORIGEM_LIMIAR_CHEIA}')
print('Periodo:',df.index.min(),'ate',df.index.max())
print('Ausentes:',df[[*COL_P,COL_Q]].isna().sum().to_dict())
print('Fim treino:',FIM_TREINO,'| fim validacao:',FIM_VALIDACAO)
fig,ax=plt.subplots(figsize=(13,4))
ax.plot(df.index,df[COL_Q],color='.25',lw=.7,label='Q observada')
for nome,a,b,cor in [('Treino',df.index[0],FIM_TREINO,'tab:blue'),
                     ('Validacao',df.index[i_treino],FIM_VALIDACAO,'tab:orange'),
                     ('Teste',df.index[i_val],df.index[-1],'tab:green')]:
    ax.axvspan(a,b,color=cor,alpha=.17,label=nome)
ax.set(title='Divisao temporal - vazao observada',xlabel='Data',ylabel='Vazao')
ax.legend();ax.grid(alpha=.2);plt.tight_layout();plt.show()

# %% 03 | JANELAS HISTORICAS E PRECIPITACAO FUTURA
FEATURES=COL_P+([COL_Q] if ENTRADAS=='PQ' else [])
def montar_sequencias(janela,horizonte,chuva):
    v=df[FEATURES].to_numpy(np.float32)
    p=df[COL_P].to_numpy(np.float32)
    q=df[COL_Q].to_numpy(np.float32)
    xs,fs,ys,origens,alvos=[],[],[],[],[]
    for i in range(janela-1,len(df)-horizonte):
        x=v[i-janela+1:i+1]
        fut=p[i+1:i+horizonte+1]
        y=q[i+horizonte]
        if not np.isfinite(x).all() or not np.isfinite(y):continue
        if chuva!='nenhuma' and not np.isfinite(fut).all():continue
        f=(fut.reshape(-1) if chuva=='discreta' else
           fut.sum(axis=0) if chuva=='acumulada' else np.empty(0,np.float32))
        xs.append(x);fs.append(f);ys.append(y)
        origens.append(df.index[i]);alvos.append(df.index[i+horizonte])
    if not xs:raise ValueError('Nenhuma sequencia valida')
    return np.stack(xs),np.stack(fs),np.asarray(ys,np.float32),pd.DatetimeIndex(origens),pd.DatetimeIndex(alvos)

def mascaras_temporais(datas):
    return (np.asarray(datas<=FIM_TREINO),
            np.asarray((datas>FIM_TREINO)&(datas<=FIM_VALIDACAO)),
            np.asarray(datas>FIM_VALIDACAO))

X_EX,YF_EX,Y_EX,OR_EX,AL_EX=montar_sequencias(JANELAS_H_GRID[0],HORIZONTES_H[0],'nenhuma')
print('LSTM:',X_EX.shape,'| Q alvo:',Y_EX.shape)
print(pd.DataFrame({'origem':OR_EX[:5],'alvo':AL_EX[:5],'Q':Y_EX[:5]}))

# %% 04 | LSTM E CABECAS DE SAIDA
class LSTMModel(nn.Module):
    def __init__(self,n_features,n_futuro,unidades,camadas,metodo,dropout,k):
        super().__init__()
        self.metodo,self.k=metodo,k
        self.lstm=nn.LSTM(n_features,unidades,num_layers=camadas,batch_first=True)
        # Dropout explicito funciona tambem com apenas UMA camada LSTM
        self.dropout=nn.Dropout(dropout if metodo=='mc_dropout' else 0.)
        self.saida=nn.Linear(unidades+n_futuro,3*k if metodo=='mdn_gmm' else 1)
    def forward(self,x,f):
        seq,_=self.lstm(x)
        return self.saida(torch.cat([self.dropout(seq[:,-1,:]),f],dim=1))

def criar_modelo(cfg,d):
    return LSTMModel(d['X'].shape[-1],d['F'].shape[-1],cfg['unidades'],
                     cfg['camadas'],cfg['metodo'],DROPOUT_P,K_GAUSSIANAS).to(DEVICE)

def parametros_gmm(out,k):
    logits,mu,raw_sigma=torch.split(out,k,dim=-1)
    return torch.softmax(logits,dim=-1),mu,torch.nn.functional.softplus(raw_sigma)+1e-4

def nll_gmm(out,y,k):
    pi,mu,sigma=parametros_gmm(out,k)
    lp=-.5*((y-mu)/sigma)**2-torch.log(sigma)-.5*np.log(2*np.pi)
    return -torch.logsumexp(torch.log(pi.clamp_min(1e-12))+lp,dim=-1).mean()

# %% 05 | NORMALIZACAO E METRICAS
CACHE={}
def preparar(cfg):
    chave=(cfg['janela_h'],cfg['horizonte_h'],cfg['chuva'])
    if chave in CACHE:return CACHE[chave]
    X,F,y,origens,datas=montar_sequencias(*chave)
    tr,va,te=mascaras_temporais(datas)
    if not(tr.any() and va.any() and te.any()):raise ValueError('Particao vazia')
    sx=StandardScaler().fit(X[tr].reshape(-1,X.shape[-1]))
    sy=StandardScaler().fit(y[tr,None])
    Xn=sx.transform(X.reshape(-1,X.shape[-1])).reshape(X.shape).astype(np.float32)
    yn=sy.transform(y[:,None]).ravel().astype(np.float32)
    sf=StandardScaler().fit(F[tr]) if F.shape[-1] else None
    Fn=sf.transform(F).astype(np.float32) if sf else F.astype(np.float32)
    d=dict(X=Xn,F=Fn,y=y,yn=yn,origem=origens,datas=datas,
           tr=tr,va=va,te=te,sx=sx,sy=sy,sf=sf,
           )
    CACHE[chave]=d
    return d

def metricas(obs,pred):
    obs,pred=np.asarray(obs,float),np.asarray(pred,float)
    den=np.sum((obs-obs.mean())**2)
    nse=1-np.sum((pred-obs)**2)/den if den>0 else np.nan
    r=np.corrcoef(obs,pred)[0,1] if np.std(obs)>0 and np.std(pred)>0 else np.nan
    a=np.std(pred)/np.std(obs) if np.std(obs)>0 else np.nan
    b=np.mean(pred)/np.mean(obs) if np.mean(obs)!=0 else np.nan
    kge=1-np.sqrt((r-1)**2+(a-1)**2+(b-1)**2)
    pbias=100*np.sum(obs-pred)/np.sum(obs) if np.sum(obs)!=0 else np.nan
    return dict(NSE=nse,KGE=kge,RMSE=np.sqrt(np.mean((pred-obs)**2)),
                MAE=np.mean(np.abs(pred-obs)),PBIAS=pbias)

def fhv_2pct(obs,pred):
    obs,pred=np.asarray(obs,float),np.asarray(pred,float)
    ids=np.argsort(obs)[-max(1,int(np.ceil(.02*len(obs)))):]
    return 100*np.sum(pred[ids]-obs[ids])/np.sum(obs[ids]) if np.sum(obs[ids]) else np.nan

def metricas_incerteza(tab):
    """PICP em %, MPIW na unidade de Q e Brier em [0,1], no periodo de tab.

    As tres metricas usam os mesmos pares finitos, com e sem Q > limiar.
    Um instante com previsao probabilistica invalida e excluido por inteiro.
    Os quantis sozinhos nao permitem calcular o Brier: prob_excedencia e exigida.
    Modelos deterministicos recebem NaN, pois nao fornecem a faixa avaliada.
    """
    resultado = dict(PICP_90_pct=np.nan, MPIW_90=np.nan, Brier_Score=np.nan,
                     n_prob_validos=0, n_prob_excluidos=len(tab),
                     n_eventos_limiar=0, limiar_cheia=np.nan)
    colunas = ['observado', 'q05', 'q95', 'prob_excedencia', 'limiar_cheia']
    if tab.empty or not set(colunas).issubset(tab.columns):
        return resultado
    obs, inferior, superior, prob, limiares = tab[colunas].to_numpy(float).T
    limiares_finitos = limiares[np.isfinite(limiares)]
    if len(limiares_finitos):
        if not np.all(limiares_finitos == limiares_finitos[0]):
            raise ValueError('A tabela deve usar um unico limiar de cheia.')
        resultado['limiar_cheia'] = float(limiares_finitos[0])
    if np.any(np.isfinite(limiares) & (limiares < 0)):
        raise ValueError('O limiar deve ser nao negativo, na unidade de Q.')
    if np.any(np.isfinite(prob) & ((prob < 0) | (prob > 1))):
        raise ValueError('prob_excedencia deve estar entre 0 e 1.')
    limites_finitos = np.isfinite(inferior) & np.isfinite(superior)
    if np.any(limites_finitos & (inferior > superior)):
        raise ValueError('Ha quantis invertidos: q05 > q95.')
    validos = np.isfinite(np.column_stack([obs, inferior, superior, prob, limiares])).all(axis=1)
    n = int(validos.sum())
    resultado.update(n_prob_validos=n, n_prob_excluidos=len(tab)-n)
    if n:
        o, li, ls, p, lim = (v[validos] for v in (obs, inferior, superior, prob, limiares))
        evento = (o > lim).astype(float)
        resultado.update(PICP_90_pct=float(100*np.mean((o >= li) & (o <= ls))),
                         MPIW_90=float(np.mean(ls-li)),
                         Brier_Score=float(np.mean((p-evento)**2)),
                         n_eventos_limiar=int(evento.sum()))
    return resultado

# %% 06 | TREINAMENTO, EARLY STOPPING E LR ADAPTATIVO
def fixar_semente():
    random.seed(SEMENTE);np.random.seed(SEMENTE);torch.manual_seed(SEMENTE)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(SEMENTE)

def treinar(modelo,cfg,d):
    tr,va=d['tr'],d['va']
    loader=DataLoader(TensorDataset(torch.from_numpy(d['X'][tr]),torch.from_numpy(d['F'][tr]),
                                    torch.from_numpy(d['yn'][tr,None])),
                      batch_size=BATCH_SIZE,shuffle=False)
    xv=torch.from_numpy(d['X'][va]).to(DEVICE)
    fv=torch.from_numpy(d['F'][va]).to(DEVICE)
    yv=torch.from_numpy(d['yn'][va,None]).to(DEVICE)
    opt=torch.optim.Adam(modelo.parameters(),lr=LR_INICIAL,weight_decay=WEIGHT_DECAY)
    scheduler=(torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',patience=PACIENCIA_LR,
                 factor=FATOR_LR,min_lr=LR_MIN) if USAR_LR_ADAPTATIVO else None)
    def loss(out,y):
        if cfg['metodo']=='mdn_gmm':return nll_gmm(out,y,K_GAUSSIANAS)
        return torch.mean((out-y)**2)
    best=np.inf;estado=None;epoca_melhor=0;espera=0;historico=[]
    for epoca in range(1,EPOCAS_MAX+1):
        modelo.train();soma=0.;cont=0
        for xb,fb,yb in loader:
            xb,fb,yb=(a.to(DEVICE) for a in (xb,fb,yb))
            opt.zero_grad();l=loss(modelo(xb,fb),yb)
            l.backward();opt.step()
            soma+=l.item()*len(xb);cont+=len(xb)
        modelo.eval()
        with torch.no_grad():val=loss(modelo(xv,fv),yv).item()
        historico.append(dict(epoca=epoca,treino=soma/cont,validacao=val,
                              lr=opt.param_groups[0]['lr']))
        if val<best-MELHORIA_MINIMA:
            best=val;estado=copy.deepcopy(modelo.state_dict());epoca_melhor=epoca;espera=0
        else:espera+=1
        if scheduler:scheduler.step(val)
        if epoca==1 or epoca%10==0:
            print(f'    epoca {epoca:3d}: treino={soma/cont:.5f} val={val:.5f} LR={opt.param_groups[0]["lr"]:.5g}')
        if USAR_EARLY_STOPPING and espera>=PACIENCIA_EARLY:
            print('    early stopping na epoca',epoca);break
    modelo.load_state_dict(estado)
    return modelo,pd.DataFrame(historico),epoca_melhor,best

# %% 06B | BATCHES HACHURADOS (COMO NA AULA ANTERIOR)
_cfg0=dict(janela_h=JANELAS_H_GRID[0],horizonte_h=HORIZONTES_H[0],chuva='nenhuma')
_dvis=preparar(_cfg0)
_dt=_dvis['datas'][_dvis['tr']]
_nb=int(np.ceil(len(_dt)/BATCH_SIZE))
fig,ax=plt.subplots(figsize=(15,4.5))
ax.plot(df.index,df[COL_Q],color='.15',lw=.65)
for i in range(_nb if MAX_BATCHES_HACHURADOS is None else min(_nb,MAX_BATCHES_HACHURADOS)):
    ini=i*BATCH_SIZE;fim=min((i+1)*BATCH_SIZE,len(_dt))-1
    ax.axvspan(_dt[ini],_dt[fim],facecolor='tab:blue' if i%2==0 else 'tab:orange',
               alpha=.1,edgecolor='.45',hatch='///',linewidth=0)
ax.axvline(FIM_TREINO,color='.4',ls='--');ax.axvline(FIM_VALIDACAO,color='.4',ls='--')
ax.set(title=f'Serie completa | batches consecutivos no treino (N={BATCH_SIZE})',
       xlabel='Data',ylabel='Vazao');ax.grid(alpha=.15)
plt.tight_layout();plt.show()
print('Amostras treino:',len(_dt),'| batches:',_nb)

# %% 07 | GRID SEARCH - SELECAO EXCLUSIVA PELA VALIDACAO
# MDN: NLL; demais modelos: MSE. Selecao apenas pela validacao.
grade=[dict(**e,horizonte_h=h,janela_h=j,camadas=c,unidades=u)
       for e,h,j,c,u in product(MODELOS_SELECIONADOS,HORIZONTES_H,JANELAS_H_GRID,
                                CAMADAS_GRID,UNIDADES_GRID)]
print('Treinamentos previstos:',len(grade))
resultados_grid=[];melhores={}
for idx,cfg in enumerate(grade,1):
    print(f'\n[{idx}/{len(grade)}]',cfg)
    d=preparar(cfg);fixar_semente();modelo=criar_modelo(cfg,d)
    modelo,hist,melhor_epoca,loss_val=treinar(modelo,cfg,d)
    linha={**cfg,'loss_validacao':loss_val,'melhor_epoca':melhor_epoca,
           'epocas_executadas':len(hist)}
    resultados_grid.append(linha)
    chave=(cfg['nome'],cfg['horizonte_h'])
    if chave not in melhores or loss_val<melhores[chave]['loss_validacao']:
        melhores[chave]={**linha,'estado':copy.deepcopy({k:v.cpu() for k,v in modelo.state_dict().items()}),
                        'historico':hist.copy()}
        print('    -> melhor configuracao na validacao')
    pd.DataFrame(resultados_grid).to_csv(SAIDA/'grid_validacao.csv',sep=';',index=False)
print('\nMELHORES CONFIGURACOES:')
print(pd.DataFrame([{k:v for k,v in m.items() if k not in ('estado','historico')}
                    for m in melhores.values()]).to_string(index=False))

# %% 08A | FUNCOES DE INFERENCIA: QUANTIS E PROBABILIDADE DE EXCEDENCIA
COLUNAS_PREVISAO = ['previsto','q025','q05','q10','q50','q90','q95','q975','prob_excedencia']

def resumir_amostras(amostras,sy,limiar_cheia,media_normalizada=None):
    """Amostras normalizadas em (instantes, sorteios); saidas na unidade de Q.

    Desnormaliza e aplica Q=max(0,Q*) a CADA amostra antes dos quantis/Brier.
    A previsao central conserva a convencao original: max(0, media de Q*).
    Uma linha com qualquer amostra nao finita fica sem quantis/probabilidade.
    """
    a=np.asarray(amostras,float)
    if a.ndim!=2 or a.shape[1]<2:
        raise ValueError('Use amostras 2D (instantes, sorteios), com pelo menos 2 sorteios.')
    if not np.isfinite(limiar_cheia) or limiar_cheia<0:
        raise ValueError('O limiar deve ser finito e nao negativo, na unidade de Q.')
    escala,media=float(sy.scale_[0]),float(sy.mean_[0])
    if not np.isfinite(escala) or escala<=0 or not np.isfinite(media):
        raise ValueError('Normalizacao da vazao invalida.')
    centro=a.mean(axis=1) if media_normalizada is None else np.asarray(media_normalizada,float)
    if centro.shape!=(len(a),):
        raise ValueError('media_normalizada deve ter um valor por instante.')
    saida=np.full((len(a),len(COLUNAS_PREVISAO)),np.nan)
    saida[:,0]=np.maximum(0,centro*escala+media)
    fisicas=np.maximum(0,a*escala+media)
    validas=np.isfinite(a).all(axis=1)&np.isfinite(fisicas).all(axis=1)
    if validas.any():
        quantis=np.quantile(fisicas[validas],[.025,.05,.10,.50,.90,.95,.975],axis=1).T
        saida[validas,1:8]=quantis
        saida[validas,8]=np.mean(fisicas[validas]>limiar_cheia,axis=1)
    return saida

def probabilidade_excedencia_gmm(pi,mu,sigma,sy,limiar_cheia):
    """P(Q > limiar) exata da mistura; nao estima uma normal pelos quantis.

    pi, mu e sigma tem forma (instantes, componentes), na escala normalizada.
    Para limiar >= 0, censurar Q* em zero preserva P(Q > limiar).
    erfc calcula a cauda de cada gaussiana sem subtrair uma CDF proxima de 1.
    """
    pi,mu,sigma=(np.asarray(v,float) for v in (pi,mu,sigma))
    if pi.ndim!=2 or pi.shape!=mu.shape or pi.shape!=sigma.shape or pi.shape[1]==0:
        raise ValueError('pi, mu e sigma devem ter a mesma forma (instantes, componentes).')
    if not np.isfinite(limiar_cheia) or limiar_cheia<0:
        raise ValueError('O limiar deve ser finito e nao negativo, na unidade de Q.')
    escala,media=float(sy.scale_[0]),float(sy.mean_[0])
    if not np.isfinite(escala) or escala<=0 or not np.isfinite(media):
        raise ValueError('Normalizacao da vazao invalida.')
    soma=pi.sum(axis=1)
    validas=(np.isfinite(pi).all(axis=1)&np.isfinite(mu).all(axis=1)
             &np.isfinite(sigma).all(axis=1)&(sigma>0).all(axis=1)
             &(pi>=0).all(axis=1)&np.isfinite(soma)&(soma>0))
    prob=np.full(len(pi),np.nan)
    if validas.any():
        limiar_normalizado=(limiar_cheia-media)/escala
        z=(limiar_normalizado-mu[validas])/(sigma[validas]*np.sqrt(2.))
        cauda=.5*np.vectorize(erfc,otypes=[float])(z)
        pesos=pi[validas]/soma[validas,None]
        prob[validas]=np.clip(np.sum(pesos*cauda,axis=1),0,1)
    return prob

def inferir(modelo,cfg,d,mascara,limiar_cheia=None):
    if limiar_cheia is None:limiar_cheia=LIMIAR_CHEIA_EFETIVO
    if not np.isfinite(limiar_cheia) or limiar_cheia<0:
        raise ValueError('O limiar deve ser finito e nao negativo, na unidade de Q.')
    if cfg['metodo']=='mc_dropout' and (not isinstance(N_MC,int) or N_MC<2):
        raise ValueError('N_MC deve ser um inteiro >= 2.')
    if cfg['metodo']=='mdn_gmm' and (not isinstance(N_AMOSTRAS_GMM,int) or N_AMOSTRAS_GMM<2):
        raise ValueError('N_AMOSTRAS_GMM deve ser um inteiro >= 2.')
    X=d['X'][mascara];F=d['F'][mascara]
    if not len(X):raise ValueError('Nenhuma amostra selecionada para inferencia.')
    sy=d['sy'];out=[]
    modelo.eval()
    if cfg['metodo']=='mc_dropout':
        # Ativar APENAS a camada Dropout; manter LSTM em modo de avaliacao.
        modelo.dropout.train()
    with torch.no_grad():
        for ini in range(0,len(X),min(BATCH_SIZE,1024)):
            xb=torch.from_numpy(X[ini:ini+min(BATCH_SIZE,1024)]).to(DEVICE)
            fb=torch.from_numpy(F[ini:ini+min(BATCH_SIZE,1024)]).to(DEVICE)
            if cfg['metodo']=='deterministico':
                z=modelo(xb,fb).cpu().numpy().ravel()
                z=np.maximum(0,z*float(sy.scale_[0])+float(sy.mean_[0]))
                # Sem distribuicao probabilistica: Brier e faixas ficam sem avaliacao.
                out.append(np.column_stack([z]*8+[np.full(len(z),np.nan)]))
            elif cfg['metodo']=='mc_dropout':
                draws=np.stack([modelo(xb,fb).cpu().numpy().ravel() for _ in range(N_MC)],axis=1)
                out.append(resumir_amostras(draws,sy,limiar_cheia,
                                           media_normalizada=draws.mean(axis=1)))
            else:
                pi,mu,sigma=parametros_gmm(modelo(xb,fb),modelo.k)
                pi,mu,sigma=(z.cpu().numpy() for z in (pi,mu,sigma))
                # GMM dinamica: cada linha tem seus proprios pi, mu e sigma.
                # Quantis estimados por amostragem da mistura condicional.
                rng=np.random.default_rng(SEMENTE+ini)
                # Um uniforme por amostra e por sorteio; escolha categorica entre K componentes.
                # Nao usar um mesmo uniforme contra todos os pesos acumulados com eixo incorreto.
                acumulada=np.cumsum(pi,axis=1)
                u=rng.random((len(pi),N_AMOSTRAS_GMM))
                comp=(u[:,:,None] > acumulada[:,None,:]).sum(axis=2)
                comp=np.clip(comp,0,pi.shape[1]-1)
                sorteios=(np.take_along_axis(mu,comp,axis=1)
                          +np.take_along_axis(sigma,comp,axis=1)*rng.standard_normal(comp.shape))
                media=np.sum(pi*mu,axis=1)
                bloco=resumir_amostras(sorteios,sy,limiar_cheia,media_normalizada=media)
                # Quantis por amostragem; probabilidade de excedencia pela CDF exata.
                bloco[:,-1]=probabilidade_excedencia_gmm(pi,mu,sigma,sy,limiar_cheia)
                out.append(bloco)
    modelo.eval()
    z=np.concatenate(out,axis=0)
    tab=pd.DataFrame(z,columns=COLUNAS_PREVISAO,
                     index=d['datas'][mascara]).rename_axis('data_alvo')
    tab['limiar_cheia']=float(limiar_cheia)
    return tab

# %% 08B | AVALIACAO E EXPORTACAO DO TESTE INDEPENDENTE
resumo_teste=[];previsoes_teste={}
for chave,m in melhores.items():
    cfg={k:m[k] for k in ('nome','etapa','chuva','metodo','horizonte_h','janela_h','camadas','unidades')}
    d=preparar(cfg);modelo=criar_modelo(cfg,d)
    modelo.load_state_dict(m['estado']);modelo.to(DEVICE)
    tab=inferir(modelo,cfg,d,d['te'])
    tab.insert(0,'observado',d['y'][d['te']])
    previsoes_teste[chave]=tab
    met=metricas(tab.observado,tab.previsto)
    met['FHV_2pct']=fhv_2pct(tab.observado,tab.previsto)
    met.update(metricas_incerteza(tab))
    met['origem_limiar_cheia']=ORIGEM_LIMIAR_CHEIA
    met['n_instantes_teste']=len(tab)
    if cfg['metodo']!='deterministico':
        for nivel,lo,hi in [('80','q10','q90'),('95','q025','q975')]:
            met['PICP_'+nivel]=float(((tab.observado>=tab[lo])&(tab.observado<=tab[hi])).mean())
            met['MPIW_'+nivel]=float((tab[hi]-tab[lo]).mean())
    resumo_teste.append({**cfg,**met})
    tab.to_csv(SAIDA/f'previsoes_{cfg["nome"]}_h{cfg["horizonte_h"]}.csv',sep=';')
    m['historico'].to_csv(SAIDA/f'convergencia_{cfg["nome"]}_h{cfg["horizonte_h"]}.csv',sep=';',index=False)
resumo=pd.DataFrame(resumo_teste)
resumo.to_csv(SAIDA/'comparacao_teste.csv',sep=';',index=False)
print('\nRESULTADOS NO TESTE:')
print(resumo[['nome','horizonte_h','NSE','KGE','RMSE','PBIAS','FHV_2pct',
              'PICP_90_pct','MPIW_90','Brier_Score','n_prob_validos']].to_string(index=False))
print('PICP_90_pct em %; MPIW_90 em',UNIDADE_Q,'; Brier_Score adimensional.')
print('PICP_80 e PICP_95 conservam a escala original (fracao de 0 a 1).')

# %% 09 | CONVERGENCIA E LEARNING RATE
for (nome,h),m in melhores.items():
    hist=m['historico'];fig,ax=plt.subplots(2,1,figsize=(10,6),sharex=True,
                                           gridspec_kw={'height_ratios':[2,1]})
    ax[0].plot(hist.epoca,hist.treino,label='Treino')
    ax[0].plot(hist.epoca,hist.validacao,label='Validacao')
    ax[0].axvline(m['melhor_epoca'],ls='--',color='tab:green',label=f'Melhor epoca: {m["melhor_epoca"]}')
    ax[0].set(ylabel='NLL (alvo normalizado)' if m['metodo']=='mdn_gmm' else 'Loss (alvo normalizado)',
              title=f'{nome} +{h}h - convergencia')
    ax[0].legend();ax[0].grid(alpha=.2)
    ax[1].step(hist.epoca,hist.lr,where='post')
    ax[1].set(xlabel='Epoca',ylabel='Learning rate');ax[1].grid(alpha=.2)
    plt.tight_layout();plt.savefig(SAIDA/f'convergencia_{nome}_h{h}.png',dpi=180);plt.show()

# %% 10A | FUNCAO DO HIDROGRAMA: FAIXA 90% E TRES METRICAS
def plotar_hidrograma(ax, tab, titulo):
    """Hidrograma e métricas calculadas exclusivamente nas linhas de ``tab``.

    Usa metricas_incerteza(tab) e UNIDADE_Q definidos no script principal.
    Os cartões ficam acima do eixo e participam do cálculo de tight_layout.
    Retorna o dicionário de métricas, inclusive para tabelas vazias.
    """
    import numpy as np
    import matplotlib.dates as mdates
    from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker

    met = metricas_incerteza(tab)
    azul, faixa, escuro = '#126D91', '#A9D7E5', '#172E43'
    laranja, vermelho, cinza = '#C47C12', '#C44A46', '#657487'
    ax.set_xlabel('Data', fontsize=9)
    ax.set_ylabel(f'Vazão ({UNIDADE_Q})', fontsize=9)
    if tab.empty:
        ax.set_title(titulo, fontsize=10, loc='left')
        return met

    obs = tab['observado'].to_numpy(dtype=float)
    prev = tab['previsto'].to_numpy(dtype=float)
    tem_colunas = {'q05', 'q95', 'prob_excedencia'}.issubset(tab.columns)
    probabilistico = False
    if tem_colunas:
        lo = tab['q05'].to_numpy(dtype=float)
        hi = tab['q95'].to_numpy(dtype=float)
        prob = tab['prob_excedencia'].to_numpy(dtype=float)
        limites_finitos = np.isfinite(lo) & np.isfinite(hi)
        if np.any(limites_finitos & (lo > hi)):
            raise ValueError('Há q05 maior que q95 na janela do hidrograma.')
        if np.any(np.isfinite(prob) & ((prob < 0) | (prob > 1))):
            raise ValueError('prob_excedencia deve estar entre 0 e 1.')
        # Determinísticos têm q05=q95 e probabilidades ausentes. Uma faixa
        # não degenerada sem probabilidades continua identificada como tal.
        probabilistico = (np.isfinite(prob).any() or
                          np.any(limites_finitos & (hi > lo)))

    ax.plot(tab.index, np.where(np.isfinite(obs), obs, np.nan),
            color=escuro, lw=1.35, label='Observado', zorder=5)
    ax.plot(tab.index, np.where(np.isfinite(prev), prev, np.nan),
            color=azul, ls='--', lw=1.3, label='Previsão central', zorder=6)
    limiar = float(met['limiar_cheia'])
    valores_eixo = [obs, prev]

    def fmt(valor, casas=1):
        return f'{valor:.{casas}f}'.replace('.', ',') if np.isfinite(valor) else 'N/D'

    if np.isfinite(limiar):
        ax.axhline(limiar, color=laranja, ls=(0, (5, 4)), lw=1.2,
                   label=f'Limiar: {fmt(limiar)} {UNIDADE_Q}', zorder=3)
        valores_eixo.append(np.array([limiar]))

    if probabilistico:
        # Máscara comum às três métricas. A probabilidade foi calculada na
        # inferência; não é inferida da largura nem da posição dos quantis.
        valido = (np.isfinite(obs) & limites_finitos & np.isfinite(prob)
                  & (prob >= 0) & (prob <= 1) & np.isfinite(limiar))
        if 'limiar_cheia' in tab:
            valido &= np.isfinite(tab['limiar_cheia'].to_numpy(dtype=float))
        lo_plot, hi_plot = np.where(valido, lo, np.nan), np.where(valido, hi, np.nan)
        ax.fill_between(tab.index, lo_plot, hi_plot, color=faixa, alpha=.85,
                        label='Faixa nominal de 90%', zorder=1)
        ax.plot(tab.index, lo_plot, color=azul, lw=.55, alpha=.35, zorder=2)
        ax.plot(tab.index, hi_plot, color=azul, lw=.55, alpha=.35, zorder=2)
        fora = valido & ((obs < lo) | (obs > hi))
        ax.scatter(tab.index[fora], obs[fora], s=22, facecolors='white',
                   edgecolors=vermelho, linewidths=1.1,
                   label='Observação fora da faixa', zorder=7)
        valores_eixo.extend([lo_plot, hi_plot])

        cards = [
            ('Cobertura · PICP', f"{fmt(met['PICP_90_pct'])}%", 'Referência: 90%', azul),
            ('Largura · MPIW', f"{fmt(met['MPIW_90'])} {UNIDADE_Q}", 'Menor, com cobertura adequada', escuro),
            ('Brier Score', fmt(met['Brier_Score'], 3), 'Menor é melhor', laranja),
        ]
        for x, (rotulo, valor, nota, cor) in zip((1/6, 1/2, 5/6), cards):
            bloco = VPacker(children=[
                TextArea(rotulo, textprops={'fontsize': 7.8, 'color': cinza, 'weight': 'bold'}),
                TextArea(valor, textprops={'fontsize': 12, 'color': cor, 'weight': 'bold'}),
                TextArea(nota, textprops={'fontsize': 6.6, 'color': cinza}),
            ], align='center', pad=0, sep=2)
            caixa = AnnotationBbox(
                bloco, (x, 1), xycoords=ax.transAxes, xybox=(0, 21),
                boxcoords='offset points', box_alignment=(.5, 0),
                frameon=True, annotation_clip=False,
                bboxprops={'boxstyle': 'round,pad=.4,rounding_size=.2',
                           'facecolor': '#F3F7FA', 'edgecolor': '#DFE8EE', 'linewidth': .65},
            )
            caixa.set_clip_on(False)
            caixa.set_in_layout(True)
            ax.add_artist(caixa)
        nota = ax.annotate(
            f"Período exibido: {met['n_prob_validos']} válidos; "
            f"{met['n_prob_excluidos']} excluídos. Brier inclui eventos e não eventos.",
            (0, 1), xycoords=ax.transAxes, xytext=(0, 4),
            textcoords='offset points', ha='left', va='bottom',
            fontsize=7, color=cinza, annotation_clip=False,
        )
        nota.set_in_layout(True)
        ax.set_title(titulo, fontsize=10, loc='left', pad=76)
    else:
        ax.set_title(titulo, fontsize=10, loc='left', pad=10)

    # Reserva espaço vertical para a legenda, mantendo os dados desobstruídos.
    dados_eixo = np.concatenate(valores_eixo)
    dados_eixo = dados_eixo[np.isfinite(dados_eixo)]
    if dados_eixo.size:
        minimo, maximo = dados_eixo.min(), dados_eixo.max()
        amplitude = max(maximo - minimo, .05 * max(abs(maximo), 1.0))
        fundo = minimo - .04 * amplitude
        ax.set_ylim(max(0, fundo) if minimo >= 0 else fundo,
                    maximo + (.30 if probabilistico else .18) * amplitude)
    ax.legend(loc='upper left', ncol=3 if probabilistico else 2,
              fontsize=7.1, frameon=False, columnspacing=1.2, handlelength=1.9)
    ax.grid(axis='y', color='#E6EDF1', lw=.7)
    ax.set_axisbelow(True)
    ax.margins(x=.015)
    ax.tick_params(axis='both', labelsize=8, length=0, pad=5, colors=cinza)
    if getattr(tab.index, 'inferred_type', '') in ('datetime64', 'datetime'):
        locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    for lado in ('top', 'right'):
        ax.spines[lado].set_visible(False)
    for lado in ('left', 'bottom'):
        ax.spines[lado].set_color('#C4D0D9')
    return met

# %% 10B | HIDROGRAMAS INDIVIDUAIS DO TESTE
for (nome,h),tab in previsoes_teste.items():
    sub=tab if HORAS_PLOT_TESTE is None else tab.iloc[:HORAS_PLOT_TESTE]
    fig,ax=plt.subplots(figsize=(14,6))
    nse_janela=metricas(sub.observado,sub.previsto)['NSE'] if len(sub) else np.nan
    plotar_hidrograma(ax,sub,f'{nome} +{h}h | teste | NSE no periodo exibido={nse_janela:.3f}')
    plt.tight_layout();plt.savefig(SAIDA/f'hidrograma_{nome}_h{h}.png',dpi=SALVAR_FIGURAS_DPI);plt.show()

# %% 10C | COMPARACAO LADO A LADO: MC DROPOUT E GMM
# Mesma chuva futura, horizonte, datas e escala vertical; as caixas usam esse recorte.
for chuva in ('nenhuma','discreta','acumulada'):
    nomes=[e['nome'] for e in MODELOS_SELECIONADOS
           if e['chuva']==chuva and e['metodo']!='deterministico']
    for h in HORIZONTES_H:
        chaves=[(nome,h) for nome in nomes if (nome,h) in previsoes_teste]
        if len(chaves)<2:continue
        comuns=previsoes_teste[chaves[0]].index
        for chave in chaves[1:]:
            comuns=comuns.intersection(previsoes_teste[chave].index)
        comuns=comuns.sort_values()
        if HORAS_PLOT_TESTE is not None:comuns=comuns[:HORAS_PLOT_TESTE]
        if not len(comuns):continue
        # Mesmo conjunto valido para as metricas dos dois paineis, inclusive
        # se um dos modelos tiver uma previsao nao finita em determinado alvo.
        validos_comuns=np.ones(len(comuns),dtype=bool)
        col_avaliacao=['observado','q05','q95','prob_excedencia','limiar_cheia']
        for chave in chaves:
            validos_comuns &= np.isfinite(previsoes_teste[chave].loc[comuns,col_avaliacao].to_numpy(float)).all(axis=1)
        fig,axes=plt.subplots(1,len(chaves),figsize=(7.3*len(chaves),5.8),
                              sharey=True,squeeze=False)
        limites_y=[]
        for ax,chave in zip(axes.flat,chaves):
            tab_comparacao=previsoes_teste[chave].loc[comuns].copy()
            tab_comparacao.loc[~validos_comuns,['q05','q95','prob_excedencia']]=np.nan
            plotar_hidrograma(ax,tab_comparacao,f'{chave[0]} | +{h}h')
            limites_y.append(ax.get_ylim())
        # Em eixos compartilhados, o ultimo painel nao pode cortar a faixa do primeiro.
        axes[0,0].set_ylim(min(v[0] for v in limites_y),max(v[1] for v in limites_y))
        fig.suptitle(f'Faixa de incerteza de 90% | chuva futura: {chuva} | mesmas datas de teste',fontsize=13)
        plt.tight_layout()
        plt.savefig(SAIDA/f'comparacao_incerteza_{chuva}_h{h}.png',dpi=SALVAR_FIGURAS_DPI)
        plt.show()

# %% 11 | EVENTOS DE CHEIA: MESMO PICO OBSERVADO PARA TODOS OS EXPERIMENTOS
indices=[set(tab.index) for tab in previsoes_teste.values()]
datas_comuns=pd.DatetimeIndex(sorted(set.intersection(*indices)))
if len(datas_comuns)==0:raise ValueError('Nao ha datas comuns entre os experimentos')
obs_ref=next(iter(previsoes_teste.values())).loc[datas_comuns,'observado']
escolhidos=[]
for data in obs_ref.sort_values(ascending=False).index:
    if all(abs(data-anterior)>pd.Timedelta(hours=JANELA_ANTES_PICO_H+JANELA_DEPOIS_PICO_H)
           for anterior in escolhidos):escolhidos.append(data)
    if len(escolhidos)>=N_EVENTOS:break
linhas_eventos=[]
for ie,pico in enumerate(sorted(escolhidos),1):
    ini=pico-pd.Timedelta(hours=JANELA_ANTES_PICO_H)
    fim=pico+pd.Timedelta(hours=JANELA_DEPOIS_PICO_H)
    datas_evento=datas_comuns[(datas_comuns>=ini)&(datas_comuns<=fim)]
    for nome in [e['nome'] for e in MODELOS_SELECIONADOS]:
        fig,axes=plt.subplots(1,len(HORIZONTES_H),figsize=(7.2*len(HORIZONTES_H),5.5),squeeze=False)
        for ih,h in enumerate(HORIZONTES_H):
            if (nome,h) not in previsoes_teste:continue
            tab=previsoes_teste[(nome,h)].loc[datas_evento]
            if len(tab)<2:continue
            plotar_hidrograma(axes[0,ih],tab,f'Evento {ie} | {nome} +{h}h')
            po=tab.observado.max();pp=tab.previsto.max()
            linhas_eventos.append(dict(evento=ie,data_pico=pico,modelo=nome,horizonte_h=h,
                  erro_pico_pct=100*(pp-po)/po if po else np.nan,
                  erro_tempo_pico_h=(tab.previsto.idxmax()-tab.observado.idxmax()).total_seconds()/3600,
                  NSE_evento=metricas(tab.observado,tab.previsto)['NSE'],
                  **metricas_incerteza(tab)))
        plt.tight_layout();plt.savefig(SAIDA/f'evento_{ie}_{nome}.png',dpi=SALVAR_FIGURAS_DPI);plt.show()
tabela_eventos=pd.DataFrame(linhas_eventos)
tabela_eventos.to_csv(SAIDA/'metricas_eventos.csv',sep=';',index=False)
print('\nMETRICAS DOS EVENTOS:')
print(tabela_eventos.to_string(index=False))

# %% 11B | RECESSAO: JANELA APOS O PICO E METRICAS ESPECIFICAS
# Janela de recessao: desde o pico observado ate JANELA_RECESSAO_H horas depois.
# Evitar interpretar como recessao pura se houver um segundo evento de chuva.
JANELA_RECESSAO_H = 36
linhas_recessao=[]
for ie,pico in enumerate(sorted(escolhidos),1):
    fim_recessao=pico+pd.Timedelta(hours=JANELA_RECESSAO_H)
    datas_recessao=datas_comuns[(datas_comuns>=pico)&(datas_comuns<=fim_recessao)]
    for (nome,h),tab_completo in previsoes_teste.items():
        tab=tab_completo.loc[datas_recessao]
        if len(tab)<3:continue
        met=metricas(tab.observado,tab.previsto)
        # Mudanca de vazao por hora: erro na taxa de recessao.
        dq_obs=np.diff(tab.observado.to_numpy())
        dq_prev=np.diff(tab.previsto.to_numpy())
        linha=dict(evento=ie,data_pico=pico,modelo=nome,horizonte_h=h,
                   n_pontos=len(tab),RMSE_recessao=met['RMSE'],NSE_recessao=met['NSE'],
                   MAE_variacao_h=float(np.mean(np.abs(dq_prev-dq_obs))))
        linha.update(metricas_incerteza(tab))
        if 'q10' in tab and (tab.q90-tab.q10).max()>0:
            linha['PICP80_recessao']=float(((tab.observado>=tab.q10)&(tab.observado<=tab.q90)).mean())
            linha['PICP95_recessao']=float(((tab.observado>=tab.q025)&(tab.observado<=tab.q975)).mean())
        linhas_recessao.append(linha)
        fig,ax=plt.subplots(figsize=(10,5.5))
        plotar_hidrograma(ax,tab,f'Recessao apos pico | {nome} +{h}h | evento {ie}')
        plt.tight_layout()
        plt.savefig(SAIDA/f'recessao_evento{ie}_{nome}_h{h}.png',dpi=SALVAR_FIGURAS_DPI)
        plt.show()
pd.DataFrame(linhas_recessao).to_csv(SAIDA/'metricas_recessao.csv',sep=';',index=False)

# %% 11C | COMPARACAO DIDATICA POR ETAPA: MAXIMAS E RECESSAO
# Compara experimentos na mesma janela temporal, para cada horizonte.
# A etapa 1 inclui a referencia sem chuva futura da aula anterior.
# Etapas 2 e 3 mostram faixas probabilisticas por experimento.
for etapa in (1,2,3):
    nomes=[e['nome'] for e in MODELOS_SELECIONADOS if e['etapa']==etapa]
    if not nomes:
        continue
    for ie,pico in enumerate(sorted(escolhidos),1):
        for periodo,ini,fim in [
            ('cheia',pico-pd.Timedelta(hours=JANELA_ANTES_PICO_H),
                     pico+pd.Timedelta(hours=JANELA_DEPOIS_PICO_H)),
            ('recessao',pico,pico+pd.Timedelta(hours=JANELA_RECESSAO_H))]:
            fig,axes=plt.subplots(len(nomes),len(HORIZONTES_H),
                                  figsize=(7.2*len(HORIZONTES_H),4.8*len(nomes)),
                                  squeeze=False)
            datas_periodo=datas_comuns[(datas_comuns>=ini)&(datas_comuns<=fim)]
            for i,nome in enumerate(nomes):
                for j,h in enumerate(HORIZONTES_H):
                    if (nome,h) not in previsoes_teste:continue
                    tab=previsoes_teste[(nome,h)].loc[datas_periodo]
                    plotar_hidrograma(axes[i,j],tab,f'{nome} | +{h}h')
            fig.suptitle(f'Etapa {etapa} | {periodo} | evento {ie}',fontsize=13)
            plt.tight_layout()
            plt.savefig(SAIDA/f'comparacao_etapa{etapa}_{periodo}_evento{ie}.png',dpi=SALVAR_FIGURAS_DPI)
            plt.show()

# %% 12 | METRICAS POR HORIZONTE
fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True)
for ax,(col,rotulo) in zip(axes.flat,[('NSE','NSE'),('KGE','KGE'),
                                    ('PBIAS','PBIAS (%)'),('FHV_2pct','FHV 2% (%)')]):
    for nome,sub in resumo.groupby('nome'):
        sub=sub.sort_values('horizonte_h')
        ax.plot(sub.horizonte_h,sub[col],'o-',label=nome)
    if col in ('PBIAS','FHV_2pct'):ax.axhline(0,color='.5',ls='--',lw=.9)
    ax.set(xlabel='Horizonte (h)',ylabel=rotulo);ax.grid(alpha=.2);ax.legend(fontsize=7)
plt.suptitle('LSTM - metricas no teste independente')
plt.tight_layout();plt.savefig(SAIDA/'metricas_por_horizonte.png',dpi=180);plt.show()

# %% 13 | SALVAR MODELOS COMPLETOS (PESOS + NORMALIZACAO + CONFIGURACAO)
for (nome,h),m in melhores.items():
    cfg={k:m[k] for k in ('nome','etapa','chuva','metodo','horizonte_h','janela_h','camadas','unidades')}
    d=preparar(cfg)
    pacote=dict(config=cfg,features=list(FEATURES),col_p=list(COL_P),col_q=COL_Q,
                state_dict=m['estado'],dropout_p=DROPOUT_P,k_gaussianas=K_GAUSSIANAS,
                sx_mean=d['sx'].mean_.tolist(),sx_scale=d['sx'].scale_.tolist(),
                sy_mean=d['sy'].mean_.tolist(),sy_scale=d['sy'].scale_.tolist(),
                sf_mean=d['sf'].mean_.tolist() if d['sf'] else [],
                sf_scale=d['sf'].scale_.tolist() if d['sf'] else [],
                fim_treino=str(FIM_TREINO),fim_validacao=str(FIM_VALIDACAO),
                limiar_cheia=LIMIAR_CHEIA_EFETIVO,origem_limiar_cheia=ORIGEM_LIMIAR_CHEIA,
                percentil_limiar_cheia=PERCENTIL_LIMIAR_CHEIA if LIMIAR_CHEIA is None else None,
                faixa_nominal=.90,unidade_q=UNIDADE_Q,n_mc=N_MC,
                n_amostras_gmm=N_AMOSTRAS_GMM,semente=SEMENTE)
    destino=SAIDA/'modelos_completos'/f'{nome}_h{h}.pt'
    torch.save(pacote,destino);print('Modelo salvo:',destino)

# %% 14 | REABRIR MODELO SALVO SEM TREINAR NOVAMENTE
# Em uma sessao nova, execute antes: 00, 01, 02, 03, 04, 05, 08A e 10A.
# As celulas 07 e 08B NAO sao necessarias para reabrir os pesos ja salvos.
# A recarga utiliza os scalers, cortes temporais e limiar persistidos no pacote.
def preparar_com_normalizacao_salva(cfg,pacote):
    if (list(FEATURES)!=pacote['features'] or list(COL_P)!=pacote['col_p']
            or COL_Q!=pacote['col_q']):
        raise ValueError('As colunas e sua ordem devem coincidir com as do modelo salvo.')
    X,F,y,origens,datas=montar_sequencias(cfg['janela_h'],cfg['horizonte_h'],cfg['chuva'])

    def restaurar_scaler(prefixo):
        scaler=StandardScaler()
        scaler.mean_=np.asarray(pacote[prefixo+'_mean'],float)
        scaler.scale_=np.asarray(pacote[prefixo+'_scale'],float)
        if (scaler.mean_.shape!=scaler.scale_.shape
                or not np.isfinite(scaler.mean_).all()
                or not np.isfinite(scaler.scale_).all() or np.any(scaler.scale_<=0)):
            raise ValueError('Normalizador salvo invalido: '+prefixo)
        scaler.var_=scaler.scale_**2
        scaler.n_features_in_=len(scaler.mean_)
        return scaler

    sx,sy=restaurar_scaler('sx'),restaurar_scaler('sy')
    sf=restaurar_scaler('sf') if F.shape[-1] else None
    Xn=sx.transform(X.reshape(-1,X.shape[-1])).reshape(X.shape).astype(np.float32)
    Fn=sf.transform(F).astype(np.float32) if sf is not None else F.astype(np.float32)
    yn=sy.transform(y[:,None]).ravel().astype(np.float32)
    fim_treino=pd.Timestamp(pacote['fim_treino'])
    fim_validacao=pd.Timestamp(pacote['fim_validacao'])
    return dict(X=Xn,F=Fn,y=y,yn=yn,origem=origens,datas=datas,sx=sx,sy=sy,sf=sf,
                tr=np.asarray(datas<=fim_treino),
                va=np.asarray((datas>fim_treino)&(datas<=fim_validacao)),
                te=np.asarray(datas>fim_validacao))

MODELO_ABRIR=MODELOS_SELECIONADOS[0]['nome']  # ou escolha um nome salvo
HORIZONTE_ABRIR=HORIZONTES_H[0]
arquivo_modelo=SAIDA/'modelos_completos'/f'{MODELO_ABRIR}_h{HORIZONTE_ABRIR}.pt'
if arquivo_modelo.exists():
    pacote=torch.load(arquivo_modelo,map_location='cpu',weights_only=True)
    cfg=pacote['config'];d=preparar_com_normalizacao_salva(cfg,pacote)
    # Usar arquitetura e parametros persistidos, sem reestimar pesos.
    modelo=LSTMModel(len(pacote['features']),len(pacote['sf_mean']),
                     cfg['unidades'],cfg['camadas'],cfg['metodo'],
                     pacote['dropout_p'],pacote['k_gaussianas']).to(DEVICE)
    modelo.load_state_dict(pacote['state_dict']);modelo.eval()
    limiar_salvo=float(pacote.get('limiar_cheia',LIMIAR_CHEIA_EFETIVO))
    if 'limiar_cheia' not in pacote:
        print('Pacote antigo, sem limiar salvo: usando o limiar configurado nesta sessao.')
    recarregado=inferir(modelo,cfg,d,d['te'],limiar_cheia=limiar_salvo)
    recarregado.insert(0,'observado',d['y'][d['te']])
    print('Modelo reaberto:',arquivo_modelo.name)
    print('NSE teste:',metricas(recarregado.observado,recarregado.previsto)['NSE'])
    print('Metricas de incerteza no teste:',metricas_incerteza(recarregado))
    recarregado.to_csv(SAIDA/f'previsoes_reaberto_{MODELO_ABRIR}_h{HORIZONTE_ABRIR}.csv',sep=';')
    fig,ax=plt.subplots(figsize=(14,6))
    plotar_hidrograma(ax,recarregado.iloc[:HORAS_PLOT_TESTE] if HORAS_PLOT_TESTE else recarregado,
                      'Modelo reaberto | teste independente')
    plt.tight_layout()
    plt.savefig(SAIDA/f'hidrograma_reaberto_{MODELO_ABRIR}_h{HORIZONTE_ABRIR}.png',dpi=SALVAR_FIGURAS_DPI)
    plt.show()
