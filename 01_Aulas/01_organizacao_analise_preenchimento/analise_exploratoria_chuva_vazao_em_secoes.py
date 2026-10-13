# %% [markdown]
# # 00 - Configuracoes e bibliotecas
# OBJETIVO: entender como as series de chuva e vazao se relacionam no tempo,
# sem treinar modelos de machine learning.
# EXECUCAO: inicie nesta celula e execute na ordem; depois e possivel
# reexecutar uma secao de interesse desde que as variaveis dela existam.
# As linhas '# %%' sao reconhecidas como celulas pelo VS Code e Spyder.

# %%
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mutual_info_score

# ========================== CONFIGURACAO DO USUARIO ==========================
# No terminal, usamos a pasta onde esta o arquivo .py.
# Na execucao por celulas / notebook, __file__ pode nao existir.
try:
    RAIZ = Path(__file__).resolve().parent
except NameError:
    RAIZ = Path.cwd()
# Caso o VS Code escolha a pasta de trabalho em vez da pasta do script,
# ajuste RAIZ manualmente para a pasta que contem o CSV.
ARQUIVO = RAIZ / 'series_horarias_preenchidas.csv'
SAIDA = RAIZ / 'resultados_exploratorios_didaticos'
COL_DATA = 'data_hora'
COLS_P = ['2463_P', '84100000_P']
COL_Q = '2463_Q'

JANELAS_CHUVA_H = [1, 3, 6, 12, 24, 48, 72, 168]
LAGS_Q_H = [0, 1, 3, 6, 12, 24, 48, 72, 168]
HORIZONTES_DESCRITIVOS_H = [1, 3, 6, 12, 24]
LAG_MAX_CHUVA_Q_H = 48
LAG_MAX_AUTOCORRELACAO_H = 168
BINS_MI = 8                  # mutual information em nats: discretizacao em ate 8 faixas
LIMIAR_PCA_VARIANCIA = 0.95
AMOSTRA_GRAFICO_PCA = 7000
SEMENTE = 42

LIMIAR_CHUVA_EVENTO_MM_H = 0.5
SEPARACAO_SECA_H = 2
CHUVA_MIN_EVENTO_MM = 20
JANELA_Q_APOS_CHUVA_H = 48
JANELA_BASE_H = 12
LIMIAR_ASCENSAO_FRAC = 0.05  # 5% da amplitude acima de Q_base; inicio do T_asc
DURACAO_MAX_EVENTO_SIMPLES_H = 48
AMPLITUDE_MIN_EVENTO = 10.0
EVENTOS_EXEMPLO = ['2021-01-28 20:00:00', '2022-11-26 07:00:00']

CORES = {'p1':'#4386b4', 'p2':'#bd86bf', 'pmed':'#2f796e', 'q':'#147569', 'laranja':'#d68139'}
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'savefig.dpi':155,
                     'axes.spines.top':False, 'axes.spines.right':False})
# True: mostrar figuras ao executar cada celula em um ambiente interativo.
# False: apenas salvar PNG (mais conveniente na execucao integral).
MOSTRAR_FIGURAS = True
SAIDA.mkdir(parents=True, exist_ok=True)
print('Pasta de trabalho:', RAIZ)
print('Arquivo de dados:', ARQUIVO)
print('Pasta dos resultados:', SAIDA)

# %% [markdown]
# # 01 - Funcoes de apoio e carregamento
# Lemos o CSV, verificamos datas duplicadas, lacunas horarias e valores invalidos.
# P_media e media aritmetica dos dois postos; logQ = log(1 + Q) reduz assimetria.
# Funcoes de apoio salvam graficos/CSV e calculam correlacoes e MI discreta.
# Conceitos: Pearson mede linearidade; Spearman mede associacao monotona;
# MI discreta detecta dependencias estatisticas mais gerais, em nats.

# %%
def gravar(df, nome, index=False):
    df.to_csv(SAIDA / nome, index=index, sep=';', decimal=',', encoding='utf-8-sig',
              date_format='%Y-%m-%d %H:%M:%S')


def salvar(fig, nome):
    fig.savefig(SAIDA / nome, bbox_inches='tight', facecolor='white')
    # A figura e gravada sempre, mesmo que a visualizacao esteja desativada.
    # Em VS Code/Spyder/Jupyter, plt.show() exibe o grafico da celula atual.
    if MOSTRAR_FIGURAS and plt.get_backend().lower() != 'agg':
        plt.show()
    plt.close(fig)


def carregar():
    if not ARQUIVO.exists():
        raise FileNotFoundError(f'Nao encontrei {ARQUIVO}. Coloque o CSV ao lado do codigo.')
    x = pd.read_csv(ARQUIVO, sep=';', encoding='utf-8-sig', parse_dates=[COL_DATA])
    faltantes = set(COLS_P + [COL_Q]) - set(x.columns)
    if faltantes:
        raise ValueError(f'Colunas ausentes: {sorted(faltantes)}')
    x = x.sort_values(COL_DATA).set_index(COL_DATA)
    if x.index.has_duplicates:
        raise ValueError('Ha datas duplicadas: verifique o CSV antes de analisar.')
    diferencas = x.index.to_series().diff().dropna()
    if not diferencas.eq(pd.Timedelta(hours=1)).all():
        raise ValueError('A serie nao e completamente horaria. Revisar lacunas temporais.')
    numericas = COLS_P + [COL_Q]
    if x[numericas].isna().any().any():
        raise ValueError('Existem valores ausentes na base; trate antes de executar.')
    if (x[numericas] < 0).any().any():
        raise ValueError('Ha valores negativos de P/Q; revisar a base.')
    # P_media e APENAS uma media aritmetica dos pluviometros.
    # Nao representa necessariamente a chuva media espacial da bacia.
    x['P_media'] = x[COLS_P].mean(axis=1)
    x['logQ'] = np.log1p(x[COL_Q])
    x['dlogQ'] = x['logQ'].diff()
    return x


def correla(a, b, metodo='pearson'):
    a, b = pd.Series(a).align(pd.Series(b), join='inner')
    z = pd.concat([a,b],axis=1).dropna()
    if len(z) < 10 or z.iloc[:, 0].nunique() < 2 or z.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(z.iloc[:,0].corr(z.iloc[:,1],method=metodo))


def classes_mi(serie, bins=BINS_MI):
    """Discretizacao fixa: zero separado de precipitacoes positivas frequentes.
    MI calculada por tabela de contingencia (nats), sem modelo preditivo.
    """
    a = np.asarray(serie, dtype=float)
    if not np.isfinite(a).all():
        raise ValueError('MI requer arrays sem NaN/Inf; aplique mascara de pares validos.')
    if np.unique(a).size < 2:
        return np.zeros(len(a), dtype=int)
    # A classe 'sem chuva' deve ficar separada das classes de chuva positiva.
    # Como MI depende da discretizacao, comparar diferentes BINS_MI e util.
    zeros = (a == 0)
    if (a >= 0).all() and zeros.mean() >= 0.30 and (~zeros).sum() > 10:
        p = a[~zeros]
        cortes = np.unique(np.quantile(p, np.linspace(0,1,bins)[1:-1]))
        codigo = np.zeros(len(a), dtype=int)
        codigo[~zeros] = 1 + np.searchsorted(cortes, p, side='right')
        return codigo
    cortes = np.unique(np.quantile(a, np.linspace(0,1,bins+1)[1:-1]))
    return np.searchsorted(cortes, a, side='right')


def mi_discreta(x, y):
    """Informacao mutua empírica em nats com mesma discretizacao em cada eixo."""
    z = pd.concat([pd.Series(x),pd.Series(y)],axis=1).dropna()
    if len(z) < 100 or z.iloc[:,0].nunique() < 2 or z.iloc[:,1].nunique() < 2:
        return np.nan
    return float(mutual_info_score(classes_mi(z.iloc[:,0]), classes_mi(z.iloc[:,1])))


def heatmap(tabela, titulo, xlab, ylab, filename, divergente=False, anotacoes=True):
    a = tabela.to_numpy(dtype=float)
    fig, ax=plt.subplots(figsize=(max(8, .85*len(tabela.columns)+3), max(5, .38*len(tabela)+2.2)))
    if divergente:
        vmax = np.nanmax(np.abs(a)); vmax = max(vmax, .01)
        im=ax.imshow(a,cmap='BrBG',vmin=-vmax,vmax=vmax,aspect='auto')
    else:
        im=ax.imshow(a,cmap='YlGnBu',vmin=0,aspect='auto')
    ax.set_xticks(np.arange(len(tabela.columns)), labels=tabela.columns.astype(str))
    ax.set_yticks(np.arange(len(tabela.index)), labels=tabela.index.astype(str))
    ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_title(titulo,loc='left',pad=15,fontweight='bold')
    if anotacoes:
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                if np.isfinite(a[i,j]):
                    ax.text(j,i,f'{a[i,j]:.2f}',ha='center',va='center',fontsize=8)
    fig.colorbar(im,ax=ax,shrink=.85,label='Pearson/Spearman (r)' if divergente else 'Informação mútua (nats)')
    salvar(fig,filename)


x = carregar()
print(f'Base carregada: {len(x):,} horas, de {x.index.min()} a {x.index.max()}')
print('Colunas analisadas:', COLS_P + [COL_Q])
print(x[COLS_P + [COL_Q]].head().to_string())

# %% [markdown]
# # 02 - Estatisticas descritivas, distribuicoes e climatologia
# Perguntas: ha muitos zeros? As variaveis sao assimetricas? Ha sazonalidade?
# Examinamos quantis e climatologia dos anos calendarios completos, evitando
# comparar anos parciais como se fossem anos inteiros.

# %%
def ics_data(x):
    """Resumo diagnostico e climatologia somente dos anos calendarios completos."""
    cols = COLS_P + [COL_Q]
    resumo = x[cols].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).T
    resumo['proporcao_zero']=(x[cols] == 0).mean()
    resumo['assimetria']=x[cols].skew()
    gravar(resumo.reset_index(names='variavel'),'01_estatisticas_descritivas.csv')
    por_ano=x.groupby(x.index.year).agg(horas=(COL_Q,'size'),P1=(COLS_P[0],'sum'),
                                            P2=(COLS_P[1],'sum'),Qmedia=(COL_Q,'mean'),
                                            Qmediana=(COL_Q,'median'),Qp95=(COL_Q,lambda s:s.quantile(.95)))
    gravar(por_ano.reset_index(names='ano'),'02_estatisticas_anuais.csv')
    completos=[ano for ano, grupo in x.groupby(x.index.year)
               if len(grupo) in (8760, 8784) and grupo.index.min()==pd.Timestamp(f'{ano}-01-01')
               and grupo.index.max()==pd.Timestamp(f'{ano}-12-31 23:00')]
    c = x.loc[x.index.year.isin(completos)].copy()
    mensais=[]
    for (ano,mes),g in c.groupby([c.index.year,c.index.month]):
        mensais.append({'ano':ano,'mes':mes,'P1_mm':g[COLS_P[0]].sum(),
                        'P2_mm':g[COLS_P[1]].sum(),'P_media_mm':g['P_media'].sum(),
                        'Qmediana':g[COL_Q].median(),'Qp95':g[COL_Q].quantile(.95),
                        'Qmax':g[COL_Q].max()})
    mensal=pd.DataFrame(mensais)
    gravar(mensal,'03_precipitacao_vazao_mensal_anos_completos.csv')
    clim = mensal.groupby('mes').agg(P1=('P1_mm','mean'),P2=('P2_mm','mean'),
                                     Qmediana=('Qmediana','median'),Qp95=('Qp95','mean')).reset_index()
    gravar(clim,'04_climatologia_mensal.csv')
    fig, ax=plt.subplots(figsize=(11.4,5.0))
    ax.bar(clim.mes-.18,clim.P1,width=.36,color=CORES['p1'],label='Precipitação 2463')
    ax.bar(clim.mes+.18,clim.P2,width=.36,color=CORES['p2'],label='Precipitação 84100000')
    ax.set(xlabel='Mês', ylabel='Precipitação média mensal (mm)',xticks=range(1,13))
    ax2=ax.twinx();ax2.plot(clim.mes,clim.Qmediana,'-o',color=CORES['q'],label='Mediana mensal Q')
    ax2.plot(clim.mes,clim.Qp95,'--o',color=CORES['laranja'],label='Média do P95 mensal Q')
    ax2.set_ylabel('Vazão (m³/s)')
    l1,v1=ax.get_legend_handles_labels(); l2,v2=ax2.get_legend_handles_labels()
    fig.legend(l1+l2,v1+v2,loc='upper center',bbox_to_anchor=(.5,1.04),ncol=4,frameon=False)
    ax.set_title(f'Climatologia mensal ({min(completos)}–{max(completos)}, anos completos)',pad=31)
    salvar(fig,'01_climatologia.png')
    # Distribuicao de Q em escalas natural e logaritmica; P em horas chuvosas.
    fig,axs=plt.subplots(1,3,figsize=(13.2,3.9))
    axs[0].hist(x[COL_Q],bins=85,color=CORES['q']);axs[0].set(xlabel='Q',ylabel='Horas',title='Distribuição de Q')
    axs[1].hist(np.log1p(x[COL_Q]),bins=85,color=CORES['q']);axs[1].set(xlabel='log(1+Q)',title='Distribuição em escala log')
    for p,cor in zip(COLS_P,[CORES['p1'],CORES['p2']]):
        chuva=x.loc[x[p]>0,p]
        axs[2].hist(chuva,bins=45,alpha=.55,label=p,color=cor)
    axs[2].set(xlabel='Chuva horária (>0)',ylabel='Horas',title='Intensidades de chuva');axs[2].legend(fontsize=8)
    salvar(fig,'02_distribuicoes.png')
    return resumo,completos


resumo, completos = ics_data(x)
print('Anos completos incluidos na climatologia:', completos)
print(resumo[['mean','50%','95%','99%','max','proporcao_zero']].round(3).to_string())

# %% [markdown]
# # 03 - Concordancia entre os pluviometros
# Comparamos chuva horaria e chuva acumulada diaria entre os dois postos.
# Correlacao diaria alta com menor concordancia horaria pode indicar
# variabilidade espacial da chuva relevante para a resposta da bacia.

# %%
def comparar_postos(x):
    raw = x[COLS_P].corr(method='pearson').iloc[0,1]
    diario=x[COLS_P].resample('D').sum()
    daycorr=diario.corr().iloc[0,1]
    wet=((x[COLS_P]>0).any(axis=1))
    discord=(x[COLS_P[0]].gt(0) != x[COLS_P[1]].gt(0)).sum()
    comparacao=pd.DataFrame([{'r_pearson_horario':raw,'r_pearson_diario':daycorr,
                               'horas_chuva_algum_posto':int(wet.sum()),
                               'frac_horas_so_um_posto_chove':float(discord/len(x)),
                               'frac_entre_horas_chuvosas_com_discordancia':float(discord/max(wet.sum(),1))}])
    gravar(comparacao,'05_comparacao_postos.csv')
    z=x.loc[wet,COLS_P]
    fig,ax=plt.subplots(figsize=(7,5.5))
    hist=ax.hist2d(np.log1p(z.iloc[:,0]), np.log1p(z.iloc[:,1]),bins=55,cmin=1,cmap='viridis')
    t=np.log1p(z.max().max());ax.plot([0,t],[0,t],linestyle='--',color='white',lw=1.5)
    ax.set(xlabel=f'log(1 + {COLS_P[0]})',ylabel=f'log(1 + {COLS_P[1]})',
           title='Concordância dos postos nas horas com chuva')
    fig.colorbar(hist[3],ax=ax,label='Número de horas')
    salvar(fig,'03_concordancia_postos.png')
    return comparacao


comparacao = comparar_postos(x)
print(comparacao.round(3).to_string(index=False))

# %% [markdown]
# # 04 - Preparacao de descritores temporais (sem ML)
# Calculamos acumulados de P, defasagens de Q e variacoes da vazao.
# Esses descritores sao usados somente para diagnostico exploratorio.
# Todos terminam na hora t: o objetivo nao e treinar modelos.

# %%
def montar_atributos(x):
    """Todas as entradas usam exclusivamente dados ate t (inclusive)."""
    d = pd.DataFrame(index=x.index)
    # Acumulados em diferentes janelas: chuva recente e memoria mais longa.
    # A soma movel termina em t; nenhuma chuva futura entra aqui.
    for p in COLS_P + ['P_media']:
        for w in JANELAS_CHUVA_H:
            d[f'{p}_acum_{w}h'] = x[p].rolling(w,min_periods=w).sum()
    for l in LAGS_Q_H:
        d[f'Q_lag_{l}h'] = x[COL_Q].shift(l)
    for w in [6,24,72]:
        d[f'Q_media_{w}h'] = x[COL_Q].rolling(w,min_periods=w).mean()
    d['dlogQ_1h'] = x['logQ'].diff()
    d['dlogQ_3h'] = x['logQ'] - x['logQ'].shift(3)
    d['dlogQ_6h'] = x['logQ'] - x['logQ'].shift(6)
    d['seno_dia_ano'] = np.sin(2*np.pi*x.index.dayofyear/365.25)
    d['cosseno_dia_ano'] = np.cos(2*np.pi*x.index.dayofyear/365.25)
    d['mes'] = x.index.month
    return d


atributos = montar_atributos(x)
print('Descritores criados:', atributos.shape[1])
print('Exemplo de colunas:', ', '.join(atributos.columns[:10]))
print(atributos.iloc[170:173, :5].round(2).to_string())

# %% [markdown]
# # 05 - Correlacao temporal e informacao mutua
# Avaliamos P(t) associada a dlogQ(t+tau), com tau de 1 ate 48 horas.
# Pico da correlacao nao equivale necessariamente a tempo de concentracao.
# MI nao mede causalidade e seu valor depende de como as series sao discretizadas.

# %%
def lags_corr_mi(x):
    """Compara P(t) com dlogQ(t+tau), e logQ(t+tau). lag>=1 para resposta futura."""
    rows=[]
    for fonte in COLS_P + ['P_media']:
        p=x[fonte]
        for lag in range(1,LAG_MAX_CHUVA_Q_H+1):
            # dlogQ(t+lag): taxa de variacao horaria na resposta posterior.
            # lag positivo indica que P antecede a resposta em Q.
            target_delta=x['dlogQ'].shift(-lag)
            target_level=x['logQ'].shift(-lag)
            mask=p.notna() & target_delta.notna()
            rows.append({'posto':fonte,'defasagem_h':lag,
                         'pearson_P_dlogQ':correla(p[mask],target_delta[mask]),
                         'spearman_P_dlogQ':correla(p[mask],target_delta[mask],'spearman'),
                         'MI_P_dlogQ_nats':mi_discreta(p[mask],target_delta[mask]),
                         'pearson_P_logQ':correla(p[mask],target_level[mask]),
                         'spearman_P_logQ':correla(p[mask],target_level[mask],'spearman'),
                         'MI_P_logQ_nats':mi_discreta(p[mask],target_level[mask])})
    lagdf=pd.DataFrame(rows)
    gravar(lagdf,'06_correlacao_MI_defasagens.csv')
    fig,axs=plt.subplots(1,3,figsize=(15,4.4),sharex=True)
    for fonte, cor in zip(COLS_P+['P_media'],[CORES['p1'],CORES['p2'],CORES['pmed']]):
        z=lagdf.loc[lagdf.posto==fonte]
        for ax,campo in zip(axs,['pearson_P_dlogQ','spearman_P_dlogQ','MI_P_dlogQ_nats']):
            ax.plot(z.defasagem_h,z[campo],label=fonte,color=cor,lw=1.9)
    for ax,titulo in zip(axs,['Pearson','Spearman','Informação mútua']):
        ax.set(xlabel='Defasagem P(t) → ΔlogQ(t+τ) (h)',title=titulo)
        ax.axhline(0,color='grey',lw=.6)
    axs[0].set_ylabel('Correlação r'); axs[2].set_ylabel('MI discreta (nats)')
    axs[1].legend(loc='upper right',fontsize=8)
    fig.suptitle('Associação temporal chuva–resposta da vazão',fontsize=14)
    salvar(fig,'04_correlacao_informacao_mutua_defasagem.png')
    return lagdf


lag = lags_corr_mi(x)
for posto, grupo in lag.groupby('posto'):
    r = grupo.loc[grupo['pearson_P_dlogQ'].idxmax()]
    mi = grupo.loc[grupo['MI_P_dlogQ_nats'].idxmax()]
    print(f'{posto}: Pearson max = {r.pearson_P_dlogQ:.3f} (tau={int(r.defasagem_h)} h); '
          f'MI max = {mi.MI_P_dlogQ_nats:.4f} nats (tau={int(mi.defasagem_h)} h)')

# %% [markdown]
# # 06 - Autocorrelacao e memoria hidrologica
# A autocorrelacao compara uma serie com ela propria em diferentes defasagens.
# Distinguimos a memoria do nivel log(1+Q), de sua variacao e da chuva.
# Autocorrelacao elevada nao prova influencia fisica ou previsibilidade.

# %%
def autocorrelacoes(x):
    chuva=x['P_media']
    var=x['dlogQ'].dropna()
    rows=[]
    for h in range(0,LAG_MAX_AUTOCORRELACAO_H+1):
        rows.append({'defasagem_h':h,'ACF_logQ':x['logQ'].autocorr(lag=h),
                     'ACF_dlogQ':var.autocorr(lag=h), 'ACF_Pmedia':chuva.autocorr(lag=h)})
    ac=pd.DataFrame(rows)
    gravar(ac,'07_autocorrelacao.csv')
    fig,ax=plt.subplots(figsize=(10,4.3))
    for var, c, label in [('ACF_logQ',CORES['q'],'log(1+Q)'),('ACF_dlogQ',CORES['laranja'],'Δlog(1+Q)'),
                           ('ACF_Pmedia',CORES['p1'],'Precipitação média')]:
        ax.plot(ac.defasagem_h,ac[var],label=label,color=c)
    ax.axhline(0,color='grey',lw=.5);ax.set(xlabel='Defasagem (horas)',ylabel='Autocorrelação',
                                      title='Memória temporal das séries')
    ax.legend();salvar(fig,'05_memoria_hidrologica.png')
    return ac


acf = autocorrelacoes(x)
print(acf.loc[acf.defasagem_h.isin([1,6,24,48,72,168])].round(3).to_string(index=False))

# %% [markdown]
# # 07 - Chuva antecedente x variacao futura de Q
# Exploramos Pearson, Spearman e MI dos acumulados antecedentes versus
# logQ(t+h)-logQ(t), apenas para caracterizar associacoes estatisticas.
# O sinal negativo de uma correlacao nao implica que chuva reduza vazao:
# frequentemente reflete a posicao na subida ou recessao do hidrograma.

# %%
def assoc_atributos(x, atributos):
    """Associacao DESCRITIVA entre variaveis conhecidas em t e ΔlogQ(t→t+h).
       Nao calcula metrica de previsao, nao treina modelos e nao faz validacao.
    """
    selecionadas=[v for v in atributos.columns if v != 'mes']
    saidas=[]
    for h in HORIZONTES_DESCRITIVOS_H:
        y=x['logQ'].shift(-h)-x['logQ']
        mask_base=y.notna()
        for v in selecionadas:
            xv=atributos[v]
            m=mask_base & xv.notna()
            a=xv[m]; b=y[m]
            saidas.append({'horizonte_descritivo_h':h,'variavel_conhecida_em_t':v,
                           'pearson_com_dlogQ_futuro':correla(a,b),
                           'spearman_com_dlogQ_futuro':correla(a,b,'spearman'),
                           'MI_com_dlogQ_futuro_nats':mi_discreta(a,b),
                           'n_pares':int(m.sum())})
    assoc=pd.DataFrame(saidas)
    gravar(assoc,'08_atributos_corr_MI_horizontes.csv')
    chuva=assoc.loc[assoc.variavel_conhecida_em_t.str.startswith('P_media_acum_')].copy()
    chuva['janela_h']=chuva.variavel_conhecida_em_t.str.extract(r'acum_(\d+)h')[0].astype(int)
    for coluna,arq,titulo,div in [
        ('spearman_com_dlogQ_futuro','06_spearman_chuva_antecedente.png','Spearman: chuva antecedente × mudança futura de Q',True),
        ('MI_com_dlogQ_futuro_nats','07_mi_chuva_antecedente.png','Informação mútua: chuva antecedente × mudança futura de Q',False)]:
        mat=chuva.pivot(index='janela_h',columns='horizonte_descritivo_h',values=coluna)
        mat=mat.reindex(JANELAS_CHUVA_H)[HORIZONTES_DESCRITIVOS_H]
        heatmap(mat,titulo,'Horizonte de resposta (h)','Janela de chuva anterior (h)',arq,divergente=div)
    # Lista de associações por horizonte (sem alegar importância causal/preditiva).
    topo=assoc.sort_values(['horizonte_descritivo_h','MI_com_dlogQ_futuro_nats'],ascending=[True,False])
    gravar(topo,'09_atributos_ordenados_por_MI.csv')
    return assoc


associacoes = assoc_atributos(x, atributos)
print('Numero de combinacoes atributo-horizonte:', len(associacoes))
print('Exemplo de associacoes (h=6):')
print(associacoes.loc[associacoes.horizonte_descritivo_h.eq(6),
      ['variavel_conhecida_em_t','spearman_com_dlogQ_futuro','MI_com_dlogQ_futuro_nats']]
      .sort_values('MI_com_dlogQ_futuro_nats', ascending=False).head(8).round(4).to_string(index=False))

# %% [markdown]
# # 08 - Redundancia: matriz de Spearman entre descritores
# Descritores de janelas muito proximas tendem a carregar informacao repetida.
# A matriz de Spearman destaca redundancia antes de calcular componentes PCA.
# Nao se trata de selecao supervisionada de atributos.

# %%
def correlacao_entre_atributos(atributos):
    """Diagnostica redundancia entre entradas antecedentes, antes da PCA."""
    cols=([f'{p}_acum_{w}h' for p in COLS_P for w in [1,6,24,72,168]]
           + [f'Q_lag_{l}h' for l in [0,3,12,24,72,168]])
    corr=atributos[cols].corr(method='spearman')
    gravar(corr.reset_index(names='atributo'),'15_correlacao_entre_atributos_spearman.csv')
    fig,ax=plt.subplots(figsize=(10.4,8.4))
    im=ax.imshow(corr.to_numpy(),vmin=-1,vmax=1,cmap='BrBG',aspect='equal')
    ax.set_xticks(range(len(cols)),cols,rotation=75,ha='right',fontsize=8)
    ax.set_yticks(range(len(cols)),cols,fontsize=8)
    fig.colorbar(im,ax=ax,fraction=.045,pad=.03,label='Spearman (r)')
    ax.set_title('Dependência e redundância entre entradas antecedentes',loc='left',pad=15)
    salvar(fig,'13_matriz_correlacao_atributos.png')
    return corr


corr_atributos = correlacao_entre_atributos(atributos)
print('Matriz de Spearman:', corr_atributos.shape)

# %% [markdown]
# # 09 - PCA (apenas P; P + Q antecedente)
# PCA procura direcoes que explicam a variancia dos descritores, sem variavel alvo.
# Etapas: log(1+x) -> padronizacao -> autovetores/componentes -> loadings.
# PC1 e PC2 nao precisam somar 95% da variancia; observamos quantas PCs
# sao necessarias para atingir 95% de variancia ACUMULADA.
# PCA nao determina relevancia para futuras predicoes.

# %%
def analise_pca(x, atributos):
    """PCA em duas familias: apenas chuva; chuva e estado antecedente Q.
       Para Q assimetrica e chuva esparsa, usar log1p antes de padronizar.
       Sem colunas y futuro; sem modelo supervisionado.
    """
    chuva=[f'{p}_acum_{w}h' for p in COLS_P for w in JANELAS_CHUVA_H]
    baseq=[f'Q_lag_{l}h' for l in LAGS_Q_H] + ['Q_media_24h','Q_media_72h']
    esquemas={'apenas_chuva':chuva, 'chuva_e_vazao_antecedente':chuva+baseq}
    outputs=[]
    for nome,colunas in esquemas.items():
        df=atributos[colunas].dropna()
        # log(1+x) reduz a assimetria e comporta zeros (muito comuns em P).
        arr=np.log1p(df.to_numpy(dtype=float))
        # Padronizacao: cada atributo passa a ter media zero e desvio um.
        # Sem isso, variaveis de grande escala dominariam as componentes.
        z=StandardScaler().fit_transform(arr)
        pca=PCA(svd_solver='full').fit(z)
        score=pca.transform(z)
        var=pca.explained_variance_ratio_
        acum=var.cumsum()
        # Menor numero de PCs que conserva ao menos 95% da variancia.
        # ATENCAO: variancia explicada nao e equivalente a utilidade preditiva.
        n95=int(np.searchsorted(acum,LIMIAR_PCA_VARIANCIA,side='left')+1)
        load=pd.DataFrame(pca.components_.T * np.sqrt(pca.explained_variance_),
                          index=colunas,columns=[f'PC{i+1}' for i in range(len(colunas))])
        # Os loadings aqui sao correlacoes de X padronizada com as PCs.
        gravar(load.reset_index(names='atributo'),f'11_pca_loadings_{nome}.csv')
        var_df=pd.DataFrame({'componente':np.arange(1,len(var)+1),
                             'variancia_explicada':var,'variancia_acumulada':acum})
        gravar(var_df,f'10_pca_variancia_{nome}.csv')
        outputs.append({'conjunto':nome,'n_variaveis':len(colunas),'n_linhas':len(df),
                        'PCs_para_95pct':n95,'PC1_pct':100*var[0],
                        'PC1_PC2_pct':100*acum[min(1,len(acum)-1)]})
        fig,ax=plt.subplots(figsize=(9.5,4.5))
        ax.bar(var_df.componente,var_df.variancia_explicada*100,color=CORES['p1'],alpha=.62,label='Individual')
        ax2=ax.twinx()
        ax2.plot(var_df.componente,var_df.variancia_acumulada*100,'-o',ms=3,color=CORES['q'],label='Acumulada')
        ax2.axhline(95,ls='--',color=CORES['laranja'],label='95%')
        ax2.set_ylim(0,105);ax2.set_ylabel('Variância acumulada (%)')
        ax.set(xlabel='Número da componente principal',ylabel='Variância individual (%)',
               title=f'PCA — {nome.replace("_"," ")} | {n95} PCs para 95%')
        salvar(fig,f'08_pca_variancia_{nome}.png')
        fig,ax=plt.subplots(figsize=(9.7,6.0))
        top=load[['PC1','PC2']].pow(2).sum(axis=1).nlargest(18).index
        a=load.loc[top,['PC1','PC2']]
        y=np.arange(len(a))
        ax.barh(y+.17,a['PC1'],height=.34,color=CORES['p1'],label='PC1')
        ax.barh(y-.17,a['PC2'],height=.34,color=CORES['laranja'],label='PC2')
        ax.set(yticks=y,yticklabels=a.index,xlabel='Loading (correlação com componente)',
               title=f'Cargas das componentes — {nome.replace("_"," ")}')
        ax.invert_yaxis(); ax.axvline(0,color='grey',lw=.6); ax.legend();salvar(fig,f'09_pca_loadings_{nome}.png')
        rng=np.random.default_rng(SEMENTE)
        chosen=np.sort(rng.choice(len(df),size=min(AMOSTRA_GRAFICO_PCA,len(df)),replace=False))
        fig,ax=plt.subplots(figsize=(8,6))
        q_cor=np.log1p(x.loc[df.index[chosen],COL_Q].to_numpy())
        dots=ax.scatter(score[chosen,0],score[chosen,1],c=q_cor,s=7,alpha=.42,cmap='viridis',rasterized=True)
        ax.set(xlabel=f'PC1 ({100*var[0]:.1f}%)',ylabel=f'PC2 ({100*var[1]:.1f}%)',
               title=f'Projeção PCA — {nome.replace("_"," ")} (amostra de horas)')
        fig.colorbar(dots,ax=ax,label='log(1+Q)')
        salvar(fig,f'10_pca_scores_{nome}.png')
    resumo=pd.DataFrame(outputs)
    gravar(resumo,'12_resumo_pca.csv')
    return resumo


pca = analise_pca(x, atributos)
print(pca.round(2).to_string(index=False))

# %% [markdown]
# # 10 - Eventos hidrologicos: T_CMp-Qp, T_CMp-CMq e T_asc
# Nesta secao, usamos EXATAMENTE os tres tempos da figura de referencia:
#
# 1. TEMPO DE RETARDO AO PICO (CM_P -> Q_p):
#    T_CMp_Qp = t(Q_p) - t(CM_P)
#    Diferenca entre o centro de massa da chuva e o instante do pico de Q.
#
# 2. DEFASAGEM ENTRE CENTROIDES (CM_P -> CM_Q):
#    T_CMp_CMq = t(CM_Q) - t(CM_P)
#    O CM_Q corresponde somente ao ESCOAMENTO EXCEDENTE, e nao ao Q total.
#
# 3. TEMPO DE ASCENSÃO (inicio -> Q_p):
#    T_asc = t(Q_p) - t(inicio_ascensao)
#    Inicio operacional = PRIMEIRA passagem de Q pelo limiar:
#        Q_limiar = Q_base + 0.05*(Q_p - Q_base)
#    O valor 0.05 e configuravel em LIMIAR_ASCENSAO_FRAC.
#
# Observacoes importantes:
# - Usamos horas reais dos timestamps; nao presumimos indices de tempo.
# - Centroide e calculado por soma(t_i * peso_i)/soma(peso_i).
# - Q_base e a mediana das 12 horas antecedentes (aproximacao didatica).
# - Se a ascensao ja passou pelo limiar antes da janela observada, T_asc
#   permanece ausente (NaN) em vez de receber uma estimativa enganosa.
# - Em eventos compostos, as metricas dependem da delimitacao das janelas.
# - Estes indicadores NAO sao transito de cheia entre secoes fluviais.

# %%
def centro_massa(indice, pesos):
    """Centro de massa temporal de uma grandeza amostrada a cada hora.

    Usamos o primeiro timestamp como origem numerica para evitar problemas
    de precisao ao multiplicar timestamps muito grandes pelos pesos.
    """
    arr = np.asarray(pesos, dtype=float)
    if len(indice) != len(arr):
        raise ValueError('Pesos e datas precisam ter o mesmo comprimento.')
    if not len(arr) or np.nansum(arr) <= 0:
        return pd.NaT
    horas = np.asarray((indice - indice[0]).total_seconds() / 3600, dtype=float)
    centro_h = float(np.nansum(horas * arr) / np.nansum(arr))
    return indice[0] + pd.Timedelta(hours=centro_h)


def diferenca_h(instante_final, instante_inicial):
    """Retorna um intervalo de tempo em horas (pode ser negativo)."""
    if pd.isna(instante_final) or pd.isna(instante_inicial):
        return np.nan
    return float((instante_final - instante_inicial).total_seconds() / 3600)


def detectar_eventos(x):
    """Separa blocos de chuva por um numero minimo de horas sem chuva."""
    chuva = x['P_media'].to_numpy()
    instantes_ativos = np.flatnonzero(chuva > LIMIAR_CHUVA_EVENTO_MM_H)
    if len(instantes_ativos) == 0:
        return []
    cortes = np.flatnonzero(np.diff(instantes_ativos) > SEPARACAO_SECA_H + 1) + 1
    return [(int(grupo[0]), int(grupo[-1]))
            for grupo in np.split(instantes_ativos, cortes)]


def obter_inicio_ascensao(hid_ate_pico, q_base, q_pico):
    """Encontra a primeira passagem observada por 5% da amplitude do evento.

    Nao confundir Q_limiar com 5% de Q_p: usamos 5% da DIFERENCA
    Q_p - Q_base, somada a Q_base.

    Se a primeira vazao observada ja excede o limiar, nao sabemos quando
    a ascensao teve inicio; nesse caso retorna NaT e uma justificativa.
    """
    amplitude = q_pico - q_base
    if amplitude <= 0 or len(hid_ate_pico) < 2:
        return pd.NaT, np.nan, 'Sem amplitude positiva ou sem observacoes'
    limiar = q_base + LIMIAR_ASCENSAO_FRAC * amplitude
    q = hid_ate_pico.astype(float)
    if q.iloc[0] >= limiar:
        return pd.NaT, limiar, 'Vazao acima do limiar no inicio da janela'
    cruzamentos = (q >= limiar) & (q.shift(1) < limiar)
    instantes = q.index[cruzamentos.fillna(False)]
    if len(instantes) == 0:
        return pd.NaT, limiar, 'Limiar nao cruzado na janela observada'
    return instantes[0], limiar, 'Cruzamento identificado'


def metricas_evento(x, grupos, n):
    """Calcula as tres metricas temporais para o evento numero n."""
    ini, fim = grupos[n]
    if ini < JANELA_BASE_H or fim >= len(x) - 1:
        return None
    chuva_evento = x.iloc[ini:fim + 1]
    if chuva_evento['P_media'].sum() < CHUVA_MIN_EVENTO_MM:
        return None

    # Q_base nao e hidrograma de base separado por metodo fisico.
    # E apenas uma referencia anterior ao evento, declarada no CSV.
    q_base = float(x[COL_Q].iloc[ini - JANELA_BASE_H:ini].median())

    # A janela do hidrograma termina apos a chuva, ou antes do proximo
    # bloco de precipitacao. Assim, reduzimos contaminacao por outro evento.
    qfim = min(len(x) - 1, fim + JANELA_Q_APOS_CHUVA_H)
    if n + 1 < len(grupos):
        qfim = min(qfim, grupos[n + 1][0] - 1)
    if qfim < ini + 1:
        return None
    hid = x.iloc[ini:qfim + 1]

    # Pico do hidrograma e instante do pico (Q_p).
    q_pico = float(hid[COL_Q].max())
    t_qp = hid[COL_Q].idxmax()
    amplitude = q_pico - q_base

    # CM_P = centro de massa da chuva media dos dois pluviometros.
    t_cmp = centro_massa(chuva_evento.index, chuva_evento['P_media'].to_numpy())

    # CM_Q = centro de massa do ESCOAMENTO EXCEDENTE.
    # Uma vazao base muito grande distorceria o centroide do Q total.
    q_excedente = (hid[COL_Q] - q_base).clip(lower=0)
    t_cmq = centro_massa(hid.index, q_excedente.to_numpy())

    duracao = len(chuva_evento)
    evento_curto = duracao <= DURACAO_MAX_EVENTO_SIMPLES_H
    resposta_bem_definida = bool(
        evento_curto and amplitude >= AMPLITUDE_MIN_EVENTO and
        q_pico >= 1.5 * max(q_base, .5) and t_qp > chuva_evento.index[0]
    )

    # Para eventos sem resposta definida, NAO declaramos um T_asc.
    # T_CMp_Qp e T_CMp_CMq ainda podem ser calculados, mas merecem cautela.
    t_inicio_ascensao = pd.NaT
    q_limiar_ascensao = np.nan
    observacao_ascensao = 'Resposta nao classificada como bem definida'
    if resposta_bem_definida:
        t_inicio_ascensao, q_limiar_ascensao, observacao_ascensao = (
            obter_inicio_ascensao(hid.loc[:t_qp, COL_Q], q_base, q_pico)
        )

    # === AS TRES METRICAS SOLICITADAS (em horas) ===
    # 1. Tempo de retardo ao pico (CM_P -> Q_p)
    T_CMp_Qp = diferenca_h(t_qp, t_cmp)
    # 2. Defasagem entre centroides (CM_P -> CM_Q)
    T_CMp_CMq = diferenca_h(t_cmq, t_cmp)
    # 3. Tempo de ascensão (inicio -> Q_p)
    T_asc = diferenca_h(t_qp, t_inicio_ascensao)

    return {
        'inicio_chuva': chuva_evento.index[0],
        'fim_chuva': chuva_evento.index[-1],
        'fim_janela_resposta': hid.index[-1],
        'duracao_chuva_h': duracao,
        'P2463_mm': float(chuva_evento[COLS_P[0]].sum()),
        'P84100000_mm': float(chuva_evento[COLS_P[1]].sum()),
        'Pmedia_acumulada_mm': float(chuva_evento['P_media'].sum()),
        'Pmax_media_mm_h': float(chuva_evento['P_media'].max()),
        'Pmedia_72h_anteriores_mm': float(x['P_media'].iloc[max(0, ini-72):ini].sum()),
        'Q_base_ref': q_base,
        'Q_pico': q_pico,
        'aumento_Q': amplitude,
        't_Qp': t_qp,
        't_CMp': t_cmp,
        't_CMq': t_cmq,
        't_inicio_ascensao': t_inicio_ascensao,
        'Q_limiar_ascensao': q_limiar_ascensao,
        'T_CMp_Qp_h': T_CMp_Qp,
        'T_CMp_CMq_h': T_CMp_CMq,
        'T_asc_h': T_asc,
        'observacao_ascensao': observacao_ascensao,
        'evento_curto': evento_curto,
        'resposta_bem_definida': resposta_bem_definida,
        '_ini': ini, '_fim': fim, '_qfim': qfim
    }


def texto_tempo(valor):
    """Formata intervalos para anotacoes (inclusive quando indefinidos)."""
    return f'{valor:.2f} h' if pd.notna(valor) else 'Nao definido'


def grafico_evento(x, e, nome):
    """Mostra ietograma e hidrograma com os quatro instantes caracteristicos.

    As linhas verticais representam t(CM_P), t(CM_Q), t(Q_p) e o inicio
    da ascensao; as tres diferencas entre instantes estao escritas acima.
    """
    ini, qfim = int(e['_ini']), int(e['_qfim'])
    bloco = x.iloc[max(0, ini-8):qfim + 1]
    fig, ax = plt.subplots(figsize=(13.5, 6.8))
    fig.subplots_adjust(bottom=.24, top=.74, left=.09, right=.91)

    # Hidrograma (curva de Q).
    ax.plot(bloco.index, bloco[COL_Q], color=CORES['q'], lw=2.2,
            label='Hidrograma de vazao')
    ax.set_ylim(0, max(float(bloco[COL_Q].max()) * 1.25, 1))
    ax.set_ylabel('Vazão (m³/s)', color=CORES['q'])
    ax.grid(axis='y', alpha=.18)

    # Ietogramas invertidos nos dois postos, no mesmo eixo temporal.
    ap = ax.twinx()
    for desloc, p, cor in [(-12, COLS_P[0], CORES['p1']),
                           (12, COLS_P[1], CORES['p2'])]:
        ap.bar(bloco.index + pd.Timedelta(minutes=desloc), bloco[p],
               width=.017, alpha=.65, color=cor, label=f'Ietograma {p}')
    ap.set_ylim(max(1, float(bloco[COLS_P].max().max())) * 2.7, 0)
    ap.set_ylabel('Precipitação horária (mm, eixo invertido)')

    # Identificacao dos instantes empregados nos tres calculos.
    referencias = [
        ('t_CMp', CORES['laranja'], r'CM$_P$ - chuva', '--'),
        ('t_CMq', '#7956a9', r'CM$_Q$ - escoamento excedente', '--'),
        ('t_Qp', '#bc4260', r'Q$_p$ - pico de vazao', ':'),
        ('t_inicio_ascensao', '#505050', 'Início da ascensão (5%)', '-.')
    ]
    for coluna, cor, rotulo, estilo in referencias:
        if pd.notna(e[coluna]):
            ax.axvline(e[coluna], c=cor, lw=1.6, ls=estilo, label=rotulo)
    if pd.notna(e['t_inicio_ascensao']):
        ax.scatter([e['t_inicio_ascensao']],
                   [x.loc[e['t_inicio_ascensao'], COL_Q]],
                   color='#505050', s=38, zorder=10)

    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=11))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    status = 'resposta bem definida' if e['resposta_bem_definida'] else 'resposta composta/ambigua'
    fig.suptitle(f'Ietograma e hidrograma - {e["inicio_chuva"]:%d/%m/%Y} ({status})',
                 y=.965, fontsize=13, fontweight='bold')
    fig.text(.09, .906, f'Chuva acumulada = {e["Pmedia_acumulada_mm"]:.1f} mm   |   '
                         f'Pico Qp = {e["Q_pico"]:.1f} (m³/s)', fontsize=9.5)
    fig.text(.09, .86,
             '1. Tempo de retardo ao pico  '
             r'$T_{CM_P\rightarrow Q_p}$' + f' = {texto_tempo(e["T_CMp_Qp_h"])}',
             color='#b75b22', fontsize=10)
    fig.text(.09, .824,
             '2. Defasagem entre centroides  '
             r'$T_{CM_P\rightarrow CM_Q}$' + f' = {texto_tempo(e["T_CMp_CMq_h"])}',
             color='#7956a9', fontsize=10)
    fig.text(.09, .788,
             '3. Tempo de ascensão  '
             r'$T_{asc}$' + f' = {texto_tempo(e["T_asc_h"])}',
             color='#32643c', fontsize=10)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ap.get_legend_handles_labels()
    fig.legend(h2 + h1, l2 + l1, loc='lower center', bbox_to_anchor=(.5, .06),
               ncol=3, fontsize=8.5, frameon=False)
    salvar(fig, nome)


def analise_eventos(x):
    """Produz catalogo CSV, figura resumo e graficos dos eventos exemplo."""
    grupos = detectar_eventos(x)
    lista_eventos = [e for n in range(len(grupos))
                     if (e := metricas_evento(x, grupos, n)) is not None]
    if not lista_eventos:
        warnings.warn('Nao foram encontrados eventos com os filtros atuais.')
        return pd.DataFrame()
    eventos = pd.DataFrame(lista_eventos)
    # O catalogo COMPLETO preserva inclusive tempos negativos/indefinidos.
    gravar(eventos.drop(columns=[c for c in eventos if c.startswith('_')]),
           '13_catalogo_eventos.csv')

    # Filtramos apenas para a figura comparativa; o CSV fica completo.
    selecao = eventos[
        eventos['resposta_bem_definida'] & eventos['T_CMp_CMq_h'].between(0, 48)
    ]
    fig, axs = plt.subplots(1, 3, figsize=(15.3, 4.2))
    axs[0].scatter(selecao['Pmedia_acumulada_mm'], selecao['T_CMp_Qp_h'],
                   c=selecao['Q_base_ref'], s=24, cmap='viridis', alpha=.7)
    axs[0].set(xlabel='P acumulada do evento (mm)',
               ylabel=r'$T_{CM_P\rightarrow Q_p}$ (h)',
               title='1. Tempo de retardo ao pico')
    axs[1].scatter(selecao['Pmedia_72h_anteriores_mm'], selecao['T_CMp_CMq_h'],
                   c=selecao['Q_pico'], s=24, cmap='plasma', alpha=.7)
    axs[1].set(xlabel='Chuva antecedente de 72 h (mm)',
               ylabel=r'$T_{CM_P\rightarrow CM_Q}$ (h)',
               title='2. Defasagem entre centroides')
    axs[2].scatter(selecao['Pmedia_acumulada_mm'], selecao['T_asc_h'],
                   c=selecao['Q_base_ref'], s=24, cmap='viridis', alpha=.7)
    axs[2].set(xlabel='P acumulada do evento (mm)',
               ylabel=r'$T_{asc}$ (h)',
               title='3. Tempo de ascensão')
    fig.suptitle('Tres tempos caracteristicos de eventos chuva-vazao', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .93))
    salvar(fig, '11_analise_tres_tempos_eventos.png')

    # Histograma conjunto para reconhecer variabilidade entre eventos.
    fig, axs = plt.subplots(1, 3, figsize=(13.6, 3.7))
    for ax, col, label, cor in zip(
        axs, ['T_CMp_Qp_h', 'T_CMp_CMq_h', 'T_asc_h'],
        ['Tempo de retardo ao pico', 'Defasagem entre centroides', 'Tempo de ascensão'],
        ['#d68139', '#7956a9', '#41794b']
    ):
        vals = selecao[col].dropna()
        ax.hist(vals, bins=18, color=cor, alpha=.8)
        if len(vals):
            ax.axvline(vals.median(), ls='--', lw=1.8, c='black',
                       label=f'Mediana: {vals.median():.2f} h')
            ax.legend(fontsize=8)
        ax.set(xlabel='Horas', ylabel='Numero de eventos', title=label)
    fig.tight_layout()
    salvar(fig, '11b_distribuicao_tres_tempos.png')

    nplots = 0
    for data in EVENTOS_EXEMPLO:
        dt = pd.Timestamp(data)
        correspondentes = eventos[(eventos.inicio_chuva <= dt) & (eventos.fim_chuva >= dt)]
        if not correspondentes.empty:
            nplots += 1
            grafico_evento(x, correspondentes.iloc[0],
                           f'12_evento_exemplo_{nplots}_{dt:%Y%m%d}.png')
    if nplots == 0:
        grafico_evento(x, eventos.sort_values('Q_pico').iloc[-1],
                       '12_evento_maior_pico.png')
    return eventos


eventos = analise_eventos(x)
if len(eventos):
    selecionados = eventos.loc[
        eventos.resposta_bem_definida & eventos.T_CMp_CMq_h.between(0, 48)
    ]
    print('Eventos catalogados:', len(eventos), '| filtrados:', len(selecionados))
    print('Tempo de retardo ao pico (T_CMp_Qp) mediano [h]:',
          round(selecionados.T_CMp_Qp_h.median(), 2))
    print('Defasagem entre centroides (T_CMp_CMq) mediana [h]:',
          round(selecionados.T_CMp_CMq_h.median(), 2))
    print('Tempo de ascensão (T_asc) mediano [h]:',
          round(selecionados.T_asc_h.median(), 2))
    cols = ['inicio_chuva', 'Pmedia_acumulada_mm', 'Q_pico',
            'T_CMp_Qp_h', 'T_CMp_CMq_h', 'T_asc_h', 'observacao_ascensao']
    print('\nExemplos de tempos calculados:')
    print(selecionados[cols].head(6).round(2).to_string(index=False))

# %% [markdown]
# # 11 - Relatorio final das descobertas
# Reunimos os principais valores estatisticos e os TRES tempos caracteristicos.
# Sem regressores, redes neurais, ajuste preditivo ou avaliacao de ML.

# %%
def relatorio(x, completos, comparacao, lag, acf, associacoes, pca, ev):
    melhor = []
    for p in COLS_P + ['P_media']:
        a = lag[lag.posto == p]
        melhor.append({
            'posto': p,
            'melhor_lag_pearson_h': int(a.loc[a.pearson_P_dlogQ.idxmax(), 'defasagem_h']),
            'r_pearson_max': float(a.pearson_P_dlogQ.max()),
            'melhor_lag_MI_h': int(a.loc[a.MI_P_dlogQ_nats.idxmax(), 'defasagem_h']),
            'MI_max_nats': float(a.MI_P_dlogQ_nats.max())
        })
    gravar(pd.DataFrame(melhor), '14_resumo_defasagens.csv')
    linhas = [
        'ANALISE EXPLORATORIA CHUVA-VAZAO (SEM MODELOS PREDITIVOS)', '=' * 66,
        f'Periodo: {x.index.min()} a {x.index.max()} | N = {len(x):,} horas',
        f'Anos completos para climatologia: {completos}',
        f'Correlacao diaria entre postos: {comparacao.iloc[0]["r_pearson_diario"]:.3f}',
        f'Q: mediana={x[COL_Q].median():.3f}; P95={x[COL_Q].quantile(.95):.3f}',
        '\nDEFASAGENS ENTRE CHUVA E VARIACAO FUTURA DE LOGQ:'
    ]
    linhas.extend([
        f'  {d["posto"]}: Pearson max lag={d["melhor_lag_pearson_h"]}h '
        f'(r={d["r_pearson_max"]:.3f}); MI max lag={d["melhor_lag_MI_h"]}h '
        f'(MI={d["MI_max_nats"]:.4f} nats)' for d in melhor
    ])
    linhas.append('\nPCA (NAO SUPERVISIONADA):')
    for _, r in pca.iterrows():
        linhas.append(f'  {r.conjunto}: {int(r.n_variaveis)} variaveis; '
                      f'{int(r.PCs_para_95pct)} PCs para 95%; PC1={r.PC1_pct:.1f}%')
    if len(ev):
        filtrados = ev.loc[
            ev.resposta_bem_definida & ev.T_CMp_CMq_h.between(0, 48)
        ]
        linhas.extend([
            '\nTEMPOS CARACTERISTICOS DOS EVENTOS:',
            f'  Eventos catalogados: {len(ev)}; filtrados: {len(filtrados)}',
            '  1. TEMPO DE RETARDO AO PICO (CM_P -> Q_p): T_CMp_Qp = t(Q_p) - t(CM_P)',
            f'     Mediana: {filtrados.T_CMp_Qp_h.median():.2f} h',
            '  2. DEFASAGEM ENTRE CENTROIDES (CM_P -> CM_Q): T_CMp_CMq = t(CM_Q) - t(CM_P)',
            f'     Mediana: {filtrados.T_CMp_CMq_h.median():.2f} h',
            '  3. TEMPO DE ASCENSÃO (inicio -> Q_p): T_asc = t(Q_p) - t(inicio_ascensao)',
            f'     Mediana: {filtrados.T_asc_h.median():.2f} h',
            f'     Limiar da ascensao: Qbase + {LIMIAR_ASCENSAO_FRAC:.0%}*(Qp-Qbase)',
            f'     T_asc nao estimado em {int(filtrados.T_asc_h.isna().sum())} eventos filtrados'
        ])
        # Para quem quiser apenas o catalogo enxuto com os tres tempos.
        gravar(ev[[
            'inicio_chuva', 'fim_chuva', 'Q_pico', 't_Qp', 't_CMp', 't_CMq',
            't_inicio_ascensao', 'T_CMp_Qp_h', 'T_CMp_CMq_h', 'T_asc_h',
            'observacao_ascensao', 'resposta_bem_definida'
        ]], '13b_tres_tempos_eventos.csv')
    linhas.extend([
        '\nCUIDADOS METODOLOGICOS:',
        '- MI discreta (nats) depende da discretizacao e da autocorrelacao.',
        '- Correlacao e MI nao demonstram causalidade ou desempenho de modelos ML.',
        '- PCA usa log1p + padronizacao e nao representa importancia preditiva.',
        '- P_media e a media de dois pluviometros; nao e media areal validada.',
        '- A base foi preenchida; faltam marcadores originais de imputacao.',
        '- Qbase e a mediana antecedente, nao uma separacao fisica do escoamento.',
        '- CM_Q usa a parcela max(Q-Qbase,0) no intervalo da resposta.',
        '- O inicio de ascensao e a primeira passagem pelo limiar na janela observada.',
        '- T_asc ausente significa limiar nao observavel ou resposta nao classificada.',
        '- Eventos compostos e janelas truncadas geram incerteza nos tres tempos.',
        '- Transito de onda de cheia entre secoes exige Q montante e Q jusante.',
        '\nNAO HA: fit de modelos, previsoes, treino/teste, NSE/MAE ou SHAP.'
    ])
    (SAIDA / 'LEIA_ME_RESULTADOS.txt').write_text('\n'.join(linhas) + '\n', encoding='utf-8')
    print('\n'.join(linhas[:27]))


relatorio(x, completos, comparacao, lag, acf, associacoes, pca, eventos)
print('Todas as secoes concluidas. Resultados em:', SAIDA.resolve())
