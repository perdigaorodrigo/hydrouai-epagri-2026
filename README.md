# Inteligência Artificial Aplicada aos Recursos Hídricos

**Hydro-UAI · EPAGRI · 2026**

Repositório de materiais práticos da capacitação em Inteligência Artificial aplicada aos Recursos Hídricos, com foco em preparação de séries hidrológicas, modelagem chuva–vazão, vazões mínimas e máximas, transferência de modelos entre bacias e interpretação dos resultados.

O curso está organizado em **oito encontros remotos de quatro horas**, totalizando **32 horas**. Os scripts e notebooks serão adicionados progressivamente.

## Objetivos

- Organizar, avaliar e preparar séries hidrológicas para aplicações de IA.
- Aplicar e avaliar técnicas de preenchimento de falhas.
- Desenvolver modelos MLP e LSTM para problemas hidrológicos.
- Avaliar estratégias específicas para estiagens e cheias.
- Explorar regionalização e transferência de modelos entre bacias.
- Investigar, com ferramentas de IA explicável, os fatores associados ao desempenho e à transferabilidade dos modelos.

## Programa e materiais

Os identificadores abaixo seguem a organização atual dos materiais: **01, 02, 03, 04, 05, 07, 08 e 09**. A ausência do identificador 06 não representa um encontro pendente.

| Encontro | Script / notebook | Conteúdo prático | Material |
| --- | --- | --- | --- |
| 1 | **01 — Organização, análise e preenchimento de dados** | Ler e organizar as séries; realizar análise exploratória; identificar inconsistências e lacunas; aplicar e comparar técnicas de preenchimento. | [Pasta do encontro](aulas/01_organizacao_analise_preenchimento/) |
| 2 | **02 — Preparação dos dados para Machine Learning** | Definir entradas e saída; criar defasagens; selecionar variáveis; dividir os períodos de treino, validação e teste; normalizar os dados. | [Pasta do encontro](aulas/02_preparacao_machine_learning/) |
| 3 | **03 — Redes neurais MLP aplicadas à Hidrologia** | Construir e treinar uma MLP; testar arquiteturas e hiperparâmetros; avaliar curvas de aprendizagem e desempenho. | [Pasta do encontro](aulas/03_mlp_hidrologia/) |
| 4 | **04 — Modelagem chuva–vazão: MLP × LSTM, com foco em mínimas** | Preparar janelas temporais; construir e treinar uma LSTM; comparar com a MLP por métricas e hidrogramas; ajustar funções de perda e avaliar métricas voltadas às mínimas. | [Pasta do encontro](aulas/04_mlp_lstm_minimas/) |
| 5 | **05 — Modelagem de vazões máximas** | Avaliar erros nos picos; implementar estratégias de treinamento voltadas aos extremos; comparar os modelos antes e depois do retreinamento. | [Pasta do encontro](aulas/05_vazoes_maximas/) |
| 6 | **07 — Transfer Learning entre bacias** | Treinar em bacias fonte; transferir o modelo para uma bacia-alvo; realizar ajuste fino; avaliar o desempenho. | [Pasta do encontro](aulas/07_transfer_learning/) |
| 7 | **08 — Comparação Index-Flood × Transfer Learning** | Integrar resultados; comparar estimativas de máximas; discutir desempenho, limitações e aplicação na EPAGRI. | [Pasta do encontro](aulas/08_index_flood_transfer_learning/) |
| 8 | **09 — XAI aplicado à transferabilidade** | **Escopo proposto:** interpretar a contribuição das variáveis nos modelos fonte e alvo, comparar explicações antes e depois da transferência e investigar fatores associados ao desempenho entre bacias. | [Pasta do encontro](aulas/09_xai_transferabilidade/) |

**Situação dos materiais:** as pastas estão preparadas; os oito scripts/notebooks ainda serão inseridos. No encontro 4, a prática de MLP × LSTM está prevista com Bruno e o ajuste de funções de perda e métricas de mínimas com André. O escopo do encontro 8 será detalhado pela equipe.

Para o encontro 7, a rotina ou os resultados de referência de Index-Flood deverão ser incorporados ao material. As estimativas serão comparadas para as mesmas bacias e uma grandeza comum, com critérios definidos na atividade.

## Estrutura do repositório

```text
.
├── README.md
├── .gitignore
├── aulas/
│   ├── 01_organizacao_analise_preenchimento/
│   ├── 02_preparacao_machine_learning/
│   ├── 03_mlp_hidrologia/
│   ├── 04_mlp_lstm_minimas/
│   ├── 05_vazoes_maximas/
│   ├── 07_transfer_learning/
│   ├── 08_index_flood_transfer_learning/
│   └── 09_xai_transferabilidade/
├── dados/
│   └── README.md
└── resultados/
    └── README.md
```

Cada pasta de aula receberá seu script Python (`.py`) ou notebook (`.ipynb`), acompanhado das instruções específicas de execução. Os arquivos de dados e resultados de execução são mantidos localmente.

## Dados e estudo de caso

A proposta de capacitação prevê a **bacia hidrográfica do Rio Araranguá, em Santa Catarina**, como estudo de caso. As estações, os períodos e as variáveis usados em cada prática serão documentados quando os materiais forem incorporados.

O pacote BD01 disponibilizado para preparação do curso inclui:

| Fonte / arquivo | Conteúdo identificado |
| --- | --- |
| ANA/Hidroweb — `vazoes.zip` | Séries diárias de vazão e indicador de consistência. |
| Monitoramento disponibilizado pela EPAGRI — `Hidrologia.csv` | Precipitação, nível e vazão em registros horários, com disponibilidade variável por estação. |
| Estação 2383 — Antônio Carlos, Bairro Usina | Temperatura do ar, umidade relativa, molhamento foliar e precipitação de uma hora. |
| `dados_hidro_82_83_84.mdb` | Banco complementar das sub-bacias 82, 83 e 84, com registros hidrológicos e outras tabelas de monitoramento. |
| `BDEpagri.xls` e arquivos SIG | Cadastro de estações e informações geográficas, incluindo seis sub-bacias de exemplo. |

Esses conjuntos não constituem, automaticamente, uma base integrada da bacia do Araranguá. A associação espacial, a sobreposição temporal e a qualidade das séries precisam ser verificadas para cada exercício.

### Acesso e uso dos dados

Segundo o `leiame.txt` do BD01, as séries de `vazoes.zip` foram obtidas do Hidroweb e são públicas. As demais séries históricas, inclusive o banco `.mdb`, estão sujeitas às restrições de uso e compartilhamento informadas para o curso.

**Os dados do BD01 não acompanham este repositório.** Os participantes deverão utilizar o canal de acesso indicado pela organização. Veja [as instruções de organização local](dados/README.md).

## Como utilizar os materiais

1. Baixe ou clone este repositório.
2. Consulte a pasta do encontro e confirme se o script/notebook já foi disponibilizado.
3. Obtenha os dados autorizados e organize-os conforme as instruções da aula.
4. Instale as dependências e utilize a versão de Python indicadas no material do encontro.
5. Execute a atividade e mantenha os resultados na pasta local `resultados/`.

As versões de Python, as bibliotecas e os comandos de instalação serão definidos após a incorporação dos scripts. A estrutura inicial ainda não contém código executável nem um ambiente de execução validado.

## Inclusão dos próximos scripts

Ao adicionar cada material, registrar na pasta da aula:

- Objetivo e nome do arquivo principal.
- Dados necessários, origem, unidades, resolução temporal e período utilizado.
- Dependências e ordem de execução.
- Saídas esperadas e critérios de avaliação.
- Referência do código utilizado, quando houver.

Usar caminhos relativos e documentar a separação entre treino, validação e teste. Antes de publicar notebooks, limpar saídas que contenham dados restritos. O `.gitignore` não remove dados já versionados nem conteúdo incorporado às células de um notebook.

## Equipe

Capacitação desenvolvida pela **Hydro-UAI** para a **EPAGRI**, com participação de Bruno Brentan, André Rodrigues e Rodrigo Perdigão nos encontros previstos.

## Uso dos materiais

As condições de licença e redistribuição dos códigos serão definidas pela equipe responsável. As permissões de uso dos dados devem ser observadas independentemente das condições aplicáveis ao código.
