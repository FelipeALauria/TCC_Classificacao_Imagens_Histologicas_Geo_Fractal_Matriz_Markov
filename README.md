# Scripts e Datasets — FNN (Fractal Neural Network)

Este repositório reúne os scripts do TCC para extração de atributos fractais de imagens histológicas, geração de representações visuais (reshapes) a partir desses atributos, e experimentos de classificação com CNN combinando a imagem original e os reshapes como canais de entrada.

O projeto está dividido em dois ambientes:

- **MATLAB**: extração dos atributos fractais (percolação, lacunaridade, dimensão fractal) e geração dos reshapes (Recurrence Plot, MTF, GASF, GADF) a partir das imagens histológicas.
- **Python (VS Code)**: leitura das imagens originais e dos reshapes gerados no MATLAB, montagem dos canais de entrada e experimentos de classificação com ResNet50 multicanal.

## Estrutura

- `Scripts/Fractais/`: extração de atributos fractais (percolação, lacunaridade, dimensão fractal) por três métricas de distância, e um script exploratório de MTF intermediária. Contém também `PercolationOnly/`, uma implementação de referência de terceiros (com licença própria) usada como consulta, não integrada ao pipeline principal.
- `Scripts/Reshape/`: geração das representações visuais (Recurrence Plot, MTF, GASF, GADF) a partir dos `.mat` de atributos fractais, e um script de visualização comparativa para apresentação ao orientador.
- `Scripts/CNN/`: pipeline Python de leitura dos dados, montagem dos canais de entrada e treino/avaliação de ResNet50 multicanal com validação cruzada (`functions.py`, `resnet.py`, `orquestrador.py`), além dos artefatos já gerados por rodadas anteriores (`metricas.csv`, `predicoes_CR-SameRes.csv`, `splits_kfold_CR-SameRes.json`).
- `Scripts/VisiumTransformers/`: espaço reservado para um pipeline com Transformers; hoje contém apenas `arq.py`, um arquivo de teste/placeholder sem lógica implementada.
- `Datasets/Imagens Histológicas/`: datasets de imagens histológicas usados no projeto (ver seção própria abaixo).
- `requirements.txt`: dependências Python do projeto.

## Visão Geral do Fluxo

1. As imagens histológicas (`Datasets/Imagens Histológicas/<Dataset>/<Classe>/*.png`) são processadas no MATLAB.
2. `Scripts/Fractais/FractalFeatures.m` extrai, para cada imagem, atributos de percolação (`p`, `g`, `h`), lacunaridade (`LAC`) e dimensão fractal (`FD`) — locais e globais — usando três métricas de distância (Chessboard, Euclidiana, Manhattan) e salva um `.mat` por imagem.
3. `Scripts/Fractais/SaveCSVPercCLACDF3Distances.m` consolida os atributos extraídos de uma classe/dataset em um único CSV.
4. A partir dos `.mat` de atributos, os scripts em `Scripts/Reshape/` montam um sinal 1D por distância (concatenando `p`, `g`, `h`, `LAC` e o vetor `nn` do cálculo de FD) e empilham as três distâncias como canais R (Chess), G (Eucl) e B (Manh) de uma imagem RGB:
   - **Recurrence Plot** — `reshapeRecPlot_mat.m` (usa `cerecurr_y.m`), normalizando cada atributo entre as três distâncias antes de gerar o plot.
   - **MTF (Markov Transition Field)** — `reshapeMTF_mat.m` / `carregar_img_fractal_mtf.m` (usa `build_mtf.m`), gerada em múltiplos níveis de quantização `Q` (8, 16, 32, 64 bins).
   - **GASF** — `reshapeGASF_mat.m` (usa `build_gasf.m`).
   - **GADF** — `reshapeGADF_mat.m` (usa `build_gadf.m`).
   - `carg_visu_mtf.m` monta uma figura comparativa (imagem original + sinal 1D de atributos + Recurrence Plot + MTF em cada resolução) para apresentação ao orientador.
5. Cada dataset termina com a estrutura `source/<Canal>/<Classe>/<size>/*.png` (um subconjunto de `RecPlot`, `MTF`, `GASF`, `GADF`) ao lado das imagens originais em `source/<Classe>/*.png` — layout esperado pelo pipeline Python.
6. No Python, `Scripts/CNN/orquestrador.py` (ou `resnet.py` para um único dataset) lê essa estrutura, monta os canais de entrada desejados (imagem original + reshapes) e treina/avalia uma ResNet50 adaptada, com validação cruzada K-fold, registrando métricas e predições em CSV.

## MATLAB

### Dependências

- MATLAB.
- Image Processing Toolbox (`imread`, `imwrite`, `imresize`, `mat2gray`, `bwlabel`).
- Statistics and Machine Learning Toolbox (`skewness`, `fitlm`, `histcounts`).
- Parallel Computing Toolbox (os scripts de percolação usam `parfor`).

### 1) Extração dos atributos fractais

Arquivo principal: `Scripts/Fractais/FractalFeatures.m`

Percorre uma pasta de imagens `.png` de uma classe, extrai os atributos com base em três métricas de distância e salva um `.mat` por imagem (todas as variáveis do workspace, incluindo `ChessLAC`/`EuclLAC`/`ManhLAC`, `ChessFD`/`EuclFD`/`ManhFD` e os vetores `p`/`g`/`h`/`nn` de cada distância).

Arquivos auxiliares usados por esse script:

- `pmrChess.m`, `pmrEucl.m`, `pmrManh.m` — matrizes de probabilidade por distância.
- `percChess.m`, `percEucl.m`, `percManh.m` — percolação (PERC) por distância.
- `lacunarity.m` — lacunaridade a partir da matriz de probabilidade.
- `calcN.m` — usado no cálculo da dimensão fractal (FD) via regressão log-log.
- `PercolationGray.m` — implementação alternativa/generalizada de percolação (`clustperc`, parametrizada por distância máxima `maxr`), não referenciada pelo pipeline atual de `FractalFeatures.m`.

Configuração esperada no script: ajustar `source` (pasta com as imagens da classe), `destination` (pasta de saída dos `.mat`), `maxL` e `Num_Img` conforme a base.

`Scripts/Fractais/SaveCSVPercCLACDF3Distances.m` consolida os `.mat` de uma classe/dataset em um CSV único com todos os atributos.

`Scripts/Fractais/loading.m` é um script exploratório que carrega os `.mat` de uma classe e gera apenas os `.mat` intermediários de MTF para vários valores de `Q` (8/16/32/64), sem exportar PNG — depende de `build_mtf.m`, que está em `Scripts/Reshape/`, então as duas pastas precisam estar no path do MATLAB.

`Scripts/Fractais/PercolationOnly/` é uma implementação de referência de terceiros (com `LICENSE` própria) para percolação, mantida como consulta — não faz parte do pipeline principal do projeto.

### 2) Geração de imagens (Recurrence Plot / MTF / GASF / GADF)

Funções de construção do sinal 2D:

- `Scripts/Reshape/build_mtf.m` — MTF (quantização + matriz de Markov + campo de transição).
- `Scripts/Reshape/build_gasf.m`, `Scripts/Reshape/build_gadf.m` — GASF/GADF (ângulo polar via `acos` da série normalizada).
- `Scripts/Reshape/cerecurr_y.m` — usado pelo Recurrence Plot.

Scripts que aplicam essas funções aos `.mat` de atributos fractais e exportam PNG (RGB, um canal por distância, redimensionado para 224×224):

- `Scripts/Reshape/reshapeRecPlot_mat.m` (e a variante `reshapeRecPlot.m`).
- `Scripts/Reshape/reshapeMTF_mat.m` e `Scripts/Fractais/carregar_img_fractal_mtf.m` (MTF, múltiplos `Q`).
- `Scripts/Reshape/reshapeGASF_mat.m`.
- `Scripts/Reshape/reshapeGADF_mat.m`.
- `Scripts/Reshape/carg_visu_mtf.m` — gera a figura comparativa (original + sinal 1D + Recurrence Plot + MTFs) usada em apresentações.
- `Scripts/Reshape/teste.m` — script de teste/scratch.

> Atenção: `reshapeRecPlot_mat.m` tem um `addpath` hardcoded para um caminho antigo (`C:\Users\felip\Documents\MATLAB\Scripts e Datasets\Scripts\Reshape`), que não corresponde ao caminho atual do repositório (`Scripts_Datasets`). Ajustar esse caminho (ou remover o `addpath` e garantir que a pasta já esteja no path) antes de rodar em outra máquina.

## Datasets

`Datasets/Imagens Histológicas/` contém sete datasets, cada um em `<Dataset>/<Classe>/*.png`, com os canais de reshape (quando já gerados) em `<Dataset>/<Canal>/<Classe>/<size>/*.png`:

| Dataset | Classes | Canais de reshape já gerados | Observação |
|---|---|---|---|
| `CR` | `Benign`, `Malignant` | `RecPlot`, `MTF`, `GASF`, `GADF` | Único dataset com todos os reshapes prontos hoje. |
| `CR-SameRes` | `Benign`, `Malignant` | — (só imagem original) | Usado na rodada de CNN já registrada em `metricas.csv`/`predicoes_CR-SameRes.csv`. |
| `LG` | `1`, `2` | — | Rótulos ainda não confirmados com o orientador (o que "1"/"2" representam). |
| `UCSB` | `Benign`, `Malignant` | — | |
| `DIS` | `healthy`, `mild`, `moderate`, `severe` | — | Multiclasse (4 classes); fora da rodada binária atual. |
| `LA` | `1`, `2`, `3`, `4` | — | Multiclasse (4 classes); fora da rodada binária atual. |
| `NHL` | `CLL`, `FL`, `MCL` | — | Multiclasse (3 classes) e sem classe saudável; não binarizável; fora da rodada atual. |

`Scripts/CNN/orquestrador.py` roda hoje apenas os datasets binários disponíveis (`CR`, `CR-SameRes`, `LG`, `UCSB`), e apenas com a combinação `img_original` (imagem original, sem reshape), já que a maioria dos datasets ainda não tem os canais de reshape gerados.

## VS Code / Python

### Dependências

Ver `requirements.txt` (instalar com `pip install -r requirements.txt`, dentro do `.venv` do projeto). Principais pacotes: `numpy`, `matplotlib`, `Pillow`, `scikit-learn`, `torch`, `torchvision`.

### `Scripts/CNN/functions.py`

Biblioteca central do pipeline Python, organizada em:

- **Descoberta de dataset** — `descobrir_datasets`, `descobrir_classes`, `descobrir_canais_reshape`, `montar_classes_dict`: layout esperado `source/<Classe>/*.png` + `source/<Canal>/<Classe>/<size>/*.png`, com `NOMES_RESHAPE_PADRAO = ("RecPlot", "MTF", "GASF", "GADF")`.
- **Carregamento e montagem dos canais** — `indexar_por_numero`, `obter_arquivo`, `carregar_classe`, `padronizar`, `montar_input_canais`: monta o tensor de entrada combinando a imagem original com os canais de reshape escolhidos.
- **Modelo** — `criar_resnet50_nch`: ResNet50 pré-treinada adaptada para receber `N` canais de entrada e `num_classes` de saída.
- **Validação cruzada** — `gerar_ou_carregar_splits` (K-fold estratificado, cacheado em JSON por dataset — ver `splits_kfold_CR-SameRes.json`), `indices_por_ids`, `_HistologiaDataset` (Dataset customizado do PyTorch), `treinar_e_avaliar_fold`.
- **Métricas e logging** — `calcular_metricas` (acurácia, precisão, recall, F1, AUC), `salva_metrica_csv` (acumula em `metricas.csv`), `salvar_predicoes_csv`, `comparar_softmax` (compara as probabilidades preditas entre duas combinações de canais).
- **Orquestração por dataset** — `interpretar_combinacoes` (filtra combinações cujos canais ainda não foram gerados naquele dataset, sem interromper as demais) e `rodar_dataset` (roda todas as combinações válidas de um dataset, com K-fold completo).

### `Scripts/CNN/resnet.py`

`rodar_resnet(source, combinacoes_desejadas)` chama `rodar_dataset` para **um** dataset e gera o relatório visual: gráfico de barras comparando as métricas (acurácia, precisão, recall, F1, AUC) entre as combinações de canais (`comparacao_combinacoes_<dataset>.png`) e, havendo 2+ combinações, um CSV de comparação de softmax entre a primeira e a última combinação. Pode ser executado isoladamente (bloco `if __name__ == "__main__"`) para um dataset específico com todas as combinações que ele já suportar.

### `Scripts/CNN/orquestrador.py`

Roda vários datasets de uma vez (`rodar_todos_datasets`), chamando `rodar_resnet` para cada um e isolando erros por dataset (um dataset com problema é registrado no console e não interrompe os demais). Configuração atual:

- `DATASETS_BINARIOS = ["CR", "CR-SameRes", "LG", "UCSB"]` (datasets multiclasse — `DIS`, `LA`, `NHL` — ficam de fora).
- `COMBINACOES_DESEJADAS` é um dicionário de ablação com todas as combinações possíveis de `img`/`recplot`/`mtf`/`gasf`/`gadf` comentadas; hoje só `img_original` (`["img"]`) está ativa.

### Saídas já geradas

- `Scripts/CNN/metricas.csv` — histórico de métricas por rodada/combinação/fold.
- `Scripts/CNN/predicoes_CR-SameRes.csv` — predições salvas da rodada do dataset `CR-SameRes`.
- `Scripts/CNN/splits_kfold_CR-SameRes.json` — splits de K-fold cacheados para `CR-SameRes` (garante reprodutibilidade entre rodadas).

### `Scripts/VisiumTransformers/arq.py`

Ainda é só um placeholder de teste (duas linhas de `print`); o pipeline de Transformers para esse projeto ainda não foi iniciado.

## Observações e inconsistências encontradas

- `Scripts/Fractais/README.md` (interno, em inglês) ainda descreve o projeto como "FNN" e cita um notebook `plot_gaf.ipynb` como gerador de GASF/GADF — hoje isso é feito em MATLAB (`build_gasf.m`/`build_gadf.m` + `reshapeGASF_mat.m`/`reshapeGADF_mat.m`), e esse notebook não existe no repositório. Esse README interno está desatualizado.
- Nem todo dataset tem os quatro canais de reshape gerados: só `CR` tem `RecPlot`/`MTF`/`GASF`/`GADF` completos hoje; os demais (`CR-SameRes`, `LG`, `UCSB`) só têm a imagem original, por isso o `orquestrador.py` está rodando apenas a combinação `img_original`.
- `reshapeRecPlot_mat.m` tem um `addpath` hardcoded apontando para um caminho de máquina/pasta antigos, que não bate com o caminho atual do repositório.
- `Scripts/Fractais/loading.m` depende de `build_mtf.m`, que está em `Scripts/Reshape/` — as duas pastas precisam estar no path do MATLAB para esse script rodar.
- `PercolationGray.m` (função `clustperc`) parece uma versão alternativa/generalizada da percolação, mas não é chamada por `FractalFeatures.m` nem por nenhum outro script do pipeline atual — vale confirmar se é código em desenvolvimento ou já obsoleto.
- Os rótulos de classe do dataset `LG` (pastas `1`/`2`) ainda não foram confirmados com o orientador; por isso o `orquestrador.py` já inclui `LG` na lista de binários, mas com uma ressalva no comentário do código.
- `Scripts/CNN/function_leitura.py`, citado em versões anteriores deste README, não existe mais — foi substituído por `functions.py` (biblioteca) + `orquestrador.py`/`resnet.py` (execução).

## Referência de Arquivos Principais

- `Scripts/Fractais/FractalFeatures.m`
- `Scripts/Fractais/SaveCSVPercCLACDF3Distances.m`
- `Scripts/Fractais/loading.m`
- `Scripts/Fractais/PercolationGray.m`
- `Scripts/Reshape/build_mtf.m`
- `Scripts/Reshape/build_gasf.m`
- `Scripts/Reshape/build_gadf.m`
- `Scripts/Reshape/reshapeRecPlot_mat.m`
- `Scripts/Reshape/reshapeMTF_mat.m`
- `Scripts/Reshape/reshapeGASF_mat.m`
- `Scripts/Reshape/reshapeGADF_mat.m`
- `Scripts/Reshape/carg_visu_mtf.m`
- `Scripts/CNN/functions.py`
- `Scripts/CNN/resnet.py`
- `Scripts/CNN/orquestrador.py`
- `Scripts/VisiumTransformers/arq.py`
- `requirements.txt`