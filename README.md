# Scripts e Datasets

Este diretório reúne os scripts usados no TCC para extração de atributos fractais, geração de reshapes e preparação de entradas para modelos em Python.

O projeto está dividido em dois ambientes principais:

- MATLAB: processamento das imagens histológicas, extração de atributos fractais e geração dos arquivos/intermediários e imagens (reshapes).
- VS Code / Python: leitura das imagens/reshapes gerados e experimentos (CNNs, análises, etc.).

## Estrutura

- `Scripts/Fractais/`: scripts MATLAB para percolação, lacunaridade, dimensão fractal, cálculo de MTF e geração de saídas auxiliares.
- `Scripts/Reshape/`: scripts MATLAB que geram representações visuais (Recurrence Plot, GASF, GADF, MTF) a partir dos `.mat` de atributos fractais.
- `Scripts/CNN/`: scripts Python para leitura de imagens/reshapes e experimentos com CNN (`function_leitura.py`, `resnet.py`).
- `Scripts/VisiumTransformers/`: espaço para pipelines Transformers; hoje contém um arquivo simples de entrada (`arq.py`).
- `Datasets/`: pasta destinada aos conjuntos de dados e aos arquivos gerados pelos scripts.

## Visão Geral do Fluxo

1. As imagens histológicas são processadas no MATLAB.
2. O script `Scripts/Fractais/FractalFeatures.m` extrai atributos fractais e salva um arquivo `.mat` por imagem.
3. Os atributos salvos podem ser convertidos em novas representações visuais (implementadas em MATLAB neste repositório):
    - Recurrence Plot, via `Scripts/Reshape/reshapeRecPlot.m` ou `Scripts/Reshape/reshapeRecPlot_mat.m`.
    - MTF, via `Scripts/Fractais/carregar_img_fractal_mtf.m` e `Scripts/Fractais/build_mtf.m`.
    - GASF/GADF, via `Scripts/Reshape/reshapeGASF_mat.m` e `Scripts/Reshape/reshapeGADF_mat.m`.

   Observação: havia referência a um notebook `plot_gaf.ipynb` na documentação anterior, porém esse notebook não foi encontrado no repositório. Atualmente as rotinas de reshape e geração de imagens estão implementadas em MATLAB.

## MATLAB

### Dependências

Instale e carregue, no mínimo:

- MATLAB.
- Image Processing Toolbox (funções como `imread`, `imwrite`, `mat2gray`).
- Statistics and Machine Learning Toolbox (quando aplicável).
- Parallel Computing Toolbox (os scripts de fractais usam `parfor` para paralelizar a execução).

Dependências nativas do MATLAB já usadas nos scripts:

- `trapz`, `skewness`, `histcounts`, `dir`, `fullfile`, `load` / `save`

### 1) Extração dos atributos fractais

Arquivo principal: `Scripts/Fractais/FractalFeatures.m`

Esse script percorre uma pasta de imagens `.png`, extrai os atributos com base em três métricas de distância e salva um `.mat` para cada imagem.

Arquivos auxiliares usados por esse pipeline (já no mesmo diretório):

- `pmrChess.m`, `pmrEucl.m`, `pmrManh.m`
- `percChess.m`, `percEucl.m`, `percManh.m`
- `lacunarity.m`, `calcN.m`

Configuração esperada no script:

- Ajustar `source` para a pasta onde estão as imagens.
- Ajustar `destination` se quiser salvar os `.mat` em outro local.
- Conferir `Num_Img` conforme o tamanho da sua base.

Saída gerada:

- Um arquivo `.mat` por imagem, contendo atributos como `ChessLAC`, `EuclLAC`, `ManhLAC`, `ChessFD`, `EuclFD` e `ManhFD`.

### 2) Geração de imagens (Recurrence Plot / GASF / GADF / MTF)

Arquivos principais (MATLAB):

- `Scripts/Reshape/reshapeRecPlot_mat.m`, `Scripts/Reshape/reshapeRecPlot.m` (Recurrence Plot)
- `Scripts/Reshape/reshapeGASF_mat.m` (GASF)
- `Scripts/Reshape/reshapeGADF_mat.m` (GADF)
- `Scripts/Fractais/carregar_img_fractal_mtf.m`, `Scripts/Fractais/build_mtf.m` (MTF)

`reshapeRecPlot_mat.m` e os scripts GAF/GADF trabalham diretamente sobre os `.mat` produzidos no passo de extração; eles montam o sinal a partir dos atributos e geram as imagens correspondentes.

## VS Code / Python

### Dependências e uso atual

Os reshapes e a maioria do pré-processamento de atributos são implementados em MATLAB. Os scripts Python presentes no repositório são, hoje, principalmente leitores e pontos de entrada para experimentos (não geram reshapes):

- `Scripts/CNN/function_leitura.py`: utilitários para leitura de imagens e leitura dos reshapes gerados no MATLAB.
- `Scripts/CNN/resnet.py`: exemplo/entrada que importa `torch`/`torchvision` e depende de imagens/reshapes pré-gerados.
- `Scripts/VisiumTransformers/arq.py`: arquivo de entrada atual com saídas de teste.

Pacotes Python observados nas implementações atuais:

- `numpy`
- `matplotlib`
- `Pillow` (usado em `function_leitura.py` como `PIL`)
- `torch`, `torchvision` (usados em `resnet.py`)

Nota: o pacote `pyts` e um notebook `plot_gaf.ipynb` foram mencionados na documentação anterior, mas não há evidência de uso ativo desses recursos no repositório atual. Se quiser suportar geração de GASF/GADF via Python, posso adicionar um notebook ou adaptar as funções existentes para Python/pyts.

### Sugestão rápida de execução (MATLAB primeiro)

1. Rode `Scripts/Fractais/FractalFeatures.m` para extrair atributos e gerar os `.mat`.
2. Use os scripts em `Scripts/Reshape/` para gerar Recurrence Plots, GASF, GADF e MTF a partir dos `.mat` gerados.
3. Se for usar os pipelines Python (CNN/experimentos), aponte as variáveis `source`/`source_rec_mtf` nos scripts Python para as pastas onde o MATLAB salvou as imagens.

## Observações e inconsistências encontradas

- Os reshapes (Recurrence Plot, GASF, GADF, MTF) são implementados em MATLAB (`Scripts/Reshape/*`, `Scripts/Fractais/*`). O README anterior sugeria um notebook `plot_gaf.ipynb` para GAF/GADF, porém esse notebook não está no repositório.
- O arquivo `Scripts/CNN/arq.py` listado anteriormente no README não foi encontrado. Em `Scripts/CNN/` existem `function_leitura.py` e `resnet.py`.
- `Scripts/VisiumTransformers/arq.py` existe e contém apenas mensagens de teste.

Se desejar, eu posso:

1. Remover referências obsoletas (ex.: `plot_gaf.ipynb`) ou adicionar um notebook Python equivalente.
2. Gerar um pequeno `requirements.txt` com as dependências Python detectadas (`numpy`, `matplotlib`, `Pillow`, `torch`, `torchvision`).
3. Atualizar os scripts Python para apontarem por padrão para caminhos relativos em vez de caminhos absolutos.

Informe qual das opções prefere e eu aplico em seguida.

## Referência de Arquivos Principais

- `Scripts/Fractais/FractalFeatures.m`
- `Scripts/Fractais/SaveCSVPercCLACDF3Distances.m`
- `Scripts/Fractais/carregar_img_fractal_mtf.m`
- `Scripts/Fractais/build_mtf.m`
- `Scripts/Reshape/reshapeRecPlot.m`
- `Scripts/Reshape/reshapeRecPlot_mat.m`
- `Scripts/Reshape/reshapeGASF_mat.m`
- `Scripts/Reshape/reshapeGADF_mat.m`
- `Scripts/CNN/function_leitura.py`
- `Scripts/CNN/resnet.py`
- `Scripts/VisiumTransformers/arq.py`
