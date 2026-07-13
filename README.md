# Scripts e Datasets

Este diretório reúne os scripts usados no TCC para extração de atributos fractais, geração de reshapes e preparação de entradas para modelos em Python.

O projeto está dividido em dois ambientes principais:

- MATLAB: processamento das imagens histológicas, extração de atributos fractais e geração dos arquivos intermediários.
- VS Code: execução dos scripts Python e do notebook de reshape em GAF/GADF.

## Estrutura

- `Scripts/Fractais/`: scripts MATLAB para percolação, lacunaridade, dimensão fractal e geração de saídas auxiliares.
- `Scripts/Reshape/`: scripts MATLAB e notebook Python para transformar os atributos em imagens do tipo Recurrence Plot e Gramian Angular Field.
- `Scripts/CNN/`: ponto de entrada para futura ou atual etapa em Python voltada à CNN.
- `Scripts/VisiumTransformers/`: ponto de entrada para futura ou atual etapa em Python voltada a modelos Transformers/Visium.
- `Datasets/`: pasta destinada aos conjuntos de dados e aos arquivos gerados pelos scripts.

## Visão Geral do Fluxo

1. As imagens histológicas são processadas no MATLAB.
2. O script `FractalFeatures.m` extrai atributos fractais e salva um arquivo `.mat` por imagem.
3. Os atributos salvos podem ser convertidos em novas representações visuais:
	- Recurrence Plot, via `reshapeRecPlot.m` ou `reshapeRecPlot_mat.m`.
	- MTF, via `carregar_img_fractal_mtf.m`.
	- GASF/GADF, via `plot_gaf.ipynb`.
4. Os arquivos gerados podem então ser usados como entrada para os experimentos em Python.

## MATLAB

### Dependências

Instale e carregue, no mínimo:

- MATLAB.
- Image Processing Toolbox, por causa de funções como `imread`, `imwrite`, `mat2gray` e manipulação de imagens.
- Statistics and Machine Learning Toolbox, por causa de `fitlm`.
- Parallel Computing Toolbox, porque os scripts de fractais usam `parfor` para paralelizar a execução.

Dependências nativas do MATLAB já usadas nos scripts:

- `trapz`
- `skewness`
- `histcounts`
- `dir`
- `fullfile`
- `load` / `save`

### 1) Extração dos atributos fractais

Arquivo principal: `Scripts/Fractais/FractalFeatures.m`

Esse script percorre uma pasta de imagens `.png`, extrai os atributos com base em três métricas de distância e salva um `.mat` para cada imagem.

Na execução dos fractais, a paralelização é controlada pelo MATLAB/Parallel Computing Toolbox. Antes de rodar, ajuste a quantidade de workers/cores disponíveis para reservar os núcleos desejados exclusivamente para o processamento do código.

Os arquivos auxiliares usados por esse pipeline já estão no mesmo diretório:

- `pmrChess.m`
- `pmrEucl.m`
- `pmrManh.m`
- `percChess.m`
- `percEucl.m`
- `percManh.m`
- `lacunarity.m`
- `calcN.m`

Configuração esperada no script:

- Ajustar `source` para a pasta onde estão as imagens.
- Ajustar `destination` se quiser salvar os `.mat` em outro local.
- Conferir `Num_Img` conforme o tamanho da sua base.
- Manter o padrão de nomeação das imagens, que no script está como `Class 2 - (n).png`.

Saída gerada:

- Um arquivo `.mat` por imagem, contendo atributos como `ChessLAC`, `EuclLAC`, `ManhLAC`, `ChessFD`, `EuclFD` e `ManhFD`.

### 2) Recurrence Plot a partir dos atributos

Arquivos principais:

- `Scripts/Reshape/reshapeRecPlot_mat.m`
- `Scripts/Reshape/reshapeRecPlot.m`

`reshapeRecPlot_mat.m` trabalha diretamente sobre os `.mat` produzidos no passo anterior. Ele monta o sinal com os atributos fractais, separa em três canais e gera uma imagem RGB no formato Recurrence Plot.

`reshapeRecPlot.m` faz a mesma ideia, mas a partir de uma planilha ou CSV carregado em memória.

Dependência adicional:

- `cerecurr_y.m`

Saída gerada:

- Imagens `.png` com prefixo `*_rp.png` ou `F-RecPlot n.png`, dependendo da versão usada.

### 3) MTF a partir dos atributos

Arquivo principal: `Scripts/Fractais/carregar_img_fractal_mtf.m`

Esse script gera uma visualização do pipeline com a imagem original, o sinal 1D dos atributos, o Recurrence Plot e as MTFs em diferentes resoluções.

Dependência principal:

- `build_mtf.m`

Saída gerada:

- Figura resumo `*_amostragem.png`.

### 4) Exportação para CSV

Arquivo relacionado: `Scripts/Fractais/SaveCSVPercCLACDF3Distances.m`

Esse passo é usado para salvar os atributos fractais em CSV, o que permite o uso posterior no notebook Python do reshape em GAF/GADF.

## VS Code

### Dependências

Para os scripts Python e para o notebook, use Python 3.x com um ambiente virtual separado.

Pacotes observados no que já foi implementado:

- `numpy`
- `matplotlib`
- `pyts`
- `scikit-learn` e `scipy` como dependências do `pyts`
- `jupyter` ou suporte a notebooks no VS Code

Se for executar no Google Colab, também será usado:

- `google.colab`

### 1) Notebook de GAF/GADF

Arquivo: `Scripts/Reshape/plot_gaf.ipynb`

Esse notebook lê `UCSBFeatures.csv`, reorganiza os dados em séries temporais e gera imagens GASF/GADF com `pyts`.

Passos de uso:

1. Abra o notebook no VS Code.
2. Selecione um kernel Python válido.
3. Instale as dependências do ambiente.
4. Ajuste `root_path` para a pasta onde está o CSV e onde as imagens devem ser salvas.
5. Execute as células na ordem.

Observação importante:

- O notebook foi escrito para um fluxo que usa Google Colab e Drive, mas também pode ser adaptado para execução local no VS Code.

### 2) Scripts Python em CNN e VisiumTransformers

Arquivos:

- `Scripts/CNN/arq.py`
- `Scripts/VisiumTransformers/arq.py`

No estado atual, esses dois arquivos estão vazios e funcionam como pontos de entrada/placeholder para a etapa Python do projeto.

Isso significa que, por enquanto, não há dependências Python específicas declaradas nesses arquivos. Quando a implementação for adicionada, o ideal é documentar ali mesmo quais bibliotecas cada pipeline usa.

## Como Rodar no MATLAB

1. Abra o MATLAB.
2. Adicione ao path as pastas `Scripts/Fractais` e `Scripts/Reshape`.
3. Ajuste as variáveis `source` e `destination` nos scripts, apontando para a sua estrutura local.
4. Rode primeiro `FractalFeatures.m` para gerar os `.mat` com os atributos.
5. Rode depois `reshapeRecPlot_mat.m` ou `reshapeRecPlot.m`, conforme a origem dos dados.
6. Se quiser gerar MTF, execute `carregar_img_fractal_mtf.m`.

Exemplo de ordem sugerida:

1. `FractalFeatures.m`
2. `SaveCSVPercCLACDF3Distances.m`
3. `reshapeRecPlot_mat.m`
4. `carregar_img_fractal_mtf.m`
5. `plot_gaf.ipynb`

## Como Rodar no VS Code

1. Abra a pasta `Scripts e Datasets` no VS Code.
2. Crie e ative um ambiente virtual Python.
3. Instale os pacotes necessários.
4. Abra `plot_gaf.ipynb` ou os scripts `.py` que forem sendo adicionados.
5. Execute o código com o kernel Python do projeto.

## Instalação Sugerida de Pacotes Python

```bash
pip install numpy matplotlib pyts jupyter scikit-learn scipy
```

Se você for usar GPU, CNNs ou Transformers de fato no futuro, adicione os pacotes correspondentes quando o código desses módulos estiver implementado.

## Observações Importantes

- Os caminhos absolutos dentro dos scripts foram escritos para o ambiente do autor e precisam ser ajustados na sua máquina.
- Os nomes de arquivos de entrada devem bater com o padrão esperado pelos scripts.
- Os scripts Python `arq.py` ainda não têm implementação, então a documentação de CNN e VisiumTransformers deve ser atualizada quando o código for adicionado.
- O notebook `plot_gaf.ipynb` contém etapas preparadas para Colab, mas também pode ser executado localmente no VS Code com pequenas adaptações.

## Referência de Arquivos Principais

- `Scripts/Fractais/FractalFeatures.m`
- `Scripts/Fractais/SaveCSVPercCLACDF3Distances.m`
- `Scripts/Fractais/carregar_img_fractal_mtf.m`
- `Scripts/Reshape/reshapeRecPlot.m`
- `Scripts/Reshape/reshapeRecPlot_mat.m`
- `Scripts/Reshape/plot_gaf.ipynb`
- `Scripts/CNN/arq.py`
- `Scripts/VisiumTransformers/arq.py`
