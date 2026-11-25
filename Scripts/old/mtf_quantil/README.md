# MTF com normalização por atributo + bins por quantil (arquivado)

Teste feito em novembro de 2026 para tentar melhorar o desempenho do MTF. **Não deu ganho relevante e não é usado no TCC**: não entra na monografia nem nos gráficos de métricas. O pipeline oficial continua com o `Scripts/Reshape/build_mtf.m` e o `Scripts/Reshape/reshapeMTF_mat.m` originais.

## A ideia

No MTF original, os problemas identificados foram:

1. **Escalas diferentes entre atributos.** O `mat2gray` era aplicado no sinal já concatenado (`p`, `g`, `h`, `LAC`, `nn`). Como `p` vai até ~15 e os outros ficam abaixo de 1, os 4 atributos menores ficavam espremidos perto de 0 e caíam quase todos no mesmo estado.
2. **Bins uniformes (`linspace`).** Muitos dos Q estados ficavam vazios (ex.: Q=32 usava ~13 estados de fato), e as linhas vazias da matriz de transição viravam 1/Q.

A versão testada:

- normaliza cada atributo separadamente para [0,1] antes de concatenar;
- quantiza por quantil (bins de mesma frequência, como no MTF original de Wang & Oates), com valores iguais no mesmo estado;
- limita Q ao tamanho do sinal N (Q acima de N vira Q = N).

## Resultado

Testado no DIS (o dataset em que o MTF ia pior), com ResNet50, 5 folds iguais aos do DIS, em 6 maxL × 12 Q:

- F1 macro médio: **0,379 (antigo) × 0,377 (novo)**; o novo ganhou em 33 das 72 configurações;
- melhor resultado: 0,501 no antigo (128bits mtf 25) × 0,454 no novo (16bits mtf 65);
- a variação do próprio treino é de ~0,05 (no maxL7, configurações com imagens idênticas variaram de 0,298 a 0,347), então as diferenças ficam dentro do ruído.

O gargalo principal do MTF continua sendo o número pequeno de transições para estimar a matriz de Markov (N−1, de 14 a 159), e essa mudança não ataca isso.

## Arquivos

- `build_mtf_quantil.m` e `reshapeMTF_mat_quantil.m`: versão testada. Os nomes foram trocados para não conflitar com as funções oficiais quando `Scripts/` inteiro está no path do MATLAB.
- `teste_mtf_dis.py`: script Python que gerou o MTF novo do DIS em `Datasets/Imagens Histológicas/TESTE_MTF` e treinou a ResNet (resultados em `Scripts/CNN/METRICAS_TESTE_MTF`).
- `comparacao_f1_mtf_DIS.xlsx`: comparação de F1 antigo × novo (resumo, por fold e matriz maxL × Q).
