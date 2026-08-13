"""Gráficos de análise pós-rodada -- lêem os CSVs já salvos em METRICAS/
(metricas.csv e predicoes_<dataset>.csv) e não dependem de rodar nada de
novo. Pode chamar a qualquer momento depois de uma rodada do orquestrador
(ou do resnet.py sozinho), inclusive pra dados de rodadas antigas.

Uso típico:

    from graficos import plotar_boxplot_folds, plotar_matriz_confusao

    plotar_boxplot_folds(
        "METRICAS/metricas.csv", "CR-SameRes",
        "METRICAS/CR-SameRes/boxplot_folds_CR-SameRes.png",
    )
    plotar_matriz_confusao(
        "METRICAS/CR-SameRes/predicoes_CR-SameRes.csv", "CR-SameRes", "img_original",
        "METRICAS/CR-SameRes/matriz_confusao_CR-SameRes_img_original.png",
        nomes_classes=["Benign", "Malignant"],
    )
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from functions import descobrir_classes


# ============================================================
# ESTILO COMPARTILHADO -- mesma paleta/cores em todos os gráficos do
# projeto (usado também por resnet.py)
# ============================================================
# Paleta colorblind-safe: Okabe, M. & Ito, K. (2008), "Color Universal
# Design (CUD)" -- a paleta categórica mais citada em publicações
# científicas pra daltonismo. 8 tons (o 8º, cinza, é a extensão de uso
# comum da paleta original quando precisa de mais de 7 categorias em vez
# do preto puro, que fica reservado pro texto/eixos).

PALETA_COMBINACOES = [
    "#E69F00",  # laranja
    "#56B4E9",  # azul claro
    "#009E73",  # verde azulado
    "#F0E442",  # amarelo
    "#0072B2",  # azul
    "#D55E00",  # vermelho (vermelion)
    "#CC79A7",  # roxo avermelhado
    "#999999",  # cinza
]
NOMES_METRICAS_PT = {
    "acuracia": "Acurácia", "precisao": "Precisão", "recall": "Recall",
    "f1": "F1", "auc": "AUC",
}
COR_SUPERFICIE = "#fcfcfb"
COR_TINTA_PRIMARIA = "#0b0b0b"
COR_TINTA_SECUNDARIA = "#52514e"
COR_TINTA_MUTED = "#898781"
COR_GRADE = "#e1e0d9"
COR_EIXO = "#c3c2b7"

_TICKS_PROPORCAO = [0, 0.2, 0.4, 0.6, 0.8, 1.0]


def _cores_combinacoes(n_combos, nome_dataset=""):
    """Lista de n_combos cores (paleta Okabe-Ito, repetindo com aviso -- 1
    só vez -- se passar de 8 combinações no mesmo gráfico)."""
    if n_combos > len(PALETA_COMBINACOES):
        print(f"    [aviso: {n_combos} combinações no gráfico de {nome_dataset}, mais que as "
              f"{len(PALETA_COMBINACOES)} cores da paleta -- cores repetem a partir da "
              f"{len(PALETA_COMBINACOES) + 1}a combinação]")
    return [PALETA_COMBINACOES[i % len(PALETA_COMBINACOES)] for i in range(n_combos)]


def _estilizar_eixo(ax):
    """Aplica os specs comuns de eixo -- grade horizontal fina, sem
    moldura em cima/direita, sem traço de tick."""
    ax.set_facecolor(COR_SUPERFICIE)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=COR_GRADE, linewidth=1)
    ax.xaxis.grid(False)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(COR_EIXO)
        ax.spines[lado].set_linewidth(1)
    ax.tick_params(axis="both", length=0)


# ============================================================
# GRÁFICO 1 -- comparação de métricas por combinação (bar chart)
# ============================================================

def plotar_comparacao_combinacoes(resumo_combinacoes, nome_dataset, caminho_saida):
    """Gráfico de barras agrupadas por métrica (eixo x = Acurácia/Precisão/
    Recall/F1/AUC, eixo y = 0 a 1.0), uma cor fixa por combinação de canais
    (média ± desvio padrão entre folds). Salva em caminho_saida (cria a
    pasta se precisar)."""
    caminho_saida = Path(caminho_saida)
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)

    nomes = list(resumo_combinacoes.keys())
    metricas_grafico = ["acuracia", "precisao", "recall", "f1", "auc"]
    n_combos = len(nomes)
    n_metricas = len(metricas_grafico)

    x = np.arange(n_metricas)
    largura_grupo = 0.8
    largura_barra = largura_grupo / n_combos * 0.82  # deixa um vão entre as barras do grupo
    cores = _cores_combinacoes(n_combos, nome_dataset)

    fig, ax = plt.subplots(figsize=(max(9, 5 * 1.6 + n_combos * 0.6), 5.5), facecolor=COR_SUPERFICIE)

    for i, nome_combinacao in enumerate(nomes):
        media, desvio = resumo_combinacoes[nome_combinacao]
        medias = [media[met] for met in metricas_grafico]
        desvios = [desvio[met] for met in metricas_grafico]
        offset = (i - (n_combos - 1) / 2) * (largura_grupo / n_combos)
        ax.bar(
            x + offset, medias, largura_barra, yerr=desvios, capsize=2.5,
            color=cores[i], edgecolor=COR_SUPERFICIE, linewidth=1.2,
            error_kw={"ecolor": COR_TINTA_SECUNDARIA, "elinewidth": 1, "capthick": 1},
            label=nome_combinacao, zorder=3,
        )

    ax.set_ylim(0, 1.0)
    ax.set_yticks(_TICKS_PROPORCAO)
    ax.set_yticklabels([f"{v:.0%}" for v in _TICKS_PROPORCAO], color=COR_TINTA_MUTED, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([NOMES_METRICAS_PT[met] for met in metricas_grafico],
                        color=COR_TINTA_SECUNDARIA, fontsize=10.5)
    _estilizar_eixo(ax)

    ax.set_title(f"Desempenho por combinação de canais — {nome_dataset}",
                 color=COR_TINTA_PRIMARIA, fontsize=13, fontweight="bold", loc="left", pad=26)

    # Com 1 só combinação não faz sentido legenda (cor não distingue nada) --
    # o nome dela entra no subtítulo em vez disso.
    if n_combos == 1:
        subtitulo = f"ResNet50 · combinação: {nomes[0]} · média ± desvio padrão entre folds"
    else:
        subtitulo = "ResNet50 · média ± desvio padrão entre folds"
    ax.text(0, 1.05, subtitulo, transform=ax.transAxes, color=COR_TINTA_SECUNDARIA, fontsize=9.5, ha="left")

    if n_combos >= 2:
        ncol = min(n_combos, 4)
        linhas_legenda = -(-n_combos // ncol)  # teto da divisão
        offset_y = -0.14 - 0.06 * (linhas_legenda - 1)
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, offset_y),
            ncol=ncol, frameon=False, fontsize=9.5, labelcolor=COR_TINTA_SECUNDARIA,
            handlelength=1.2, handleheight=1.2, columnspacing=1.4,
        )

    fig.savefig(caminho_saida, dpi=200, facecolor=COR_SUPERFICIE, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# GRÁFICO 2 -- boxplot da distribuição entre folds ("outliers")
# ============================================================

def plotar_boxplot_folds(caminho_metricas_csv, nome_dataset, caminho_saida, metricas=None):
    """Boxplot da distribuição de cada métrica entre os folds do K-fold,
    uma caixa por combinação de canais -- os pontos fora da caixa (além de
    1.5x o intervalo interquartil, padrão do matplotlib) são os folds
    atípicos (outliers) daquela combinação.

    Lê metricas.csv (salva_metrica_csv em functions.py), filtra pelo
    dataset e pelas linhas de fold individual (exclui a linha 'media').
    metricas: subconjunto de acuracia/precisao/recall/f1/auc a plotar
    (default: as 5).
    """
    metricas = metricas or ["acuracia", "precisao", "recall", "f1", "auc"]
    df = pd.read_csv(caminho_metricas_csv)
    df = df[(df["dataset"] == nome_dataset) & (df["fold"] != "media")]
    if df.empty:
        raise ValueError(
            f"Nenhuma linha de fold individual pra '{nome_dataset}' em {caminho_metricas_csv} "
            "(rodou pelo menos uma vez esse dataset?)."
        )

    combinacoes = list(dict.fromkeys(df["combinacao"]))  # ordem de aparição, sem duplicar
    n_combos = len(combinacoes)
    n_metricas = len(metricas)

    cores = _cores_combinacoes(n_combos, nome_dataset)

    fig, eixos = plt.subplots(
        1, n_metricas, figsize=(3.3 * n_metricas, 5), facecolor=COR_SUPERFICIE, sharey=True,
    )
    eixos = np.atleast_1d(eixos)

    for ax, met in zip(eixos, metricas):
        dados = [df.loc[df["combinacao"] == c, met].dropna().to_numpy() for c in combinacoes]
        caixas = ax.boxplot(
            dados, patch_artist=True, widths=0.55,
            medianprops={"color": COR_TINTA_PRIMARIA, "linewidth": 1.5},
            whiskerprops={"color": COR_EIXO, "linewidth": 1},
            capprops={"color": COR_EIXO, "linewidth": 1},
            flierprops={"marker": "o", "markersize": 5, "markerfacecolor": COR_TINTA_SECUNDARIA,
                        "markeredgecolor": COR_SUPERFICIE, "alpha": 0.9},
        )
        for i, caixa in enumerate(caixas["boxes"]):
            caixa.set_facecolor(cores[i])
            caixa.set_edgecolor(COR_SUPERFICIE)
            caixa.set_linewidth(1.2)
            caixa.set_alpha(0.9)

        ax.set_title(NOMES_METRICAS_PT[met], color=COR_TINTA_PRIMARIA, fontsize=11, fontweight="bold")
        ax.set_xticks(range(1, n_combos + 1))
        ax.set_xticklabels(combinacoes, rotation=35, ha="right", color=COR_TINTA_SECUNDARIA, fontsize=8.5)
        ax.set_ylim(0, 1.0)
        ax.set_yticks(_TICKS_PROPORCAO)
        _estilizar_eixo(ax)

    eixos[0].set_yticklabels([f"{v:.0%}" for v in _TICKS_PROPORCAO], color=COR_TINTA_MUTED, fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.8])
    fig.suptitle(f"Distribuição das métricas entre os folds — {nome_dataset}",
                 color=COR_TINTA_PRIMARIA, fontsize=13, fontweight="bold", x=0.01, y=0.99, ha="left")
    fig.text(0.01, 0.9, "ResNet50 · cada caixa resume os folds de 1 combinação · "
                        "pontos fora da caixa = fold atípico (outlier)",
              color=COR_TINTA_SECUNDARIA, fontsize=9.5, ha="left")

    caminho_saida = Path(caminho_saida)
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho_saida, dpi=200, facecolor=COR_SUPERFICIE, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# GRÁFICO 3 -- matriz de confusão (out-of-fold, todas as classes)
# ============================================================

def plotar_matriz_confusao(caminho_predicoes_csv, nome_dataset, combinacao, caminho_saida,
                            nomes_classes=None, source=None):
    """Matriz de confusão (contagens) agregando as predições de TODOS os
    folds pra uma combinação -- como é k-fold CV, cada imagem cai no
    conjunto de teste de exatamente 1 fold, então a soma cobre o dataset
    inteiro sem vazamento (matriz "out-of-fold").

    Lê predicoes_<dataset>.csv (salvar_predicoes_csv em functions.py).
    nomes_classes: rótulos na ordem 0..N-1 (ex. ["Benign", "Malignant"]).
    Se não passar nomes_classes mas passar source (pasta do dataset),
    descobre os nomes via descobrir_classes(source). Sem nenhum dos dois,
    usa "Classe 0", "Classe 1"...
    """
    df = pd.read_csv(caminho_predicoes_csv)
    df = df[df["combinacao"] == combinacao]
    if df.empty:
        raise ValueError(f"Nenhuma predição pra combinação '{combinacao}' em {caminho_predicoes_csv}.")

    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()
    n_classes = int(max(y_true.max(), y_pred.max())) + 1
    matriz = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    if nomes_classes is None and source is not None:
        rotulos_dict = descobrir_classes(source)
        nomes_classes = [nome for nome, _ in sorted(rotulos_dict.items(), key=lambda kv: kv[1])]
    rotulos = nomes_classes if nomes_classes else [f"Classe {i}" for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=(1.35 * n_classes + 3, 1.35 * n_classes + 2.2), facecolor=COR_SUPERFICIE)
    ax.set_facecolor(COR_SUPERFICIE)

    # Sequential (magnitude) = 1 hue só, claro->escuro -- azul, o padrão
    # do método pra "quanto maior, mais escuro" (nunca arco-íris).
    im = ax.imshow(matriz, cmap="Blues", vmin=0)

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(rotulos, color=COR_TINTA_SECUNDARIA, fontsize=10)
    ax.set_yticklabels(rotulos, color=COR_TINTA_SECUNDARIA, fontsize=10)
    ax.set_xlabel("Predito", color=COR_TINTA_SECUNDARIA, fontsize=10.5)
    ax.set_ylabel("Real", color=COR_TINTA_SECUNDARIA, fontsize=10.5)
    ax.tick_params(axis="both", length=0)

    limiar = matriz.max() / 2 if matriz.max() > 0 else 0
    for i in range(n_classes):
        for j in range(n_classes):
            valor = matriz[i, j]
            cor_texto = "#ffffff" if valor > limiar else COR_TINTA_PRIMARIA
            ax.text(j, i, str(valor), ha="center", va="center", color=cor_texto, fontsize=12, fontweight="bold")

    for lado in ax.spines.values():
        lado.set_visible(False)
    ax.set_xticks(np.arange(-0.5, n_classes, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_classes, 1), minor=True)
    ax.grid(which="minor", color=COR_SUPERFICIE, linewidth=2.5)
    ax.tick_params(which="minor", length=0)

    barra = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    barra.ax.tick_params(labelsize=8.5, color=COR_EIXO, length=0)
    barra.set_label("Nº de imagens", color=COR_TINTA_SECUNDARIA, fontsize=9)
    barra.outline.set_visible(False)

    fig.suptitle(f"Matriz de confusão — {nome_dataset} ({combinacao})",
                 color=COR_TINTA_PRIMARIA, fontsize=12.5, fontweight="bold", x=0.02, ha="left")
    fig.text(0.02, 0.92, "Agregado de todos os folds (out-of-fold — cada imagem entra no teste 1x)",
              color=COR_TINTA_SECUNDARIA, fontsize=9, ha="left")

    caminho_saida = Path(caminho_saida)
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(caminho_saida, dpi=200, facecolor=COR_SUPERFICIE, bbox_inches="tight")
    plt.close(fig)
