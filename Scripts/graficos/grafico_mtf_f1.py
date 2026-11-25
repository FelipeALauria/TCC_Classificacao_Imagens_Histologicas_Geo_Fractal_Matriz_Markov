from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PASTA_SCRIPTS = Path(__file__).resolve().parent.parent
PASTA_SAIDA = Path(__file__).resolve().parent
CSVS_METRICAS = [
    PASTA_SCRIPTS / "CNN" / "METRICAS" / "metricas.csv",
    PASTA_SCRIPTS / "VisiumTransformers" / "METRICAS" / "metricas.csv",
]

COMBINACAO = "mtf"
N_FOLDS = 5
NOMES_MODELOS = {"resnet50": "ResNet50", "vit_b16": "ViT-B/16", "coatnet": "CoAtNet"}
# NHL fora: só tem reshape em maxL65, entraria só num ponto e distorceria a média.
DATASETS = ["CR-SameRes", "LG", "UCSB", "DIS", "LA"]
DATASETS_POR_MODELO = {modelo: DATASETS for modelo in NOMES_MODELOS}

COR_SUPERFICIE = "#fcfcfb"
COR_TINTA_PRIMARIA = "#0b0b0b"
COR_TINTA_SECUNDARIA = "#52514e"
COR_TINTA_MUTED = "#898781"
COR_GRADE = "#e1e0d9"
COR_EIXO = "#c3c2b7"


def carregar_f1_por_dataset():
    """DataFrame (modelo, dataset, L, Q) -> f1 médio dos folds, só configs completas."""
    df = pd.concat([pd.read_csv(c) for c in CSVS_METRICAS], ignore_index=True)
    df = df[(df["combinacao"] == COMBINACAO) & (df["modelo"].isin(NOMES_MODELOS))
            & (df["fold"].astype(str) != "media")]
    df = df.dropna(subset=["size_maxL", "mtf_q", "f1"])
    df = df.drop_duplicates(subset=["modelo", "dataset", "size_maxL", "mtf_q", "fold"], keep="last")
    df["L"] = df["size_maxL"].str.replace("maxL", "").astype(int)
    df["Q"] = df["mtf_q"].astype(int)

    agrupado = df.groupby(["modelo", "dataset", "L", "Q"])["f1"].agg(["mean", "count"]).reset_index()
    for _, linha in agrupado[agrupado["count"] < N_FOLDS].iterrows():
        print(f"  [aviso] {linha['modelo']} {linha['dataset']} L={linha['L']} Q={linha['Q']}: "
              f"só {linha['count']}/{N_FOLDS} folds -- ignorado")
    return agrupado[agrupado["count"] >= N_FOLDS].rename(columns={"mean": "f1"})


def media_entre_datasets(f1_modelo, datasets, rotulo):
    """Tabela L x Q com F1 médio entre datasets (NaN se faltar algum)."""
    f1_modelo = f1_modelo[f1_modelo["dataset"].isin(datasets)]
    grupos = f1_modelo.groupby(["L", "Q"])
    contagem = grupos["dataset"].nunique()
    media = grupos["f1"].mean()
    for (L, Q), n in contagem.items():
        if n < len(datasets):
            faltam = set(datasets) - set(grupos.get_group((L, Q))["dataset"])
            print(f"  [aviso] {rotulo} L={L} Q={Q}: faltam {sorted(faltam)} -- ponto omitido")
    media[contagem < len(datasets)] = np.nan
    return media.unstack("Q").sort_index()


def plotar(tabela, titulo, subtitulo, caminho_saida):
    tabela = tabela.dropna(how="all").dropna(axis=1, how="all")
    valores_L = tabela.index.tolist()

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=COR_SUPERFICIE)
    cores = plt.cm.viridis(np.linspace(0, 0.85, len(tabela.columns)))
    for cor, Q in zip(cores, tabela.columns):
        bits = int(np.log2(Q))
        ax.plot(valores_L, tabela[Q].values, marker="o", markersize=5, linewidth=2,
                color=cor, label=f"Q = {Q} ({bits} bits)", zorder=3)

    ax.set_xticks(valores_L)
    ax.set_xticklabels([str(L) for L in valores_L], color=COR_TINTA_SECUNDARIA, fontsize=10)
    ax.tick_params(axis="y", labelcolor=COR_TINTA_MUTED, labelsize=9)
    ax.set_xlabel("Tamanho L (maxL)", color=COR_TINTA_SECUNDARIA, fontsize=11)
    ax.set_ylabel("F1 médio (todos os datasets)", color=COR_TINTA_SECUNDARIA, fontsize=11)

    ax.set_facecolor(COR_SUPERFICIE)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=COR_GRADE, linewidth=1)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(COR_EIXO)
    ax.tick_params(axis="both", length=0)

    ax.set_title(titulo, color=COR_TINTA_PRIMARIA, fontsize=13, fontweight="bold", loc="left", pad=22)
    ax.text(0, 1.02, subtitulo, transform=ax.transAxes, color=COR_TINTA_MUTED, fontsize=9.5)
    ax.legend(title="Quantização", frameon=False, fontsize=9, title_fontsize=10,
              loc="center left", bbox_to_anchor=(1.01, 0.5))

    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=200, facecolor=COR_SUPERFICIE, bbox_inches="tight")
    print(f"Gráfico salvo em {caminho_saida}")
    plt.show()


if __name__ == "__main__":
    f1_por_dataset = carregar_f1_por_dataset()

    for modelo, nome in NOMES_MODELOS.items():
        f1_modelo = f1_por_dataset[f1_por_dataset["modelo"] == modelo]
        datasets = DATASETS_POR_MODELO[modelo]
        tabela = media_entre_datasets(f1_modelo, datasets, nome)
        print(f"\n{nome}\n{tabela.round(3).to_string()}\n")
        plotar(tabela, f"F1 do MTF por tamanho L e quantização — {nome}",
               f"Média de {', '.join(datasets)} · {N_FOLDS} folds",
               PASTA_SAIDA / f"f1_mtf_por_L_{modelo}.png")

    # Geral: junta todos os resultados (todo modelo x dataset) e tira a média
    # por (L, Q) -- cada ponto usa o que existir naquele (L, Q).
    filtro = [d in DATASETS_POR_MODELO[m] for m, d in zip(f1_por_dataset["modelo"], f1_por_dataset["dataset"])]
    todos = f1_por_dataset[filtro]
    tabela_geral = todos.groupby(["L", "Q"])["f1"].mean().unstack("Q").sort_index()
    n_resultados = todos.groupby(["L", "Q"]).size().unstack("Q").sort_index()
    print(f"\nGeral\n{tabela_geral.round(3).to_string()}\n")
    print(f"Nº de resultados (modelo x dataset) por ponto\n{n_resultados.fillna(0).astype(int).to_string()}\n")
    plotar(tabela_geral, "F1 do MTF por tamanho L e quantização — todos os modelos",
           f"Todos os resultados de {', '.join(NOMES_MODELOS.values())} · {N_FOLDS} folds",
           PASTA_SAIDA / "f1_mtf_por_L_geral.png")