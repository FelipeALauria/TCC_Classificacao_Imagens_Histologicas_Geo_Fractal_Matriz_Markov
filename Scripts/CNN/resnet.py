from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from Scripts_Datasets.Scripts.CNN.functions import (
    rodar_dataset,
    comparar_softmax,
)


def rodar_resnet(source, combinacoes_desejadas, pasta_saida=".", **kwargs):
    """Roda rodar_dataset (functions.py) pra um dataset e gera o relatório
    visual: gráfico comparativo entre combinações
    (comparacao_combinacoes_<dataset>.png) e, com 2+ combinações, um
    exemplo de comparar_softmax entre a primeira e a última.

    Retorna resumo_combinacoes (mesmo formato de rodar_dataset).
    """
    source = Path(source)
    pasta_saida = Path(pasta_saida)

    resumo_combinacoes = rodar_dataset(source, combinacoes_desejadas, pasta_saida=pasta_saida, **kwargs)

    if not resumo_combinacoes:
        print(f"\nNenhuma combinação rodou com sucesso para {source.name} -- sem gráfico a gerar.")
        return resumo_combinacoes

    print(f"\n{'=' * 60}\nRESUMO -- {source.name}\n{'=' * 60}")
    for nome_combinacao, (media, desvio) in resumo_combinacoes.items():
        print(f"{nome_combinacao:35s} acc={media['acuracia']:.4f}±{desvio['acuracia']:.4f}  "
              f"f1={media['f1']:.4f}±{desvio['f1']:.4f}  auc={media['auc']:.4f}±{desvio['auc']:.4f}")

    nomes = list(resumo_combinacoes.keys())
    metricas_grafico = ["acuracia", "precisao", "recall", "f1", "auc"]
    x = np.arange(len(nomes))
    largura = 0.15

    fig, ax = plt.subplots(figsize=(max(10, len(nomes) * 2), 6))
    for i, met in enumerate(metricas_grafico):
        medias = [resumo_combinacoes[n][0][met] for n in nomes]
        desvios = [resumo_combinacoes[n][1][met] for n in nomes]
        ax.bar(x + i * largura, medias, largura, yerr=desvios, capsize=3, label=met)

    ax.set_xticks(x + largura * (len(metricas_grafico) - 1) / 2)
    ax.set_xticklabels(nomes, rotation=30, ha="right")
    ax.set_ylabel("Valor da métrica (média ± desvio, folds)")
    ax.set_title(f"Desempenho da ResNet50 por combinação de canais — {source.name}")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(pasta_saida / f"comparacao_combinacoes_{source.name}.png", dpi=150)
    plt.close(fig)

    if len(nomes) >= 2:
        caminho_predicoes = pasta_saida / f"predicoes_{source.name}.csv"
        comparacao = comparar_softmax(nomes[0], nomes[-1], caminho_csv=caminho_predicoes)
        comparacao.to_csv(pasta_saida / f"comparacao_{nomes[0]}_vs_{nomes[-1]}_{source.name}.csv")

    return resumo_combinacoes


if __name__ == "__main__":
    # Uso manual: 1 dataset, todas as combinações que ele já suportar
    # (auto-geradas do que existe em disco). Pra rodar vários datasets de
    # uma vez com um conjunto específico de combinações, ver orquestrador.py.
    source = Path(r"C:/Users/felip/Desktop/Facul/TCC/Scripts_Datasets/Datasets/Imagens Histológicas/CR")

    from Scripts_Datasets.Scripts.CNN.functions import descobrir_canais_reshape

    canais_reshape = descobrir_canais_reshape(source)
    canais_disponiveis = ["img"] + [c.lower() for c in canais_reshape]

    combinacoes = {"img_original": ["img"]}
    for canal in canais_disponiveis[1:]:
        combinacoes[f"img+{canal}"] = ["img", canal]
    if len(canais_disponiveis) > 2:
        combinacoes["img+" + "+".join(canais_disponiveis[1:])] = canais_disponiveis

    rodar_resnet(
        source, combinacoes,
        size_maxL="maxL15", MTF_size="MTF_Q8_N35", rec_plot_size="rp",
    )