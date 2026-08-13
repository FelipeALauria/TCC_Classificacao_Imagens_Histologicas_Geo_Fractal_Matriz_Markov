from pathlib import Path

from functions import rodar_dataset, comparar_softmax, pasta_modelo
from graficos import plotar_comparacao_combinacoes


def rodar_resnet(source, combinacoes_desejadas, pasta_saida="METRICAS", **kwargs):
    """Roda rodar_dataset (functions.py) pra um dataset e gera o relatório
    visual dentro de pasta_saida/<DATASET>/ (mesma grafia do dataset, ver
    pasta_modelo): gráfico comparativo entre combinações
    (comparacao_combinacoes_<dataset>.png, ver graficos.py) e, com 2+
    combinações, um exemplo de comparar_softmax entre a primeira e a
    última.

    Retorna resumo_combinacoes (mesmo formato de rodar_dataset).
    """
    source = Path(source)
    pasta_saida = Path(pasta_saida)
    pasta_dataset = pasta_modelo(pasta_saida, source.name)

    resumo_combinacoes = rodar_dataset(source, combinacoes_desejadas, pasta_saida=pasta_saida, **kwargs)

    if not resumo_combinacoes:
        print(f"\nNenhuma combinação rodou com sucesso para {source.name} -- sem gráfico a gerar.")
        return resumo_combinacoes

    print(f"\n{'=' * 60}\nRESUMO -- {source.name}\n{'=' * 60}")
    for nome_combinacao, (media, desvio) in resumo_combinacoes.items():
        print(f"{nome_combinacao:35s} acc={media['acuracia']:.4f}±{desvio['acuracia']:.4f}  "
              f"f1={media['f1']:.4f}±{desvio['f1']:.4f}  auc={media['auc']:.4f}±{desvio['auc']:.4f}")

    plotar_comparacao_combinacoes(
        resumo_combinacoes, source.name, pasta_dataset / f"comparacao_combinacoes_{source.name}.png"
    )

    nomes = list(resumo_combinacoes.keys())
    if len(nomes) >= 2:
        caminho_predicoes = pasta_dataset / f"predicoes_{source.name}.csv"
        comparacao = comparar_softmax(nomes[0], nomes[-1], caminho_csv=caminho_predicoes)
        comparacao.to_csv(pasta_dataset / f"comparacao_{nomes[0]}_vs_{nomes[-1]}_{source.name}.csv")

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
        size_maxL="maxL15", rec_plot_size="rp",
    )