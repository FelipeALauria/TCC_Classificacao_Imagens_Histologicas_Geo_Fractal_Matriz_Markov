import os
import time
from pathlib import Path

from functions import (
    rodar_dataset, salvar_tempo_csv, formatar_tempo, _CONTADOR, rodar_ablacao_softmax, comparar_softmax, pasta_modelo,
    caminho_cache_dataset, descobrir_canais_reshape,
)
from graficos import plotar_comparacao_combinacoes


def rodar_resnet(source, combinacoes_base, combinacoes_ablacao=None, pasta_saida="METRICAS", **kwargs):
    """Roda rodar_dataset (functions.py) pra treinar as combinações de canal
    único em combinacoes_base e, se combinacoes_ablacao for passado, chama
    rodar_ablacao_softmax pra calcular as combinações de 2+ canais por soft
    voting (média das probabilidades já salvas das combinações-base --
    não treina nada de novo). Gera o relatório visual dentro de
    pasta_saida/<DATASET>/ (mesma grafia do dataset, ver pasta_modelo): gráfico
    comparativo entre todas as combinações que rodaram
    (comparacao_combinacoes_<dataset>.png, ver graficos.py) e, com 2+
    combinações, um exemplo de comparar_softmax entre a primeira e a
    última.

    Retorna resumo_combinacoes (mesmo formato de rodar_dataset), já com
    base + ablação combinados.
    """
    source = Path(source)
    pasta_saida = Path(pasta_saida)
    pasta_dataset = pasta_modelo(pasta_saida, source.name)

    t_inicio_dataset = time.perf_counter()
    folds_antes = _CONTADOR["folds_treinados"]

    resumo_combinacoes = rodar_dataset(source, combinacoes_base, pasta_saida=pasta_saida, **kwargs)

    if combinacoes_ablacao:
        kwargs_ablacao = {
            k: kwargs[k] for k in ("n_folds", "nomes_reshape", "size_maxL", "mtf_q") if k in kwargs
        }
        resumo_ablacao = rodar_ablacao_softmax(
            source, combinacoes_ablacao, combinacoes_base, pasta_saida=pasta_saida, **kwargs_ablacao
        )
        resumo_combinacoes.update(resumo_ablacao)

    tempo_dataset = time.perf_counter() - t_inicio_dataset
    folds_treinados = _CONTADOR["folds_treinados"] - folds_antes
    if folds_treinados:
        salvar_tempo_csv(
            "resnet50", source.name, "TOTAL", "dataset", folds_treinados,
            tempo_dataset, tempo_dataset, size_maxL=kwargs.get("size_maxL"), mtf_q=kwargs.get("mtf_q", 8),
            caminho_csv=pasta_saida / "tempos.csv",
        )
    print(f"\nTempo total de execução do dataset {source.name}: {formatar_tempo(tempo_dataset)}")

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
        caminho_predicoes = caminho_cache_dataset(
            pasta_saida, source.name, kwargs.get("size_maxL"), kwargs.get("mtf_q", 8)
        )
        comparacao = comparar_softmax(nomes[0], nomes[-1], caminho_csv=caminho_predicoes)
        comparacao.to_csv(pasta_dataset / f"comparacao_{nomes[0]}_vs_{nomes[-1]}_{source.name}.csv")

    return resumo_combinacoes


if __name__ == "__main__":
    # Uso manual: 1 dataset. Treina as combinações de canal único que ele já
    # suportar (base) e calcula por softmax -- sem retreinar nada -- todas
    # as combinações de 2+ canais possíveis com o que existe em disco
    # (ablação). Pra rodar vários datasets de uma vez, ver orquestrador.py.
    base_dir = Path(os.environ.get("TCC_DATASETS", Path(__file__).resolve().parents[2] / "Datasets" / "Imagens Histológicas"))
    source = base_dir / "CR"

    from itertools import combinations

    canais_reshape = descobrir_canais_reshape(source)
    canais_disponiveis = ["img"] + [c.lower() for c in canais_reshape]

    combinacoes_base = {("img_original" if canal == "img" else canal): [canal] for canal in canais_disponiveis}

    combinacoes_ablacao = {}
    for tamanho in range(2, len(canais_disponiveis) + 1):
        for combo in combinations(canais_disponiveis, tamanho):
            combinacoes_ablacao["+".join(combo)] = list(combo)

    rodar_resnet(
        source, combinacoes_base, combinacoes_ablacao,
        size_maxL="maxL15", rec_plot_size="rp",
    )
