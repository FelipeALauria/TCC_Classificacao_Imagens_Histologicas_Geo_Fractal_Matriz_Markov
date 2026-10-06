from pathlib import Path

from functions_vit import (
    rodar_dataset, rodar_ablacao_softmax, comparar_softmax, pasta_modelo,
    caminho_cache_dataset, descobrir_canais_reshape,
)
from graficos_vit import plotar_comparacao_combinacoes


def rodar_vit(source, modelo, combinacoes_base, combinacoes_ablacao=None, pasta_saida="METRICAS", **kwargs):
    """Treina combinacoes_base (canal único) com o modelo ("vit_b16" ou
    "coatnet") e, se combinacoes_ablacao for passado, calcula as combinações
    de 2+ canais por soft voting das predições desse mesmo modelo (sem
    retreinar). Gera o gráfico comparativo em pasta_saida/<DATASET>/.

    Retorna resumo_combinacoes (base + ablação).
    """
    source = Path(source)
    pasta_saida = Path(pasta_saida)
    pasta_dataset = pasta_modelo(pasta_saida, source.name)

    resumo_combinacoes = rodar_dataset(source, combinacoes_base, modelo, pasta_saida=pasta_saida, **kwargs)

    if combinacoes_ablacao:
        kwargs_ablacao = {
            k: kwargs[k] for k in ("n_folds", "nomes_reshape", "size_maxL", "mtf_q") if k in kwargs
        }
        resumo_ablacao = rodar_ablacao_softmax(
            source, combinacoes_ablacao, combinacoes_base, modelo, pasta_saida=pasta_saida, **kwargs_ablacao
        )
        resumo_combinacoes.update(resumo_ablacao)

    if not resumo_combinacoes:
        print(f"\nNenhuma combinação rodou com sucesso para {source.name} ({modelo}) -- sem gráfico a gerar.")
        return resumo_combinacoes

    print(f"\n{'=' * 60}\nRESUMO -- {source.name} | {modelo}\n{'=' * 60}")
    for nome_combinacao, (media, desvio) in resumo_combinacoes.items():
        print(f"{nome_combinacao:35s} acc={media['acuracia']:.4f}±{desvio['acuracia']:.4f}  "
              f"f1={media['f1']:.4f}±{desvio['f1']:.4f}  auc={media['auc']:.4f}±{desvio['auc']:.4f}")

    configuracao = f"{modelo}_Q{kwargs.get('mtf_q', 8)}_{kwargs.get('size_maxL') or 'auto'}"
    plotar_comparacao_combinacoes(
        resumo_combinacoes, f"{source.name} ({modelo})",
        pasta_dataset / f"comparacao_combinacoes_{source.name}_{configuracao}.png",
    )

    nomes = list(resumo_combinacoes.keys())
    if len(nomes) >= 2:
        caminho_predicoes = caminho_cache_dataset(
            pasta_saida, source.name, kwargs.get("size_maxL"), kwargs.get("mtf_q", 8), modelo
        )
        comparacao = comparar_softmax(nomes[0], nomes[-1], caminho_csv=caminho_predicoes)
        comparacao.to_csv(pasta_dataset / f"comparacao_{nomes[0]}_vs_{nomes[-1]}_{source.name}_{configuracao}.csv")

    return resumo_combinacoes


if __name__ == "__main__":
    # Uso manual: 1 dataset, 1 modelo. Pra rodar tudo, ver orquestrador_vit.py.
    from itertools import combinations

    source = Path(r"C:/Users/felip/Desktop/Facul/TCC/Scripts_Datasets/Datasets/Imagens Histológicas/LG")
    modelo = "vit_b16"  # ou "coatnet"

    canais_reshape = descobrir_canais_reshape(source)
    canais_disponiveis = ["img"] + [c.lower() for c in canais_reshape]

    combinacoes_base = {("img_original" if canal == "img" else canal): [canal] for canal in canais_disponiveis}
    combinacoes_ablacao = {}
    for tamanho in range(2, len(canais_disponiveis) + 1):
        for combo in combinations(canais_disponiveis, tamanho):
            combinacoes_ablacao["+".join(combo)] = list(combo)

    rodar_vit(
        source, modelo, combinacoes_base, combinacoes_ablacao,
        pasta_saida=Path(__file__).resolve().parent / "METRICAS",
        size_maxL="maxL15", mtf_q=32, rec_plot_size="rp", lr=2e-5,
    )
