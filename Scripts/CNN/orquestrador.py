from pathlib import Path

from Scripts_Datasets.Scripts.CNN.functions import descobrir_datasets
from Scripts_Datasets.Scripts.CNN.resnet import rodar_resnet

# ============================================================
# CONFIGURAÇÃO
# ============================================================

BASE_DIR = Path(r"C:/Users/felip/Desktop/Facul/TCC/Scripts_Datasets/Datasets/Imagens Histológicas")

# datasets = toda subpasta de BASE_DIR com pelo menos uma classe com .png
DATASETS = descobrir_datasets(BASE_DIR)

# combinações desejadas nesta rodada (dict nome -> canais em minúsculo).
# Combinação com canal ainda não gerado num dataset é pulada só pra esse
# dataset (interpretar_combinacoes em functions.py).
#
# Ablação 
COMBINACOES_DESEJADAS = {
    "img_original": ["img"],
}

SIZE_MAXL = "maxL25"
MTF_SIZE = "MTF_Q8_N35"
# REC_PLOT_SIZE = "rp"

PASTA_SAIDA = Path(".")


# ============================================================
# LAÇO: um dataset de cada vez, erro isolado por dataset
# ============================================================

def rodar_todos_datasets(nomes_datasets, combinacoes_desejadas, base_dir, **kwargs):
    """Chama rodar_resnet (resnet.py) pra cada nome em nomes_datasets
    (source = base_dir / nome), isolando erro por dataset -- um dataset com
    problema fica registrado no console e não impede os demais.

    Retorna {nome_dataset: resumo_combinacoes} só com os datasets que
    rodaram pelo menos uma combinação.
    """
    resumo_geral = {}
    datasets_com_erro = []

    for nome_dataset in nomes_datasets:
        source = base_dir / nome_dataset
        try:
            resumo = rodar_resnet(source, combinacoes_desejadas, **kwargs)
            if resumo:
                resumo_geral[nome_dataset] = resumo
        except Exception as erro:
            print(f"\n[erro rodando o dataset '{nome_dataset}', pulando pro próximo: {erro}]")
            datasets_com_erro.append(nome_dataset)

    print(f"\n{'#' * 60}\nRESUMO GERAL -- {len(resumo_geral)}/{len(nomes_datasets)} "
          f"datasets concluídos\n{'#' * 60}")
    for nome_dataset, resumo in resumo_geral.items():
        for nome_combinacao, (media, desvio) in resumo.items():
            print(f"{nome_dataset:15s} {nome_combinacao:25s} "
                  f"acc={media['acuracia']:.4f}±{desvio['acuracia']:.4f}  "
                  f"f1={media['f1']:.4f}±{desvio['f1']:.4f}  "
                  f"auc={media['auc']:.4f}±{desvio['auc']:.4f}")
    if datasets_com_erro:
        print(f"\nDatasets com erro (pulados): {datasets_com_erro}")

    return resumo_geral


if __name__ == "__main__":
    print(f"Datasets encontrados em {BASE_DIR}: {DATASETS}")
    print(f"Combinações desejadas: {list(COMBINACOES_DESEJADAS.keys())}")

    rodar_todos_datasets(
        DATASETS, COMBINACOES_DESEJADAS, BASE_DIR,
        pasta_saida=PASTA_SAIDA,
        size_maxL=SIZE_MAXL, MTF_size=MTF_SIZE, rec_plot_size=REC_PLOT_SIZE,
    )