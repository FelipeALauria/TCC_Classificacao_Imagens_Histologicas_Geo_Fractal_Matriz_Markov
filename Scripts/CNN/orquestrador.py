from pathlib import Path

from functions import descobrir_datasets
from resnet import rodar_resnet

# ============================================================
# CONFIGURAÇÃO
# ============================================================

BASE_DIR = Path(r"C:\Users\felip\Desktop\Facul\TCC\Scripts_Datasets\Datasets\Imagens Histológicas")

# datasets = toda subpasta de BASE_DIR com pelo menos uma classe com .png
TODOS_DATASETS = descobrir_datasets(BASE_DIR)

# Rodada atual: só os datasets já binários (Benign/Malignant) no disco.
# Fora por serem multiclasse nativo: DIS (4), LA (4), NHL (3, sem classe
# saudável -- não dá pra binarizar). LG também tem só 2 pastas ("1"/"2")
# mas fica de fora até confirmar com o orientador o que os rótulos significam.
DATASETS_BINARIOS = ["CR"]
DATASETS = [d for d in DATASETS_BINARIOS if d in TODOS_DATASETS]

# combinações desejadas nesta rodada (dict nome -> canais em minúsculo).
# Combinação com canal ainda não gerado num dataset é pulada só pra esse
# dataset (interpretar_combinacoes em functions.py).
#
# Ablação -- descomentar as combinações que quiser incluir na rodada:
COMBINACOES_DESEJADAS = {
    "img_original": ["img"],
    "recplot": ["recplot"],
    "mtf": ["mtf"],
    "gasf": ["gasf"],
    "gadf": ["gadf"],
    "img+recplot": ["img", "recplot"],
    "img+mtf": ["img", "mtf"],
    "img+gasf": ["img", "gasf"],
    "img+gadf": ["img", "gadf"],
    "img+recplot+mtf": ["img", "recplot", "mtf"],
    "img+recplot+gasf": ["img", "recplot", "gasf"],
    "img+recplot+gadf": ["img", "recplot", "gadf"],
    "img+mtf+gasf": ["img", "mtf", "gasf"],
    "img+mtf+gadf": ["img", "mtf", "gadf"],
    "img+gasf+gadf": ["img", "gasf", "gadf"],
    "img+recplot+mtf+gasf": ["img", "recplot", "mtf", "gasf"],
    "img+recplot+mtf+gadf": ["img", "recplot", "mtf", "gadf"],
    "img+recplot+gasf+gadf": ["img", "recplot", "gasf", "gadf"],
    "img+mtf+gasf+gadf": ["img", "mtf", "gasf", "gadf"],
    "img+recplot+mtf+gasf+gadf": ["img", "recplot", "mtf", "gasf", "gadf"],
    "recplot+mtf": ["recplot", "mtf"],
    "recplot+gasf": ["recplot", "gasf"],
    "recplot+gadf": ["recplot", "gadf"],
    "mtf+gasf": ["mtf", "gasf"],
    "mtf+gadf": ["mtf", "gadf"],
    "gasf+gadf": ["gasf", "gadf"],
    "recplot+mtf+gasf": ["recplot", "mtf", "gasf"],
    "recplot+mtf+gadf": ["recplot", "mtf", "gadf"],
    "recplot+gasf+gadf": ["recplot", "gasf", "gadf"],
    "mtf+gasf+gadf": ["mtf", "gasf", "gadf"],
    "recplot+mtf+gasf+gadf": ["recplot", "mtf", "gasf", "gadf"],
}

SIZE_MAXL = "maxL25"
# MTF_size (o "N" no nome do arquivo) não é mais fixado à mão -- é descoberto
# automaticamente a partir do que já existe em disco pra SIZE_MAXL + MTF_Q
# (ver descobrir_mtf_size em functions.py). Só o Q é configurável aqui.
MTF_Q = 8
REC_PLOT_SIZE = "rp"

PASTA_SAIDA = Path("METRICAS")


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
        size_maxL=SIZE_MAXL, mtf_q=MTF_Q, rec_plot_size=REC_PLOT_SIZE,
    )