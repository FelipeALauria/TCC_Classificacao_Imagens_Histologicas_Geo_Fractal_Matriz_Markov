"""Orquestrador dos Vision Transformers (ViT-B/16 e CoAtNet).

Como executar (precisa do timm: pip install timm):
    cd Scripts/VisiumTransformers
    python orquestrador_vit.py

Saída em Scripts/VisiumTransformers/METRICAS/ (separada da CNN). Os folds
são os mesmos da ResNet: se o splits_kfold de um dataset/config já existir
em Scripts/CNN/METRICAS/, ele é reaproveitado (só leitura).
"""
import os
from pathlib import Path

from functions_vit import descobrir_datasets
from vit import rodar_vit

# ============================================================
# CONFIGURAÇÃO
# ============================================================

# Raiz do repositório (Scripts_Datasets), achada a partir deste arquivo: funciona
# no Windows e no servidor sem editar nada. Pra usar outra pasta de datasets:
#   export TCC_DATASETS="/caminho/para/Imagens Histológicas"
RAIZ_REPO = Path(__file__).resolve().parents[2]
BASE_DIR = Path(os.environ.get("TCC_DATASETS", RAIZ_REPO / "Datasets" / "Imagens Histológicas"))
PASTA_SAIDA = Path(__file__).resolve().parent / "METRICAS"

TODOS_DATASETS = descobrir_datasets(BASE_DIR)

DATASETS_BINARIOS = ["CR-SameRes"]  # , "LG", "UCSB"
DATASETS_MULTICLASSE = ["DIS", "LA"]  # , "NHL"

ESCOLHA = 2  # 1-Binários 2-Multiclasse
DATASETS_DESEJADOS = DATASETS_BINARIOS if ESCOLHA == 1 else DATASETS_MULTICLASSE
DATASETS = [d for d in DATASETS_DESEJADOS if d in TODOS_DATASETS]

# ------------------------------------------------------------
# Modelos e hiperparâmetros (repassados pra rodar_dataset).
# Comente um modelo pra não rodá-lo. Se faltar memória na GPU: baixe
# batch_size ou troque o coatnet pra "coatnet_0_rw_224.sw_in1k" em
# MODELOS_TIMM (functions_vit.py).
# ------------------------------------------------------------
MODELOS = {
    "vit_b16": dict(lr=2e-5, n_epocas=20, batch_size=8),
    "coatnet": dict(lr=5e-5, n_epocas=20, batch_size=8),
}

# Canal único -- os únicos que treinam de fato (resume por fold).
COMBINACOES_BASE = {
    "img_original": ["img"],
    "recplot": ["recplot"],
    "mtf": ["mtf"],
    "gasf": ["gasf"],
    "gadf": ["gadf"],
}

# Ablação -- soft voting das predições da base do MESMO modelo, sem retreino.
COMBINACOES_ABLACAO = {
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

SIZE_MAXL_LIST = ["maxL15", "maxL25", "maxL41"]
MTF_Q_LIST = [128, 256, 512]
REC_PLOT_SIZE = "rp"


# ============================================================
# LAÇO: modelo x maxL x Q x dataset, erro isolado por dataset
# ============================================================

def rodar_todos_datasets(nomes_datasets, modelo, hiperparametros, **kwargs):
    resumo_geral = {}
    datasets_com_erro = []

    for nome_dataset in nomes_datasets:
        try:
            resumo = rodar_vit(
                BASE_DIR / nome_dataset, modelo, COMBINACOES_BASE, COMBINACOES_ABLACAO,
                pasta_saida=PASTA_SAIDA, **hiperparametros, **kwargs,
            )
            if resumo:
                resumo_geral[nome_dataset] = resumo
        except Exception as erro:
            print(f"\n[erro rodando '{nome_dataset}' com {modelo}, pulando pro próximo: {erro}]")
            datasets_com_erro.append(nome_dataset)

    print(f"\n{'#' * 60}\nRESUMO GERAL ({modelo}) -- {len(resumo_geral)}/{len(nomes_datasets)} "
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
    print(f"Datasets: {DATASETS}")
    print(f"Modelos: {list(MODELOS)}")
    print(f"maxL: {SIZE_MAXL_LIST} | MTF_Q: {MTF_Q_LIST}")

    for modelo, hiperparametros in MODELOS.items():
        for size_maxL in SIZE_MAXL_LIST:
            for mtf_q in MTF_Q_LIST:
                print(f"\n{'@' * 60}\n{modelo} | {size_maxL} | MTF_Q = {mtf_q}\n{'@' * 60}")
                rodar_todos_datasets(
                    DATASETS, modelo, hiperparametros,
                    size_maxL=size_maxL, mtf_q=mtf_q, rec_plot_size=REC_PLOT_SIZE,
                )
