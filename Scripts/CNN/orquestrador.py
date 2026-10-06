from pathlib import Path

from functions import descobrir_datasets
from resnet import rodar_resnet

# ============================================================
# CONFIGURAÇÃO
# ============================================================

BASE_DIR = Path(r"C:\Users\felip\Desktop\Facul\TCC\Scripts_Datasets\Datasets\Imagens Histológicas")

# datasets = toda subpasta de BASE_DIR com pelo menos uma classe com .png
TODOS_DATASETS = descobrir_datasets(BASE_DIR)

# Duas listas fixas -- troque DATASETS_DESEJADOS pra escolher qual rodar.
# CR não é mais usado (decisão de 31/08/2026). LG entra como binário (só 2
# pastas, "1"/"2") e DIS/LA/NHL como multiclasse nativo.
DATASETS_BINARIOS = ["CR-SameRes"] #, "CR-SameRes LG UCSB"
DATASETS_MULTICLASSE = ["DIS", "LA"] # , "DIS LA NHL"

ESCOLHA = 1 # 1-Binários 2-Multiclasse

if ESCOLHA == 1:
    DATASETS_DESEJADOS = DATASETS_BINARIOS
else:
    DATASETS_DESEJADOS = DATASETS_MULTICLASSE

DATASETS = [d for d in DATASETS_DESEJADOS if d in TODOS_DATASETS]

# ------------------------------------------------------------
# Combinações de canal único -- as únicas que treinam de fato uma CNN (via
# rodar_dataset, dentro de rodar_resnet). Rodam uma vez só: combinação x
# fold que já tem resultado salvo em predicoes.csv é pulada automaticamente
# (resume, ver rodar_dataset em functions.py).
# ------------------------------------------------------------
COMBINACOES_BASE = {
    "img_original": ["img"],
    "recplot": ["recplot"],
    "mtf": ["mtf"],
    "gasf": ["gasf"],
    "gadf": ["gadf"],
}

# ------------------------------------------------------------
# Ablação -- combinações de 2+ canais. Não treinam CNN nenhuma: são
# calculadas combinando por soft voting (média das probabilidades softmax)
# os resultados já salvos das combinações de COMBINACOES_BASE acima (ver
# rodar_ablacao_softmax em functions.py). Combinação com canal ainda não    
# gerado num dataset, ou cuja base ainda não rodou, é pulada só pra esse
# dataset.
#
# Descomente/comente as combinações que quiser incluir na rodada:
# ------------------------------------------------------------
COMBINACOES_ABLACAO = {
    # "img+recplot": ["img", "recplot"],
    "img+mtf": ["img", "mtf"],
    # "img+gasf": ["img", "gasf"],
    # "img+gadf": ["img", "gadf"],
    "img+recplot+mtf": ["img", "recplot", "mtf"],
    # "img+recplot+gasf": ["img", "recplot", "gasf"],
    # "img+recplot+gadf": ["img", "recplot", "gadf"],
    "img+mtf+gasf": ["img", "mtf", "gasf"],
    "img+mtf+gadf": ["img", "mtf", "gadf"],
    # "img+gasf+gadf": ["img", "gasf", "gadf"],
    "img+recplot+mtf+gasf": ["img", "recplot", "mtf", "gasf"],
    "img+recplot+mtf+gadf": ["img", "recplot", "mtf", "gadf"],
    # "img+recplot+gasf+gadf": ["img", "recplot", "gasf", "gadf"],
    "img+mtf+gasf+gadf": ["img", "mtf", "gasf", "gadf"],
    "img+recplot+mtf+gasf+gadf": ["img", "recplot", "mtf", "gasf", "gadf"],
    "recplot+mtf": ["recplot", "mtf"],
    # "recplot+gasf": ["recplot", "gasf"],
    # "recplot+gadf": ["recplot", "gadf"],
    "mtf+gasf": ["mtf", "gasf"],
    "mtf+gadf": ["mtf", "gadf"],
    # "gasf+gadf": ["gasf", "gadf"],
    "recplot+mtf+gasf": ["recplot", "mtf", "gasf"],
    "recplot+mtf+gadf": ["recplot", "mtf", "gadf"],
    # "recplot+gasf+gadf": ["recplot", "gasf", "gadf"],
    "mtf+gasf+gadf": ["mtf", "gasf", "gadf"],
    "recplot+mtf+gasf+gadf": ["recplot", "mtf", "gasf", "gadf"],
}

# MTF_size (o "N" no nome do arquivo) não é fixado à mão -- é descoberto
# automaticamente a partir do que já existe em disco pra cada SIZE_MAXL +
# MTF_Q (ver descobrir_mtf_size em functions.py). SIZE_MAXL e MTF_Q agora são
# listas: o orquestrador roda o pipeline inteiro (treino + ablação) uma vez
# pra cada combinação maxL x Q, cobrindo tudo que já foi gerado em disco
# (conferido em Datasets/.../MTF/*/maxL*/ -- ordem crescente, começando em
# maxL7; ver também Scripts/Fractais/orquestradorFractais.m -- MAXL_LIST e
# MTF_Q_LIST). Saída (predicoes/splits) fica separada por maxL e Q
# automaticamente (ver caminho_cache_dataset em functions.py); metricas.csv
# agora grava size_maxL e mtf_q em colunas próprias pra dar pra distinguir
# as linhas depois.
SIZE_MAXL_LIST = ["maxL15", "maxL25", "maxL41"]
MTF_Q_LIST = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
REC_PLOT_SIZE = "rp"

PASTA_SAIDA = Path("METRICAS")


# ============================================================
# LAÇO: um dataset de cada vez, erro isolado por dataset
# ============================================================

def rodar_todos_datasets(nomes_datasets, combinacoes_base, combinacoes_ablacao, base_dir, **kwargs):
    """Chama rodar_resnet (resnet.py) pra cada nome em nomes_datasets
    (source = base_dir / nome): treina combinacoes_base (canal único) e
    calcula combinacoes_ablacao por softmax (sem retreinar), isolando erro
    por dataset -- um dataset com problema fica registrado no console e
    não impede os demais.

    Retorna {nome_dataset: resumo_combinacoes} só com os datasets que
    rodaram pelo menos uma combinação.
    """
    resumo_geral = {}
    datasets_com_erro = []

    for nome_dataset in nomes_datasets:
        source = base_dir / nome_dataset
        try:
            resumo = rodar_resnet(source, combinacoes_base, combinacoes_ablacao, **kwargs)
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
    print(f"Combinações base (treinam CNN): {list(COMBINACOES_BASE.keys())}")
    print(f"Combinações de ablação (softmax, sem retreino): {list(COMBINACOES_ABLACAO.keys())}")
    print(f"maxL a rodar (SIZE_MAXL_LIST): {SIZE_MAXL_LIST}")
    print(f"Bits de MTF a rodar (MTF_Q_LIST): {MTF_Q_LIST}")

    for size_maxL in SIZE_MAXL_LIST:
        for mtf_q in MTF_Q_LIST:
            print(f"\n{'@' * 60}\n{size_maxL} | MTF_Q = {mtf_q}\n{'@' * 60}")
            rodar_todos_datasets(
                DATASETS, COMBINACOES_BASE, COMBINACOES_ABLACAO, BASE_DIR,
                pasta_saida=PASTA_SAIDA,
                size_maxL=size_maxL, mtf_q=mtf_q, rec_plot_size=REC_PLOT_SIZE,
            )
