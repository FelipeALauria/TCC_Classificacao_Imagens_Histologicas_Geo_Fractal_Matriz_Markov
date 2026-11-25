"""Roda de uma vez tudo que falta nos resultados (ResNet-50, ViT-B/16, CoAtNet).
Script temporário: apagar depois que as lacunas forem preenchidas.

Como executar:
    cd Scripts
    python orquestrador_lacunas_delete.py

Lê os metricas.csv de CNN/ e VisiumTransformers/, acha cada (modelo, dataset,
maxL, Q) com menos de 31 combinações x 5 folds e roda só esses -- o resume dos
orquestradores originais pula fold/combinação que já existe. Config sem reshape
gerado em disco (ex. NHL fora do maxL65, que depende do MATLAB) é pulada com aviso.
DRY_RUN = True só lista o que vai rodar.
"""
import csv
import sys
from pathlib import Path

DRY_RUN = False

PASTA_SCRIPTS = Path(__file__).resolve().parent
PASTA_CNN = PASTA_SCRIPTS / "CNN"
PASTA_VIT = PASTA_SCRIPTS / "VisiumTransformers"
sys.path[:0] = [str(PASTA_VIT), str(PASTA_CNN)]

import orquestrador as orq_cnn          # noqa: E402
import orquestrador_vit as orq_vit      # noqa: E402

BASE_DIR = orq_vit.BASE_DIR
DATASETS = ["CR-SameRes", "DIS", "LA", "LG", "UCSB", "NHL"]
N_FOLDS = 5
CANAIS_RESHAPE = ["RecPlot", "MTF", "GASF", "GADF"]

# modelo -> (modelo da ablação no csv, pasta METRICAS, grid maxL, grid Q)
# ViT/CoAtNet: maxL15-65 (sem 7 e 11) com os 12 Q do MTF, igual à ResNet
GRADES = {
    "vit_b16": ("ensemble_softmax_vit_b16", PASTA_VIT / "METRICAS",
                ["maxL15", "maxL25", "maxL41", "maxL65"], orq_cnn.MTF_Q_LIST),
    "coatnet": ("ensemble_softmax_coatnet", PASTA_VIT / "METRICAS",
                ["maxL15", "maxL25", "maxL41", "maxL65"], orq_cnn.MTF_Q_LIST),
    "resnet50": ("ensemble_softmax", PASTA_CNN / "METRICAS",
                 orq_cnn.SIZE_MAXL_LIST, orq_cnn.MTF_Q_LIST),
}
COMBINACOES = set(orq_cnn.COMBINACOES_BASE) | set(orq_cnn.COMBINACOES_ABLACAO)


def combinacoes_completas(caminho_metricas, modelos):
    """{(modelo_base, dataset, maxL, Q): nº de combinações com os 5 folds}."""
    folds = {}
    with open(caminho_metricas, newline="", encoding="utf-8") as arquivo:
        for linha in csv.DictReader(arquivo):
            if linha["modelo"] not in modelos or linha["combinacao"] not in COMBINACOES:
                continue
            if linha["fold"] == "media" or not linha["size_maxL"] or not linha["mtf_q"]:
                continue
            chave = (modelos[linha["modelo"]], linha["dataset"], linha["size_maxL"],
                     int(float(linha["mtf_q"])), linha["combinacao"])
            folds.setdefault(chave, set()).add(linha["fold"])
    contagem = {}
    for (*config, _), f in folds.items():
        if len(f) >= N_FOLDS:
            contagem[tuple(config)] = contagem.get(tuple(config), 0) + 1
    return contagem


def reshapes_prontos(dataset, size_maxL, mtf_q):
    """Todos os 4 reshapes existem em disco pra esse maxL (e o MTF pra esse Q)?"""
    for canal in CANAIS_RESHAPE:
        padrao = f"*/{size_maxL}/*_MTF_Q{mtf_q}_N*.png" if canal == "MTF" else f"*/{size_maxL}/*.png"
        if next((BASE_DIR / dataset / canal).glob(padrao), None) is None:
            return False
    return True


def listar_lacunas():
    contagens = {}
    for caminho in {g[1] / "metricas.csv" for g in GRADES.values()}:
        mapa = {m: m for m, g in GRADES.items() if g[1] / "metricas.csv" == caminho}
        mapa.update({g[0]: m for m, g in GRADES.items() if g[1] / "metricas.csv" == caminho})
        contagens.update(combinacoes_completas(caminho, mapa))

    rodar, sem_reshape = [], []
    for modelo, (_, _, lista_maxl, lista_q) in GRADES.items():
        for size_maxL in lista_maxl: 
            for mtf_q in lista_q:
                for dataset in DATASETS:
                    if contagens.get((modelo, dataset, size_maxL, mtf_q), 0) >= len(COMBINACOES):
                        continue
                    destino = rodar if reshapes_prontos(dataset, size_maxL, mtf_q) else sem_reshape
                    destino.append((modelo, dataset, size_maxL, mtf_q))
    return rodar, sem_reshape


def resumir(lacunas):
    grupos = {}
    for modelo, dataset, size_maxL, mtf_q in lacunas:
        grupos.setdefault((modelo, dataset, size_maxL), []).append(mtf_q)
    for (modelo, dataset, size_maxL), qs in grupos.items():
        print(f"  {modelo:9s} {dataset:11s} {size_maxL:7s} Q={qs}")


if __name__ == "__main__":
    orq_vit.configurar_log(PASTA_SCRIPTS / "logs_lacunas_delete")
    rodar, sem_reshape = listar_lacunas()

    print(f"\n[lacunas] {len(rodar)} configs a rodar:")
    resumir(rodar)
    print(f"\n[pulando] {len(sem_reshape)} configs sem reshape em disco (gerar no MATLAB antes):")
    resumir(sem_reshape)

    if DRY_RUN:
        sys.exit(0)

    for i, (modelo, dataset, size_maxL, mtf_q) in enumerate(rodar, 1):
        print(f"\n{'@' * 60}\n[{i}/{len(rodar)}] {modelo} | {dataset} | {size_maxL} | Q{mtf_q}\n{'@' * 60}")
        kwargs = dict(size_maxL=size_maxL, mtf_q=mtf_q, rec_plot_size="rp")
        if modelo == "resnet50":
            orq_cnn.rodar_todos_datasets(
                [dataset], orq_cnn.COMBINACOES_BASE, orq_cnn.COMBINACOES_ABLACAO, orq_cnn.BASE_DIR,
                pasta_saida=orq_cnn.PASTA_SAIDA, **kwargs,
            )
        else:
            orq_vit.rodar_todos_datasets([dataset], modelo, orq_vit.MODELOS[modelo], **kwargs)

    rodar_depois, _ = listar_lacunas()
    print(f"\n[lacunas] fim. Ainda faltando (com reshape pronto): {len(rodar_depois)}")
    resumir(rodar_depois)