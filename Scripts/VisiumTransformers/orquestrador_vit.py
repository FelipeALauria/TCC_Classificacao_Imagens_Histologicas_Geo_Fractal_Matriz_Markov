"""Orquestrador dos Vision Transformers (ViT-B/16 e CoAtNet).

Como executar (precisa do timm: pip install timm):
    cd Scripts/VisiumTransformers
    python orquestrador_vit.py

Saída em Scripts/VisiumTransformers/METRICAS/ (separada da CNN). Os folds
são os mesmos da ResNet: se o splits_kfold de um dataset/config já existir
em Scripts/CNN/METRICAS/, ele é reaproveitado (só leitura).
"""
import os
import sys
import traceback
from datetime import datetime
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


# ============================================================
# LOG: tudo que aparece no console vai pra METRICAS/logs/execucao_<data>.log;
# erros/avisos/pulos (e tracebacks) também vão pra METRICAS/logs/erros.log
# ============================================================

_PADROES_ERRO = ("[erro", "[pulando", "pulando dataset", "AVISO", "Traceback", "Error")
_ARQ_ERROS = None


class _Tee:
    """Escreve no console e no log da execução; linhas de erro também em erros.log.
    tudo_erro=True (stderr): toda linha vai pra erros.log."""

    def __init__(self, original, arq_execucao, tudo_erro=False):
        self.original, self.arq_execucao, self.tudo_erro = original, arq_execucao, tudo_erro
        self.buffer = ""

    def write(self, texto):
        self.original.write(texto)
        self.arq_execucao.write(texto)
        self.buffer += texto
        *linhas, self.buffer = self.buffer.split("\n")
        for linha in linhas:
            if self.tudo_erro or any(p in linha for p in _PADROES_ERRO):
                _ARQ_ERROS.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {linha}\n")

    def flush(self):
        self.original.flush()
        self.arq_execucao.flush()


def configurar_log(pasta_logs):
    global _ARQ_ERROS
    pasta_logs = Path(pasta_logs)
    pasta_logs.mkdir(parents=True, exist_ok=True)
    caminho_execucao = pasta_logs / f"execucao_{datetime.now():%Y%m%d_%H%M%S}.log"
    arq_execucao = open(caminho_execucao, "a", encoding="utf-8", buffering=1)
    _ARQ_ERROS = open(pasta_logs / "erros.log", "a", encoding="utf-8", buffering=1)
    _ARQ_ERROS.write(f"\n===== execução {datetime.now():%Y-%m-%d %H:%M:%S} ({Path(__file__).name}) =====\n")
    sys.stdout = _Tee(sys.stdout, arq_execucao)
    sys.stderr = _Tee(sys.stderr, arq_execucao, tudo_erro=True)
    print(f"[log] console em {caminho_execucao} | erros em {pasta_logs / 'erros.log'}")


def registrar_traceback():
    """Chamar dentro de um except: grava o traceback completo em erros.log."""
    if _ARQ_ERROS:
        _ARQ_ERROS.write(traceback.format_exc())

TODOS_DATASETS = descobrir_datasets(BASE_DIR)

DATASETS_BINARIOS = ["CR-SameRes", "LG", "UCSB"]
DATASETS_MULTICLASSE = ["DIS", "LA", "NHL"]
DATASETS_LACUNAS = ["LG", "UCSB", "NHL"]  # sem nenhum resultado ainda (03/11/2026)

ESCOLHA = 3  # 1-Binários 2-Multiclasse 3-Lacunas
DATASETS_DESEJADOS = {1: DATASETS_BINARIOS, 2: DATASETS_MULTICLASSE, 3: DATASETS_LACUNAS}[ESCOLHA]
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
# dataset com maxL próprio (substitui SIZE_MAXL_LIST): NHL só tem reshape em maxL65
SIZE_MAXL_POR_DATASET = {"NHL": ["maxL65"]}
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
            registrar_traceback()
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
    configurar_log(PASTA_SAIDA / "logs")

if __name__ == "__main__":
    print(f"Datasets: {DATASETS}")
    print(f"Modelos: {list(MODELOS)}")
    maxl_por_dataset = {d: SIZE_MAXL_POR_DATASET.get(d, SIZE_MAXL_LIST) for d in DATASETS}
    todos_maxl = list(dict.fromkeys(m for lista in maxl_por_dataset.values() for m in lista))
    print(f"maxL: {maxl_por_dataset} | MTF_Q: {MTF_Q_LIST}")

    for modelo, hiperparametros in MODELOS.items():
        for size_maxL in todos_maxl:
            datasets_maxl = [d for d in DATASETS if size_maxL in maxl_por_dataset[d]]
            for mtf_q in MTF_Q_LIST:
                print(f"\n{'@' * 60}\n{modelo} | {size_maxL} | MTF_Q = {mtf_q} | {datasets_maxl}\n{'@' * 60}")
                rodar_todos_datasets(
                    datasets_maxl, modelo, hiperparametros,
                    size_maxL=size_maxL, mtf_q=mtf_q, rec_plot_size=REC_PLOT_SIZE,
                )
