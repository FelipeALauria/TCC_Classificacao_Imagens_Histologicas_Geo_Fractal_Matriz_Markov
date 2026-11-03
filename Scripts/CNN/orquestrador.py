import csv
import os
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path

from functions import (
    descobrir_datasets, rodar_ablacao_softmax, caminho_cache_dataset,
    _carregar_predicoes_cache, calcular_metricas, salva_metrica_csv,
)
from resnet import rodar_resnet

# ============================================================
# CONFIGURAÇÃO
# ============================================================

# Raiz do repositório (Scripts_Datasets), achada a partir deste arquivo: funciona
# no Windows e no servidor sem editar nada. Pra usar outra pasta de datasets:
#   export TCC_DATASETS="/caminho/para/Imagens Histológicas"
RAIZ_REPO = Path(__file__).resolve().parents[2]
BASE_DIR = Path(os.environ.get("TCC_DATASETS", RAIZ_REPO / "Datasets" / "Imagens Histológicas"))

# datasets = toda subpasta de BASE_DIR com pelo menos uma classe com .png
TODOS_DATASETS = descobrir_datasets(BASE_DIR)

# Duas listas fixas -- troque DATASETS_DESEJADOS pra escolher qual rodar.
# CR não é mais usado (decisão de 31/08/2026). LG entra como binário (só 2
# pastas, "1"/"2") e DIS/LA/NHL como multiclasse nativo.
DATASETS_BINARIOS = ["CR-SameRes", "LG", "UCSB"]
DATASETS_MULTICLASSE = ["DIS", "LA", "NHL"]

ESCOLHA = 2 # 1-Binários 2-Multiclasse

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
SIZE_MAXL_LIST = ["maxL7", "maxL11", "maxL15", "maxL25", "maxL41", "maxL65"]
MTF_Q_LIST = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
REC_PLOT_SIZE = "rp"

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

# ------------------------------------------------------------
# MODO
#   "completo": laço normal maxL x Q x DATASETS (resume pula o que já rodou).
#   "lacunas":  só preenche o que falta, em todos os datasets (binários +
#               multiclasse), em 3 passos:
#     1. grava em metricas.csv os folds que já estão em predicoes_*.csv mas
#        sumiram do metricas.csv (Q32 em maxL15/25/41) -- sem treinar;
#     2. treina só as configs de LACUNAS_TREINO (fold faltando);
#     3. roda a ablação por softmax em todo o grid (calcula img+recplot,
#        img+recplot+gasf e img+recplot+gadf onde faltam) -- sem treinar.
# ------------------------------------------------------------
MODO = "lacunas"

# (dataset, size_maxL, mtf_q) com fold de canal único faltando
LACUNAS_TREINO = [
    ("UCSB", "maxL7", 16384),  # mtf: só 4 de 5 folds
]


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
            registrar_traceback()
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


def _escanear_predicoes(caminho_pred):
    """Uma leitura de predicoes_*.csv: {combinacao: {folds}} e o conjunto de
    (combinacao, fold) com softmax NaN (treino que divergiu)."""
    folds, com_nan = {}, set()
    with open(caminho_pred, newline="", encoding="utf-8") as arquivo:
        for linha in csv.DictReader(arquivo):
            chave = (linha["combinacao"], int(linha["fold"]))
            folds.setdefault(chave[0], set()).add(chave[1])
            if "nan" in linha["y_prob"].lower():
                com_nan.add(chave)
    return folds, com_nan


def _remover_folds(caminho_pred, chaves):
    """Reescreve predicoes_*.csv sem os (combinacao, fold) de chaves -- backup
    em .bak_nan. Faz o resume treinar esses folds de novo."""
    caminho_pred = Path(caminho_pred)
    shutil.copy2(caminho_pred, caminho_pred.with_name(caminho_pred.name + ".bak_nan"))
    with open(caminho_pred, newline="", encoding="utf-8") as arquivo:
        leitor = csv.DictReader(arquivo)
        campos = leitor.fieldnames
        linhas = [l for l in leitor if (l["combinacao"], int(l["fold"])) not in chaves]
    with open(caminho_pred, "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(linhas)


def completar_metricas_de_predicoes(nomes_datasets, combinacoes, n_folds=5):
    """Grava em metricas.csv os folds que existem em predicoes_*.csv mas não
    têm linha em metricas.csv (ex. metricas.csv sobrescrito num merge do
    git), recalculando as métricas a partir do softmax salvo -- não treina.
    Folds com softmax NaN são tirados do predicoes (backup .bak_nan) pra
    serem retreinados."""
    caminho_metricas = PASTA_SAIDA / "metricas.csv"
    norm_q = lambda q: str(int(float(q))) if q not in ("", None) else ""
    ja_salvos = set()
    with open(caminho_metricas, newline="", encoding="utf-8") as arquivo:
        for linha in csv.DictReader(arquivo):
            ja_salvos.add((linha["dataset"], linha["size_maxL"], norm_q(linha["mtf_q"]),
                           linha["combinacao"], linha["fold"]))

    n_gravados = 0
    for nome_dataset in nomes_datasets:
        for size_maxL in SIZE_MAXL_LIST:
            for mtf_q in MTF_Q_LIST:
                caminho_pred = caminho_cache_dataset(PASTA_SAIDA, nome_dataset, size_maxL, mtf_q)
                if not caminho_pred.exists():
                    continue
                folds_salvos, com_nan = _escanear_predicoes(caminho_pred)
                if com_nan:
                    print(f"[lacunas] {caminho_pred.name}: removendo folds com NaN {sorted(com_nan)}")
                    _remover_folds(caminho_pred, com_nan)
                config = (nome_dataset, size_maxL, str(mtf_q))
                for nome_combinacao in combinacoes:
                    folds = folds_salvos.get(nome_combinacao, set()) - {f for c, f in com_nan if c == nome_combinacao}
                    faltando = [f for f in folds if (*config, nome_combinacao, str(f)) not in ja_salvos]
                    falta_media = len(folds) == n_folds and (*config, nome_combinacao, "media") not in ja_salvos
                    if not faltando and not falta_media:
                        continue
                    cache = _carregar_predicoes_cache(caminho_pred, nome_combinacao)
                    modelo = "resnet50" if nome_combinacao in COMBINACOES_BASE else "ensemble_softmax"
                    metricas = {f: calcular_metricas(d["y_true"], d["y_pred"], d["y_prob_matriz"],
                                                     d["y_prob_matriz"].shape[1])
                                for f, d in cache.items()}
                    for f in sorted(faltando):
                        salva_metrica_csv(modelo, nome_dataset, nome_combinacao, fold=f, mtf_q=mtf_q,
                                          size_maxL=size_maxL, caminho_csv=caminho_metricas, **metricas[f])
                        n_gravados += 1
                    if falta_media:
                        media = {k: sum(m[k] for m in metricas.values()) / n_folds for k in metricas[1]}
                        salva_metrica_csv(modelo, nome_dataset, nome_combinacao, fold="media", mtf_q=mtf_q,
                                          size_maxL=size_maxL, caminho_csv=caminho_metricas, **media)
                        n_gravados += 1
    print(f"\n[lacunas] {n_gravados} linhas recuperadas de predicoes_*.csv pro metricas.csv")


def rodar_lacunas(n_folds=5):
    datasets = [d for d in DATASETS_BINARIOS + DATASETS_MULTICLASSE if d in TODOS_DATASETS]
    print(f"[lacunas] datasets: {datasets}")

    # 1. metricas.csv a partir das predições já salvas (+ limpa folds NaN)
    completar_metricas_de_predicoes(datasets, {**COMBINACOES_BASE, **COMBINACOES_ABLACAO}, n_folds)

    # 2. treino só das configs com fold faltando (resume treina só o que falta)
    for nome_dataset, size_maxL, mtf_q in LACUNAS_TREINO:
        print(f"\n{'@' * 60}\n[lacunas] treino {nome_dataset} | {size_maxL} | Q{mtf_q}\n{'@' * 60}")
        rodar_todos_datasets(
            [nome_dataset], COMBINACOES_BASE, COMBINACOES_ABLACAO, BASE_DIR,
            pasta_saida=PASTA_SAIDA, size_maxL=size_maxL, mtf_q=mtf_q, rec_plot_size=REC_PLOT_SIZE,
        )

    # 3. ablação só das combinações que faltam, em todo o grid
    for nome_dataset in datasets:
        for size_maxL in SIZE_MAXL_LIST:
            for mtf_q in MTF_Q_LIST:
                caminho_pred = caminho_cache_dataset(PASTA_SAIDA, nome_dataset, size_maxL, mtf_q)
                if not caminho_pred.exists():
                    continue
                folds_salvos, _ = _escanear_predicoes(caminho_pred)
                faltando = {nome: canais for nome, canais in COMBINACOES_ABLACAO.items()
                            if len(folds_salvos.get(nome, ())) < n_folds}
                if not faltando:
                    continue
                print(f"\n[lacunas] ablação {nome_dataset} | {size_maxL} | Q{mtf_q}: {list(faltando)}")
                try:
                    rodar_ablacao_softmax(BASE_DIR / nome_dataset, faltando, COMBINACOES_BASE,
                                          n_folds=n_folds, pasta_saida=PASTA_SAIDA,
                                          size_maxL=size_maxL, mtf_q=mtf_q)
                except Exception as erro:
                    print(f"[erro na ablação de {nome_dataset} {size_maxL} Q{mtf_q}: {erro}]")
                    registrar_traceback()


if __name__ == "__main__":
    configurar_log(PASTA_SAIDA / "logs")

if __name__ == "__main__" and MODO == "lacunas":
    rodar_lacunas()

elif __name__ == "__main__":
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
