"""Teste do MTF novo (normalização por atributo + bins por quantil) no DIS.

1. Gera o MTF novo a partir dos .mat do DIS em Datasets/.../TESTE_MTF (mesmo
   layout de dataset: TESTE_MTF/<classe>/*.png + TESTE_MTF/MTF/<classe>/<maxL>/),
   copiando as imagens originais. Mesma lógica do reshapeMTF_mat.m/build_mtf.m.
2. Treina só a combinação "mtf" (ResNet50) com os MESMOS folds do DIS e
   compara com o F1 do MTF antigo em METRICAS/metricas.csv.

Como executar (VS Code, com o .venv selecionado): abrir este arquivo e Run
(ou `python Scripts/CNN/teste_mtf_dis.py`). Saída em Scripts/CNN/METRICAS_TESTE_MTF/.
"""
import csv
import shutil
from pathlib import Path

import numpy as np
import scipy.io as sio
from PIL import Image

from functions import descobrir_mtf_size, montar_classes_dict, _universo_ids_labels
from resnet import rodar_resnet

# ============================================================
# CONFIGURAÇÃO
# ============================================================
RAIZ_REPO = Path(__file__).resolve().parents[2]
BASE_DIR = RAIZ_REPO / "Datasets" / "Imagens Histológicas"
ORIGEM = BASE_DIR / "DIS"
DESTINO = BASE_DIR / "TESTE_MTF"

MAXL_LIST = [7, 11, 15, 25, 41, 65]
Q_LIST = MTF_Q_LIST = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

GERAR_MTF = True
TREINAR = True

PASTA_CNN = Path(__file__).resolve().parent
PASTA_SAIDA = PASTA_CNN / "METRICAS_TESTE_MTF"
METRICAS_ANTIGAS = PASTA_CNN / "METRICAS" / "metricas.csv"

RESHAPES = {"GASF", "GADF", "MTF", "RecPlot"}
ATRIBUTOS = ("p", "g", "h", "LAC", "nn")
DISTANCIAS = ("Chess", "Eucl", "Manh")  # R, G, B


# ============================================================
# MTF (port de reshapeMTF_mat.m + build_mtf.m)
# ============================================================
def norm01(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    amplitude = v.max() - v.min()
    return np.zeros_like(v) if amplitude == 0 else (v - v.min()) / amplitude


def build_mtf(x, Q):
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    Q = min(Q, n)

    # posto médio pra empates (mesmo estado pra valores iguais)
    r = np.empty(n)
    r[np.argsort(x, kind="stable")] = np.arange(1, n + 1)
    _, ic = np.unique(x, return_inverse=True)
    r = (np.bincount(ic, weights=r) / np.bincount(ic))[ic]
    q = np.ceil(r * Q / n).astype(int) - 1  # base 0

    M = np.zeros((Q, Q))
    np.add.at(M, (q[:-1], q[1:]), 1)
    somas = M.sum(axis=1, keepdims=True)
    vazias = somas[:, 0] == 0
    M[vazias, :] = 1.0 / Q
    somas[vazias] = 1
    M = M / somas

    return M[np.ix_(q, q)]


def imresize_nearest(img, tamanho=224):
    """Igual ao imresize(..., 'nearest') do MATLAB pra ampliação."""
    n = img.shape[0]
    escala = tamanho / n
    u = np.arange(1, tamanho + 1) / escala + 0.5 * (1 - 1 / escala)
    idx = np.clip(np.floor(u + 0.5).astype(int), 1, n) - 1
    return img[np.ix_(idx, idx)]


def gerar_mtf_classe(pasta_mat, pasta_png, maxL, q_list):
    m = 5 * len(range(3, maxL + 1, 2))
    pasta_png.mkdir(parents=True, exist_ok=True)
    arquivos = sorted(pasta_mat.glob(f"*_maxL{maxL}.mat"), key=lambda p: int(p.name.split("_")[0]))
    gerados = 0
    for arq in arquivos:
        n = int(arq.name.split("_")[0])
        nome = lambda q: pasta_png / f"{n}_MTF_Q{q}_N{m}.png"
        faltando = [q for q in q_list if not nome(q).exists()]
        if not faltando:
            continue
        dados = sio.loadmat(arq)
        sinais = [np.concatenate([norm01(dados[f"{d}{a}"]) for a in ATRIBUTOS]) for d in DISTANCIAS]
        for q in faltando:
            img = np.stack([build_mtf(s, q) for s in sinais], axis=-1)
            img = np.clip(np.floor(img * 255 + 0.5), 0, 255).astype(np.uint8)
            Image.fromarray(imresize_nearest(img)).save(nome(q))
            gerados += 1
    return len(arquivos), gerados


def gerar_tudo():
    classes = sorted(p for p in ORIGEM.iterdir() if p.is_dir() and p.name not in RESHAPES)
    for classe in classes:
        destino_img = DESTINO / classe.name
        destino_img.mkdir(parents=True, exist_ok=True)
        for png in classe.glob("*.png"):
            if not (destino_img / png.name).exists():
                shutil.copy2(png, destino_img / png.name)

        for maxL in MAXL_LIST:
            pasta_mat = classe / f"maxL{maxL}"
            if not pasta_mat.is_dir():
                print(f"[pulando] {pasta_mat} não existe")
                continue
            n_mat, n_png = gerar_mtf_classe(pasta_mat, DESTINO / "MTF" / classe.name / f"maxL{maxL}", maxL, Q_LIST)
            print(f"{classe.name:10s} maxL{maxL}: {n_mat} .mat -> {n_png} PNG novos")


# ============================================================
# TREINO + COMPARAÇÃO
# ============================================================
def copiar_splits_dis(size_maxL, q):
    """Usa os mesmos folds do DIS, se os ids baterem (comparação justa)."""
    origem = PASTA_CNN / "METRICAS" / "DIS" / f"splits_kfold_DIS_Q{q}_{size_maxL}.json"
    destino = PASTA_SAIDA / "TESTE_MTF" / f"splits_kfold_TESTE_MTF_Q{q}_{size_maxL}.json"
    if destino.exists() or not origem.exists():
        return
    mtf_size = descobrir_mtf_size(DESTINO, size_maxL, mtf_q=q)
    classes = montar_classes_dict(DESTINO, ["MTF"], None, mtf_size, size_maxL)
    ids, _ = _universo_ids_labels(classes, ["MTF"], None, mtf_size)
    import json
    with open(origem, encoding="utf-8") as f:
        ids_dis = set(json.load(f)["ids"])
    if ids_dis == set(ids):
        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origem, destino)
    else:
        print(f"[aviso] ids do DIS diferem em {size_maxL} Q{q} -- gerando folds novos")


def f1_antigo(size_maxL, q):
    if not METRICAS_ANTIGAS.exists():
        return None
    with open(METRICAS_ANTIGAS, newline="", encoding="utf-8") as f:
        for linha in csv.DictReader(f):
            if (linha["dataset"] == "DIS" and linha["combinacao"] == "mtf" and linha["fold"] == "media"
                    and linha["size_maxL"] == size_maxL and linha["mtf_q"]
                    and int(float(linha["mtf_q"])) == q):
                return float(linha["f1"])
    return None


def treinar_tudo():
    resultados = []
    for maxL in MAXL_LIST:
        size_maxL = f"maxL{maxL}"
        for q in Q_LIST:
            print(f"\n{'@' * 60}\nTESTE_MTF | {size_maxL} | Q{q}\n{'@' * 60}")
            copiar_splits_dis(size_maxL, q)
            resumo = rodar_resnet(DESTINO, {"mtf": ["mtf"]}, None, pasta_saida=PASTA_SAIDA,
                                  size_maxL=size_maxL, mtf_q=q, rec_plot_size="rp")
            if "mtf" in resumo:
                media, desvio = resumo["mtf"]
                resultados.append((size_maxL, q, f1_antigo(size_maxL, q), media["f1"], desvio["f1"]))

    print(f"\n{'#' * 60}\nMTF antigo x novo -- DIS (F1 médio)\n{'#' * 60}")
    for size_maxL, q, antigo, novo, desvio in resultados:
        txt_antigo = f"{antigo:.4f}" if antigo is not None else "  --  "
        ganho = f"{novo - antigo:+.4f}" if antigo is not None else ""
        print(f"{size_maxL:7s} Q{q:<5d} antigo={txt_antigo}  novo={novo:.4f}±{desvio:.4f}  {ganho}")


if __name__ == "__main__":
    if GERAR_MTF:
        gerar_tudo()
    if TREINAR:
        treinar_tudo()
