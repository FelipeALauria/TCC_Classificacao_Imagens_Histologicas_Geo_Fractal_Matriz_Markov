"""Extrator de atributos fractais (percolação, lacunaridade, dimensão fractal)
em GPU -- port de FractalFeatures.m + percChess/percEucl/percManh +
pmrChess/pmrEucl/pmrManh + helpers, gerando o MESMO .mat por imagem/maxL
(mesmos nomes de variável), pra os reshapes do MATLAB continuarem lendo igual.

Como executar: usado pelo orquestrador_fractais.py. Pra uma classe só:
    python fractal_features.py "<pasta da classe>" 7 11 15 25 41 65
"""
import re
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from PIL import Image

LIMIAR_PERCOLACAO = 0.59275
DISTANCIAS = ("Chess", "Eucl", "Manh")


def escolher_device(exigir_gpu=False):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[device] GPU: {torch.cuda.get_device_name(0)}")
    else:
        if exigir_gpu:
            raise RuntimeError(
                "GPU não encontrada pelo PyTorch (torch.cuda.is_available() == False). "
                f"Versão instalada: torch {torch.__version__} -- se terminar em '+cpu', "
                "reinstale o torch com CUDA (ver README).")
        device = torch.device("cpu")
        print(f"[device] AVISO: rodando na CPU (torch {torch.__version__}).")
    return device


# ============================================================
# PARTE CARA (GPU): máscara por caixa, rótulos 4-conexos, histograma
# ============================================================

def _mascaras(patches, centros, r, distancia):
    """patches (3, L, r, r) float32, centros (3, L) -> máscara (L, r, r) bool
    dos pixels da caixa que satisfazem a distância até o pixel central."""
    if distancia == "Chess":
        mask = None
        for c in range(3):
            ok = (patches[c] - centros[c, :, None, None]).abs() <= r
            mask = ok if mask is None else mask & ok
        return mask
    acc = None
    for c in range(3):
        d = patches[c] - centros[c, :, None, None]
        d = d * d if distancia == "Eucl" else d.abs()
        acc = d if acc is None else acc + d
    # Euclidiana: sqrt(s) <= r  <=>  s <= r^2 (inteiros, sem erro de arredondamento)
    return acc <= (r * r if distancia == "Eucl" else r)


def _rotular_4conexo(mask):
    """Equivalente ao bwlabel(box, 4) aplicado em lote: devolve
    (n_componentes, tamanho_maior_componente) por caixa. Propagação do
    menor índice entre vizinhos 4-conexos + pointer jumping até convergir."""
    L, r, _ = mask.shape
    n = r * r
    idx = torch.arange(n, device=mask.device, dtype=torch.int64).view(1, r, r)
    lab = torch.where(mask, idx, n)
    fundo = torch.full((L, 1), n, device=mask.device, dtype=torch.int64)
    while True:
        m = lab.clone()
        m[:, 1:, :] = torch.minimum(m[:, 1:, :], lab[:, :-1, :])
        m[:, :-1, :] = torch.minimum(m[:, :-1, :], lab[:, 1:, :])
        m[:, :, 1:] = torch.minimum(m[:, :, 1:], lab[:, :, :-1])
        m[:, :, :-1] = torch.minimum(m[:, :, :-1], lab[:, :, 1:])
        m = torch.where(mask, m, n).view(L, n)
        m = torch.minimum(m, torch.gather(torch.cat([m, fundo], 1), 1, m))
        m = m.view(L, r, r)
        if torch.equal(m, lab):
            break
        lab = m
    flat = lab.view(L, n)
    n_comp = (flat == idx.view(1, n)).sum(1)
    contagem = torch.zeros((L, n + 1), device=mask.device, dtype=torch.int64)
    contagem.scatter_add_(1, flat, torch.ones_like(flat))
    maior = contagem[:, :n].amax(1)
    return n_comp, maior


def curvas_imagem(img, maxL, device, memoria_gb=1.5):
    """Calcula, pra r = 3:2:maxL e as 3 distâncias, as curvas p/g/h
    (percChess/Eucl/Manh) e a matriz de probabilidade (pmrChess/Eucl/Manh).
    Retorna {dist: (p, g, h, ProbMatrix)} em numpy float64."""
    t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).to(device, torch.float32)
    H, W = t.shape[1:]
    rs = list(range(3, maxL + 1, 2))
    n_max = rs[-1] ** 2
    saida = {d: (np.zeros(len(rs)), np.zeros(len(rs)), np.zeros(len(rs)), np.zeros((n_max, len(rs))))
             for d in DISTANCIAS}
    orcamento = memoria_gb * 1024 ** 3

    for k, r in enumerate(rs):
        n = r * r
        nl, nc = H - r + 1, W - r + 1
        ncaixas = nl * nc
        lim = r // 2
        janelas = t.unfold(1, r, 1).unfold(2, r, 1)  # (3, nl, nc, r, r), sem cópia
        centros = t[:, lim:lim + nl, lim:lim + nc]
        # menor m tal que m / r^2 >= limiar (mesma comparação em double do MATLAB)
        m_limiar = next(m for m in range(n + 1) if m / n >= LIMIAR_PERCOLACAO)
        linhas_por_bloco = max(1, int(orcamento // (80 * n * nc)))

        acum = {d: dict(comp=0, perc=0, maior=0, hist=torch.zeros(n + 1, device=device, dtype=torch.int64))
                for d in DISTANCIAS}
        for y0 in range(0, nl, linhas_por_bloco):
            y1 = min(nl, y0 + linhas_por_bloco)
            # .contiguous(): com bloco de 1 linha o reshape devolve view não contígua e o .view() quebra
            patches = janelas[:, y0:y1].reshape(3, -1, r, r).contiguous()
            cent = centros[:, y0:y1].reshape(3, -1).contiguous()
            for d in DISTANCIAS:
                mask = _mascaras(patches, cent, r, d)
                cnt = mask.sum((1, 2))
                n_comp, maior = _rotular_4conexo(mask)
                a = acum[d]
                a["hist"] += torch.bincount(cnt, minlength=n + 1)
                a["comp"] += int(n_comp.sum())
                a["perc"] += int((cnt >= m_limiar).sum())
                a["maior"] += int(maior.sum())
                del mask
        for d in DISTANCIAS:
            p, g, h, prob = saida[d]
            a = acum[d]
            p[k] = a["comp"] / ncaixas
            g[k] = a["perc"] / ncaixas
            h[k] = (a["maior"] / n) / ncaixas
            prob[:n, k] = a["hist"][1:].double().cpu().numpy() / ncaixas
    return saida


# ============================================================
# PARTE BARATA (CPU): agregados, mesmas fórmulas do MATLAB
# ============================================================

def _trapz(y):
    y = np.asarray(y, dtype=float)
    return float(np.sum((y[1:] + y[:-1]) / 2.0)) if y.size > 1 else 0.0


def _skewness(y):
    """skewness(y) do MATLAB (flag=1, viesada)."""
    y = np.asarray(y, dtype=float)
    d = y - y.mean()
    m2 = np.mean(d ** 2)
    m3 = np.mean(d ** 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(m3 / m2 ** 1.5)


def _razao(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.float64(a) / np.float64(b))


def _max_idx(y):
    i = int(np.argmax(y))
    return float(y[i]), float(i + 1)  # índice 1-based como no MATLAB


def perc_stats(p, g, h):
    """percStatsFromCurve.m"""
    half = int(np.ceil(len(p) / 2))
    s = {}
    for nome, c in (("Cluster", p), ("Perc", g), ("MaxCluster", h)):
        s[f"Area{nome}"] = _trapz(c)
        s[f"Skewness{nome}"] = _skewness(c)
        s[f"AreaRatio{nome}"] = _razao(_trapz(c[half:]), _trapz(c[:half]))
    s["MaxCluster"], s["MaxClusterIndex"] = _max_idx(p)
    s["MaxPerc"], s["MaxPercIndex"] = _max_idx(g)
    s["MaxMaxCluster"], s["MaxMaxClusterIndex"] = _max_idx(h)
    return s


def lacunaridade(prob):
    """lacunarity.m"""
    j = np.arange(1, prob.shape[0] + 1, dtype=float)[:, None]
    m1 = np.sum(j * prob, axis=0)
    m2 = np.sum(j ** 2 * prob, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (m2 - m1 ** 2) / m1 ** 2


def lac_stats(lac):
    """lacStatsFromCurve.m"""
    half = int(np.ceil(len(lac) / 2))
    mx, mi = _max_idx(lac)
    return {"AreaLAC": _trapz(lac), "SkewnessLAC": _skewness(lac),
            "AreaRatioLAC": _razao(_trapz(lac[half:]), _trapz(lac[:half])),
            "MaxLAC": mx, "MaxLACIndex": mi}


def _wls(X, y, w):
    sw = np.sqrt(w)
    return np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]


def robustfit_bisquare(x, y, tune=4.685, iterlim=50):
    """Mesmo IRLS do fitlm(x, y, 'RobustOpts', 'on') (statrobustfit do MATLAB):
    bisquare, ajuste por leverage e escala por MAD. Retorna [intercepto, inclinação]."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    n, p = X.shape
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    Q, _ = np.linalg.qr(X)
    h = np.minimum(0.9999, np.sum(Q * Q, axis=1))
    adj = 1.0 / np.sqrt(1.0 - h)
    tiny_s = 1e-6 * np.std(y, ddof=1)
    if tiny_s == 0:
        tiny_s = 1.0
    D = np.sqrt(np.finfo(float).eps)
    b0 = np.zeros_like(b)
    it = 0
    while it == 0 or np.any(np.abs(b - b0) > D * np.maximum(np.abs(b), np.abs(b0))):
        it += 1
        if it > iterlim:
            break
        radj = (y - X @ b) * adj
        rs = np.sort(np.abs(radj))
        s = np.median(rs[max(1, p) - 1:]) / 0.6745
        u = radj / (max(s, tiny_s) * tune)
        w = (np.abs(u) < 1) * (1 - u ** 2) ** 2
        b0 = b
        b = _wls(X, y, w)
    return b


def dimensao_fractal(prob, r):
    """fractalDimensionFromProbMatrix.m + calcN.m"""
    j = np.arange(1, prob.shape[0] + 1, dtype=float)[:, None]
    nn = np.sum(prob / j, axis=0)
    b = robustfit_bisquare(np.log(r), -np.log(nn))
    return float(b[1]), nn


def atributos_maxL(curvas, maxL):
    """Fatia as curvas (calculadas no maior maxL) e monta o dict de variáveis
    do .mat desse maxL, com os mesmos nomes que o FractalFeatures.m salvava."""
    k = len(range(3, maxL + 1, 2))
    r = np.arange(3, maxL + 1, 2, dtype=float)
    out = {}
    for d in DISTANCIAS:
        p, g, h, prob = curvas[d]
        p, g, h, prob = p[:k], g[:k], h[:k], prob[:, :k]
        out[f"{d}p"], out[f"{d}g"], out[f"{d}h"] = p, g, h
        for nome, v in perc_stats(p, g, h).items():
            out[f"{d}{nome}"] = v
        lac = lacunaridade(prob)
        out[f"{d}LAC"] = lac
        for nome, v in lac_stats(lac).items():
            out[f"{d}{nome}"] = v
        out[f"{d}FD"], out[f"{d}nn"] = dimensao_fractal(prob, r)
    return out


# ============================================================
# LAÇO POR CLASSE (equivalente ao FractalFeatures.m)
# ============================================================

def listar_imagens(source):
    """*.png com número entre parênteses, ordenados pelo número real."""
    imagens = []
    for f in Path(source).glob("*.png"):
        m = re.search(r"\((\d+)\)", f.name)
        if m:
            imagens.append((int(m.group(1)), f))
        else:
            print(f"  [aviso] ignorando arquivo sem número entre parênteses: {f.name}")
    return sorted(imagens)


def ler_rgb(caminho):
    img = Image.open(caminho)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.asarray(img)


def extrair_classe(source, destinos, maxL_list, device, memoria_gb=1.5):
    """destinos: {maxL: pasta}. Pula imagem/maxL cujo .mat já existe."""
    imagens = listar_imagens(source)
    print(f"  Total de imagens encontradas: {len(imagens)}")
    for n, caminho in imagens:
        saidas = {mL: Path(destinos[mL]) / f"{n}_maxL{mL}.mat" for mL in maxL_list}
        faltando = [mL for mL in maxL_list if not saidas[mL].is_file()]
        if not faltando:
            continue
        t0 = time.time()
        print(f"  Processando: {caminho.name}")
        # só precisa ir até o maior maxL que ainda falta (as curvas são fatiáveis)
        curvas = curvas_imagem(ler_rgb(caminho), max(faltando), device, memoria_gb)
        for mL in faltando:
            Path(destinos[mL]).mkdir(parents=True, exist_ok=True)
            sio.savemat(str(saidas[mL]), atributos_maxL(curvas, mL), do_compression=True)
        print(f"    salvo maxL {faltando} em {time.time() - t0:.1f}s")


if __name__ == "__main__":
    pasta = Path(sys.argv[1])
    lista = [int(v) for v in sys.argv[2:]] or [7, 11, 15, 25, 41]
    dev = escolher_device()
    extrair_classe(pasta, {mL: pasta / f"maxL{mL}" for mL in lista}, lista, dev)
