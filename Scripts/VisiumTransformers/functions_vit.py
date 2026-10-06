"""Funções do pipeline de Vision Transformers (ViT-B/16 e CoAtNet via timm).
Cópia adaptada de Scripts/CNN/functions.py -- a versão da CNN não é alterada.
"""
import math
import re
import csv
import json
from datetime import date
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def indexar_por_numero(diretorio, padrao_glob="*.png"):
    """Indexa os arquivos de um diretório pelo número REAL da imagem.

    Extrai o ÚLTIMO grupo de dígitos do nome, depois de remover o sufixo
    literal já conhecido de padrao_glob (a parte fixa depois do "*") --
    não o primeiro grupo de dígitos do nome inteiro. Isso importa porque
    o nome de classe pode ser só um número (datasets LA/LG, classes
    "1".."4") e apareceria ANTES do número real da imagem: "Class 1 -
    (100).png" -> 100 (não 1, que é só o nome da classe); "1_100_rp.png"
    (RecPlot da classe "1") -> 100; "100_MTF_Q8_N60.png" -> 100 (aqui o
    prefixo já É o número, sem nome de classe). padrao_glob filtra quais
    arquivos entram. Retorna {} se o diretório não existir."""
    diretorio = Path(diretorio)
    sufixo = padrao_glob[1:] if padrao_glob.startswith("*") else ""
    padrao_numero = re.compile(r"\d+")
    mapa = {}
    if not diretorio.is_dir():
        return mapa
    for arquivo in sorted(diretorio.glob(padrao_glob)):
        nome = arquivo.name
        resto = nome[:-len(sufixo)] if sufixo and nome.endswith(sufixo) else arquivo.stem
        numeros = padrao_numero.findall(resto)
        if not numeros:
            continue
        indice = int(numeros[-1])
        mapa[indice] = arquivo
    return mapa


def obter_arquivo(mapa_indices, indice, contexto=""):
    """"GET": busca o caminho já indexado para um índice específico."""
    caminho = mapa_indices.get(indice)
    if caminho is None:
        sufixo = f" ({contexto})" if contexto else ""
        raise FileNotFoundError(f"Nenhum arquivo indexado para o índice {indice}{sufixo}")
    return caminho


# ============================================================
# DESCOBERTA DE DATASET (classes + canais reshape)
# ============================================================
# Layout em disco: source/<Classe>/*.png (imagem original) e
# source/<Canal>/<Classe>/<size_maxL>/*.png (RecPlot/MTF/GASF/GADF).

NOMES_RESHAPE_PADRAO = ("RecPlot", "MTF", "GASF", "GADF")


def _chave_ordenacao_natural(nome):
    """Ordena "1", "2", "10" numericamente e nomes não numéricos
    (Benign, healthy...) alfabeticamente."""
    return (0, int(nome)) if nome.isdigit() else (1, nome.lower())


def descobrir_classes(source, nomes_reshape=NOMES_RESHAPE_PADRAO):
    """Descobre as classes de um dataset a partir das subpastas de source
    (excluindo os nomes de reshape, case-insensitive). Rótulo (0, 1, 2...)
    em ordem natural do nome da pasta."""
    source = Path(source)
    nomes_reshape_lower = {n.lower() for n in nomes_reshape}
    pastas_classe = sorted(
        (p for p in source.iterdir() if p.is_dir() and p.name.lower() not in nomes_reshape_lower),
        key=lambda p: _chave_ordenacao_natural(p.name),
    )
    if not pastas_classe:
        raise FileNotFoundError(f"Nenhuma subpasta de classe encontrada em {source}")
    return {p.name: idx for idx, p in enumerate(pastas_classe)}


def descobrir_canais_reshape(source, nomes_reshape=NOMES_RESHAPE_PADRAO):
    """Retorna, dentre nomes_reshape, os que já existem como subpasta de
    source nesse dataset."""
    source = Path(source)
    return [nome for nome in nomes_reshape if (source / nome).is_dir()]


def montar_classes_dict(source, canais_reshape, rec_plot_size=None, MTF_size=None, size_maxL=None):
    """Monta o dicionário de configuração por classe: img_dir
    (source/<Classe>) e dirs_reshape (source/<Canal>/<Classe>/<size_maxL>)
    para cada classe descoberta em descobrir_classes."""
    source = Path(source)
    rotulos = descobrir_classes(source)
    classes = {}
    for nome_classe, label in rotulos.items():
        classes[nome_classe] = {
            "label": label,
            "img_dir": source / nome_classe,
            "dirs_reshape": {
                canal: source / canal / nome_classe / size_maxL for canal in canais_reshape
            },
        }
    return classes


PADROES_GLOB_RESHAPE = {
    "RecPlot": lambda rec_plot_size, MTF_size: f"*_{rec_plot_size}.png",
    "MTF": lambda rec_plot_size, MTF_size: f"*_{MTF_size}.png",
    "GASF": lambda rec_plot_size, MTF_size: "*_gasf.png",
    "GADF": lambda rec_plot_size, MTF_size: "*_gadf.png",
}


def carregar_classe(configuracao_classe, canais_reshape, rec_plot_size=None, MTF_size=None):
    """Carrega imagem original + canais reshape de uma classe, todos em RGB.
    Usa só os índices presentes em todos os canais pedidos (interseção).

    canais_reshape: subconjunto de NOMES_RESHAPE_PADRAO, ou [] pra só imagem
    original. Retorna (dados, labels, ids):
    - dados: {"img": [...], "mtf": [...], ...} (chaves em minúsculo)
    - labels: um rótulo por amostra
    - ids: um id único por imagem, ex. "Benign_12"
    """
    mapa_img = indexar_por_numero(configuracao_classe["img_dir"])

    mapas_reshape = {}
    for canal in canais_reshape:
        padrao = PADROES_GLOB_RESHAPE[canal](rec_plot_size, MTF_size)
        mapas_reshape[canal] = indexar_por_numero(configuracao_classe["dirs_reshape"][canal], padrao)

    indices = set(mapa_img.keys())
    for canal, mapa in mapas_reshape.items():
        if not mapa:
            raise FileNotFoundError(
                f"Nenhum arquivo encontrado para o canal '{canal}' em "
                f"{configuracao_classe['dirs_reshape'][canal]}"
            )
        indices &= set(mapa.keys())
    indices = sorted(indices)

    nome_classe = Path(configuracao_classe["img_dir"]).name

    dados = {"img": []}
    for canal in canais_reshape:
        dados[canal.lower()] = []
    labels, ids = [], []

    for indice in indices:
        img_path = obter_arquivo(mapa_img, indice, "imagem original")
        # RGB sempre -- RecPlot (grayscale) é replicado nos 3 canais pelo convert
        dados["img"].append(np.array(Image.open(img_path).convert("RGB")))
        for canal in canais_reshape:
            caminho = obter_arquivo(mapas_reshape[canal], indice, canal)
            dados[canal.lower()].append(np.array(Image.open(caminho).convert("RGB")))
        labels.append(configuracao_classe["label"])
        ids.append(f"{nome_classe}_{indice}")

    return dados, labels, ids


def padronizar(img_array, tamanho):
    """Redimensiona pra tamanho e devolve (3, H, W). Grayscale (ndim==2) é
    replicado em 3 canais como rede de segurança."""
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    if img_array.shape[:2] != tamanho:
        img_array = np.array(Image.fromarray(img_array).resize(tamanho, Image.BILINEAR))
    return np.transpose(img_array, (2, 0, 1))  # (H, W, 3) -> (3, H, W)


def montar_input_canais(dados_canais, nomes_canais, tamanho):
    """Concatena os canais de nomes_canais numa entrada (3*N, H, W) por
    amostra, ex. ["img"] -> 3 canais, ["img", "mtf"] -> 6.

    dados_canais: {"img": [...], "recplot": [...], "mtf": [...], ...}
    nomes_canais: canais a concatenar nessa combinação, ex. ["img", "mtf"]
    """
    n_amostras = len(dados_canais[nomes_canais[0]])
    entradas = []
    for i in range(n_amostras):
        blocos = [padronizar(dados_canais[nome][i], tamanho) for nome in nomes_canais]
        entradas.append(np.concatenate(blocos, axis=0))
    return entradas


# Nome curto (usado em arquivos/metricas.csv) -> checkpoint do timm
MODELOS_TIMM = {
    "vit_b16": "vit_base_patch16_224.augreg_in21k",
    "coatnet": "coatnet_1_rw_224.sw_in1k",
}

# Pasta de métricas da CNN -- se o splits_kfold de um dataset/config já
# existir lá, ele é reaproveitado (só leitura) pra ter os mesmos folds da ResNet.
PASTA_SPLITS_CNN = Path(__file__).resolve().parents[1] / "CNN" / "METRICAS"


def criar_modelo(modelo, in_channels, num_classes):
    """Cria o modelo do timm pré-treinado com cabeça pra num_classes.
    in_chans != 3 é adaptado pelo próprio timm (repete/escala os pesos da
    primeira camada)."""
    if modelo not in MODELOS_TIMM:
        raise ValueError(f"modelo '{modelo}' desconhecido -- opções: {list(MODELOS_TIMM)}")
    return timm.create_model(
        MODELOS_TIMM[modelo], pretrained=True, in_chans=in_channels, num_classes=num_classes
    )


def normalizar(X, rede):
    """(N, C, H, W) em 0..255 -> normalizado com mean/std do checkpoint
    (repetidos por bloco de 3 canais)."""
    n_blocos = X.shape[1] // 3
    mean = torch.tensor(list(rede.pretrained_cfg["mean"]) * n_blocos, dtype=X.dtype).view(1, -1, 1, 1)
    std = torch.tensor(list(rede.pretrained_cfg["std"]) * n_blocos, dtype=X.dtype).view(1, -1, 1, 1)
    return (X / 255.0 - mean) / std


# ============================================================
# MÉTRICAS (binário ou multi-classe)
# ============================================================

def calcular_metricas(y_true, y_pred, y_prob_matriz, n_classes):
    """Acurácia/precisão/recall/f1/auc -- binário (métricas da classe 1) ou
    multi-classe (médias macro + AUC one-vs-rest).

    y_prob_matriz: (n_amostras, n_classes), saída completa do softmax.
    """
    acuracia = accuracy_score(y_true, y_pred)
    if n_classes == 2:
        precisao = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_prob_matriz[:, 1])
    else:
        precisao = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        auc = roc_auc_score(y_true, y_prob_matriz, multi_class="ovr", average="macro")
    return {"acuracia": acuracia, "precisao": precisao, "recall": recall, "f1": f1, "auc": auc}


# ============================================================
# CROSS-VALIDATION (StratifiedKFold 5 folds, 80/20) COM SPLIT FIXO
# ============================================================

def gerar_ou_carregar_splits(ids, labels, n_splits=5, seed=42, caminho_splits="splits_kfold.json"):
    """Gera (ou carrega, se já existir) uma divisão StratifiedKFold de
    n_splits folds (80% treino+val / 20% teste), indexada por id -- garante
    que toda combinação/execução usa exatamente os mesmos folds.

    Pra gerar uma nova divisão (ex. mudou o dataset), apague caminho_splits
    manualmente. Um caminho_splits por dataset (ex. f"splits_kfold_{nome}.json").
    """
    caminho_splits = Path(caminho_splits)
    ids = list(ids)
    labels = np.asarray(labels)
    set_ids_atual = set(ids)

    if caminho_splits.exists():
        with open(caminho_splits, "r", encoding="utf-8") as f:
            salvo = json.load(f)

        if set(salvo["ids"]) != set_ids_atual:
            raise ValueError(
                f"O conjunto de IDs salvo em {caminho_splits} não bate com o "
                "conjunto de IDs atual (dataset mudou?). Apague o arquivo de "
                "splits para gerar uma nova divisão, ou confirme que está "
                "usando o mesmo dataset em todas as combinações."
            )

        return salvo["folds"]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    posicoes = np.arange(len(ids))
    folds = []
    for idx_treino, idx_teste in skf.split(posicoes, labels):
        folds.append({
            "treino_ids": [ids[i] for i in idx_treino],
            "teste_ids": [ids[i] for i in idx_teste],
        })

    with open(caminho_splits, "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "n_splits": n_splits, "ids": ids, "folds": folds}, f, indent=2)

    return folds


def indices_por_ids(ids_referencia, ids_desejados):
    """Converte uma lista de ids em posições dentro de ids_referencia."""
    posicao = {id_: i for i, id_ in enumerate(ids_referencia)}
    return [posicao[id_] for id_ in ids_desejados]


# ============================================================
# REGISTRO DE MÉTRICAS
# ============================================================

CAMPOS_METRICAS_CSV = [
    "id_run",
    "data",
    "modelo",
    "dataset",
    "size_maxL",
    "mtf_q",
    "combinacao",
    "fold",
    "acuracia",
    "precisao",
    "recall",
    "f1",
    "auc",
    "perda_treino",
]


def _proximo_id_run(caminho_csv):
    """Lê o CSV existente e retorna max(id_run) + 1, ou 1 se não existir/estiver vazio."""
    if not caminho_csv.exists():
        return 1
    maior_id = 0
    with open(caminho_csv, mode="r", newline="", encoding="utf-8") as arquivo:
        leitor = csv.DictReader(arquivo)
        for linha in leitor:
            try:
                maior_id = max(maior_id, int(linha["id_run"]))
            except (KeyError, ValueError, TypeError):
                continue
    return maior_id + 1


def salva_metrica_csv(modelo, dataset, combinacao, precisao, recall, f1,
                       acuracia=None, auc=None, perda_treino=None, fold=None,
                       mtf_q=None, size_maxL=None, caminho_csv="metricas.csv",
                       sobrescrever=False):
    """Adiciona uma linha em metricas.csv (append; cria header se não
    existir). sobrescrever=True reescreve o arquivo do zero (id_run reinicia
    em 1).

    - id_run: automático (maior id_run salvo + 1)
    - data: hoje, formato yyyymmdd
    - combinacao: ex. "img_original", "img_original+mtf"
    - fold: 1..n_folds, ou "media" pra linha com a média dos folds
    - mtf_q: bits de quantização do MTF dessa rodada (None quando a
      combinação não envolve canal MTF), pra distinguir linhas de
      combinações com o mesmo nome rodadas com Q diferentes (ver MTF_Q_LIST
      em orquestrador.py)
    - size_maxL: maxL dessa rodada (ex. "maxL15"), pra distinguir linhas de
      combinações com o mesmo nome rodadas com maxL diferentes (ver
      SIZE_MAXL_LIST em orquestrador.py)
    - precisao/recall/f1: obrigatórias; acuracia/auc/perda_treino: opcionais

    Retorna o id_run gerado.
    """
    caminho_csv = Path(caminho_csv)
    modo = "w" if sobrescrever else "a"
    escreve_header = sobrescrever or not caminho_csv.exists()
    id_run = 1 if sobrescrever else _proximo_id_run(caminho_csv)

    linha = {
        "id_run": id_run,
        "data": date.today().strftime("%Y%m%d"),
        "modelo": modelo,
        "dataset": dataset,
        "size_maxL": size_maxL if size_maxL is not None else "",
        "mtf_q": mtf_q if mtf_q is not None else "",
        "combinacao": combinacao,
        "fold": fold if fold is not None else "",
        "acuracia": acuracia if acuracia is not None else "",
        "precisao": precisao,
        "recall": recall,
        "f1": f1,
        "auc": auc if auc is not None else "",
        "perda_treino": perda_treino if perda_treino is not None else "",
    }

    with open(caminho_csv, mode=modo, newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=CAMPOS_METRICAS_CSV)
        if escreve_header:
            escritor.writeheader()
        escritor.writerow(linha)

    return id_run


# ============================================================
# REGISTRO E COMPARAÇÃO DE PREDIÇÕES (softmax) POR IMAGEM
# ============================================================

CAMPOS_PREDICOES_CSV = ["id", "combinacao", "fold", "y_true", "y_pred", "y_prob"]


def salvar_predicoes_csv(ids_teste, y_true, y_prob_matriz, y_pred, combinacao, fold,
                          caminho_csv="predicoes.csv"):
    """Registra uma linha por imagem de teste (id, y_true, y_pred, e o vetor
    de softmax completo em JSON) pra uma combinação/fold. Sempre em modo
    append. y_prob_matriz: (n_amostras, n_classes)."""
    caminho_csv = Path(caminho_csv)
    escreve_header = not caminho_csv.exists()
    with open(caminho_csv, mode="a", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=CAMPOS_PREDICOES_CSV)
        if escreve_header:
            escritor.writeheader()
        for id_img, yt, prob_linha, ypred in zip(ids_teste, y_true, y_prob_matriz, y_pred):
            escritor.writerow({
                "id": id_img,
                "combinacao": combinacao,
                "fold": fold,
                "y_true": int(yt),
                "y_pred": int(ypred),
                "y_prob": json.dumps([float(p) for p in prob_linha]),
            })


def comparar_softmax(combinacao_a, combinacao_b, caminho_csv="predicoes.csv"):
    """Compara, imagem a imagem (mesmo id + fold), a saída softmax de duas
    combinações em predicoes.csv. Requer que ambas tenham rodado com o
    mesmo splits_kfold.json.

    Imprime correlação de Pearson, diferença absoluta média e concordância
    de classe predita; retorna o DataFrame da comparação.
    """
    import pandas as pd
    from scipy.stats import pearsonr

    df = pd.read_csv(caminho_csv)
    # Dedup: linha (combinacao, fold, id) repetida (ex. dataset rodado mais
    # de uma vez antes do resume existir) -- fica só a última.
    df = df.drop_duplicates(subset=["combinacao", "fold", "id"], keep="last")
    df["y_prob"] = df["y_prob"].apply(json.loads)

    a = df[df["combinacao"] == combinacao_a].set_index(["id", "fold"])
    b = df[df["combinacao"] == combinacao_b].set_index(["id", "fold"])

    comuns = a.index.intersection(b.index)
    if len(comuns) == 0:
        raise ValueError(
            f"Nenhum id+fold em comum entre '{combinacao_a}' e '{combinacao_b}' em {caminho_csv}."
        )

    probs_a = np.stack(a.loc[comuns, "y_prob"].values)
    probs_b = np.stack(b.loc[comuns, "y_prob"].values)
    diferencas = np.abs(probs_a - probs_b)

    comparacao = pd.DataFrame({
        "y_true": a.loc[comuns, "y_true"].values,
        f"y_pred_{combinacao_a}": a.loc[comuns, "y_pred"].values,
        f"y_pred_{combinacao_b}": b.loc[comuns, "y_pred"].values,
        "diferenca_media_abs_softmax": diferencas.mean(axis=1),
        "diferenca_max_abs_softmax": diferencas.max(axis=1),
    }, index=comuns)
    comparacao["mesma_predicao"] = (
        comparacao[f"y_pred_{combinacao_a}"] == comparacao[f"y_pred_{combinacao_b}"]
    )

    corr, _ = pearsonr(probs_a.flatten(), probs_b.flatten())

    print(f"\nComparação {combinacao_a} x {combinacao_b} ({len(comparacao)} imagens em comum)")
    print(f"Correlação de Pearson entre as probabilidades: {corr:.4f}")
    print(f"Diferença absoluta média (todas as classes):   {diferencas.mean():.4f}")
    print(f"Concordância na classe predita:                 {comparacao['mesma_predicao'].mean():.2%}")

    return comparacao


def _carregar_predicoes_cache(caminho_predicoes, nome_combinacao):
    """Lê as linhas já salvas de uma combinação em predicoes.csv e agrupa
    por fold: {fold: {"ids_teste": [...], "y_true": ndarray, "y_pred":
    ndarray, "y_prob_matriz": ndarray}}. {} se o arquivo não existir ou a
    combinação ainda não ter nenhuma linha salva -- usado tanto pro resume
    (pular fold já treinado) quanto pra ablação por softmax.

    Dedup: se o mesmo (fold, id) aparecer mais de uma vez (ex. dataset
    rodado mais de uma vez antes do resume existir), fica só a ÚLTIMA
    linha lida. As listas de cada fold saem sempre ordenadas por id (não
    pela ordem de inserção no CSV) -- garante que folds de combinações
    diferentes fiquem alinhados linha a linha na ablação por softmax."""
    caminho_predicoes = Path(caminho_predicoes)
    if not caminho_predicoes.exists():
        return {}

    por_fold = {}
    with open(caminho_predicoes, mode="r", newline="", encoding="utf-8") as arquivo:
        leitor = csv.DictReader(arquivo)
        for linha in leitor:
            if linha["combinacao"] != nome_combinacao:
                continue
            fold = int(linha["fold"])
            por_id = por_fold.setdefault(fold, {})
            por_id[linha["id"]] = linha  # duplicata: última linha lida vence

    resultado = {}
    for fold, por_id in por_fold.items():
        ids_ordenados = sorted(por_id)
        resultado[fold] = {
            "ids_teste": ids_ordenados,
            "y_true": np.array([int(por_id[i]["y_true"]) for i in ids_ordenados], dtype=np.int64),
            "y_pred": np.array([int(por_id[i]["y_pred"]) for i in ids_ordenados], dtype=np.int64),
            "y_prob_matriz": np.array(
                [json.loads(por_id[i]["y_prob"]) for i in ids_ordenados], dtype=np.float64
            ),
        }
    return resultado


# ============================================================
# ORQUESTRAÇÃO: split fixo por dataset, filtro de combinações e
# execução completa (combinação x fold) de um dataset
# ============================================================

_MAPA_CANAL_EXATO = {nome.lower(): nome for nome in NOMES_RESHAPE_PADRAO}

TAMANHO_PADRAO = (224, 224)


def pasta_modelo(pasta_saida, nome_dataset):
    """Subpasta de pasta_saida dedicada a um dataset/rodada -- mesmo nome já
    descoberto do dataset (source.name), sem alterar a grafia (ex.
    METRICAS/CR-SameRes/). Cria a pasta se não existir. É onde ficam
    splits_kfold, predições e o gráfico de métricas desse dataset --
    metricas.csv fica direto em pasta_saida, fora dela."""
    pasta = Path(pasta_saida) / nome_dataset
    pasta.mkdir(parents=True, exist_ok=True)
    return pasta


def caminho_cache_dataset(pasta_saida, nome_dataset, size_maxL, mtf_q, modelo):
    """Cache de predições por modelo + configuração do reshape
    (ex. predicoes_LG_vit_b16_Q32_maxL15.csv)."""
    pasta = pasta_modelo(pasta_saida, nome_dataset)
    configuracao = f"Q{mtf_q}_{size_maxL or 'auto'}"
    return pasta / f"predicoes_{nome_dataset}_{modelo}_{configuracao}.csv"


def descobrir_mtf_size(source, size_maxL, mtf_q=8, nomes_reshape=NOMES_RESHAPE_PADRAO):
    """Descobre o sufixo real dos arquivos MTF em disco pro Q pedido
    (mtf_q) e o size_maxL atual (ex. "MTF_Q8_N60") -- o N não é fixo, é o
    tamanho do sinal fractal concatenado, que muda com maxL (cada maxL tem
    seu próprio N). Olha os arquivos já salvos em vez de exigir digitar N
    à mão.

    Retorna None se a pasta MTF nem existir nesse dataset (comum quando só
    tá rodando img_original). Levanta FileNotFoundError se a pasta existir
    mas não tiver nenhum arquivo Q{mtf_q} pra esse size_maxL, ou
    ValueError se achar mais de um N diferente (dado inconsistente entre
    classes/imagens).
    """
    source = Path(source)
    pasta_mtf = source / "MTF"
    if not pasta_mtf.is_dir():
        return None

    padrao_n = re.compile(rf"_MTF_Q{mtf_q}_N(\d+)\.png$")
    encontrados = set()
    for arquivo in pasta_mtf.glob(f"*/{size_maxL}/*_MTF_Q{mtf_q}_N*.png"):
        m = padrao_n.search(arquivo.name)
        if m:
            encontrados.add(int(m.group(1)))

    if not encontrados:
        raise FileNotFoundError(
            f"Nenhum arquivo MTF_Q{mtf_q}_N*.png em {pasta_mtf}/*/{size_maxL}/ -- "
            f"conferir se esse Q e esse size_maxL já foram gerados nesse dataset."
        )
    if len(encontrados) > 1:
        raise ValueError(
            f"Mais de um N encontrado pra MTF_Q{mtf_q} em {pasta_mtf}/*/{size_maxL}/: "
            f"{sorted(encontrados)} -- dado inconsistente entre classes/imagens."
        )

    n = encontrados.pop()
    return f"MTF_Q{mtf_q}_N{n}"


def descobrir_datasets(base_dir, nomes_reshape=NOMES_RESHAPE_PADRAO):
    """Lista as subpastas de base_dir que têm pelo menos uma classe com .png."""
    base_dir = Path(base_dir)
    validos = []
    for pasta in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        try:
            rotulos = descobrir_classes(pasta, nomes_reshape)
        except FileNotFoundError:
            continue
        tem_imagem = any(
            next((pasta / nome_classe).glob("*.png"), None) is not None for nome_classe in rotulos
        )
        if tem_imagem:
            validos.append(pasta.name)
    return validos


def interpretar_combinacoes(source, combinacoes_desejadas, nomes_reshape=NOMES_RESHAPE_PADRAO):
    """Filtra combinacoes_desejadas (dict nome -> canais em minúsculo, ex.
    {"img_original": ["img"], "img+mtf": ["img", "mtf"]}) pras que esse
    dataset já suporta. Combinação com canal ainda não gerado é pulada
    (aviso no console), sem interromper as demais."""
    canais_disponiveis = {c.lower() for c in descobrir_canais_reshape(source, nomes_reshape)}
    validas = {}
    for nome_combinacao, canais in combinacoes_desejadas.items():
        faltando = [c for c in canais if c != "img" and c.lower() not in canais_disponiveis]
        if faltando:
            print(f"    [pulando combinação '{nome_combinacao}' em {Path(source).name}: "
                  f"canal(is) ainda não gerado(s) nesse dataset: {faltando}]")
            continue
        validas[nome_combinacao] = canais
    return validas


def _universo_ids_labels(classes_dict, canais_reshape_existentes, rec_plot_size, MTF_size):
    """Universo fixo (ids, labels) de um dataset: interseção, por classe,
    dos índices com imagem original + todo reshape já existente. Vira o
    split de cross-validation em rodar_dataset. Só indexa nomes de
    arquivo, não abre nenhuma imagem."""
    ids, labels = [], []
    for nome_classe, cfg in classes_dict.items():
        mapa_img = indexar_por_numero(cfg["img_dir"])
        indices = set(mapa_img.keys())
        for canal in canais_reshape_existentes:
            padrao = PADROES_GLOB_RESHAPE[canal](rec_plot_size, MTF_size)
            mapa = indexar_por_numero(cfg["dirs_reshape"][canal], padrao)
            indices &= set(mapa.keys())
        for indice in sorted(indices):
            ids.append(f"{nome_classe}_{indice}")
            labels.append(cfg["label"])
    return ids, np.array(labels, dtype=np.int64)


class _HistologiaDataset(Dataset):
    """Dataset PyTorch simples: X já vem pronto como tensor (N, C, H, W)."""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def treinar_e_avaliar_fold(X_treino, y_treino, X_val, y_val, X_teste, y_teste, ids_teste,
                            n_canais, n_classes, modelo, batch_size=8, n_epocas=20, lr=2e-5,
                            weight_decay=0.05, epocas_aquecimento=2):
    """Fine-tuning de um modelo do timm num fold: AdamW + warmup linear +
    cosseno, AMP na GPU, e teste com os pesos da época de menor loss de
    validação. Retorna as predições no conjunto de teste."""
    rede = criar_modelo(modelo, n_canais, n_classes).to(_DEVICE)

    X_treino_t = normalizar(torch.tensor(np.stack(X_treino), dtype=torch.float32), rede)
    X_val_t = normalizar(torch.tensor(np.stack(X_val), dtype=torch.float32), rede)
    X_teste_t = normalizar(torch.tensor(np.stack(X_teste), dtype=torch.float32), rede)
    y_treino_t = torch.tensor(y_treino, dtype=torch.long)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    y_teste_t = torch.tensor(y_teste, dtype=torch.long)

    train_loader = DataLoader(_HistologiaDataset(X_treino_t, y_treino_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(_HistologiaDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(_HistologiaDataset(X_teste_t, y_teste_t), batch_size=batch_size, shuffle=False)

    criterio = nn.CrossEntropyLoss()
    otimizador = optim.AdamW(rede.parameters(), lr=lr, weight_decay=weight_decay)

    passos_totais = n_epocas * len(train_loader)
    passos_aquecimento = min(epocas_aquecimento * len(train_loader), passos_totais - 1)

    def fator_lr(passo):
        if passo < passos_aquecimento:
            return (passo + 1) / passos_aquecimento
        progresso = (passo - passos_aquecimento) / max(1, passos_totais - passos_aquecimento)
        return 0.5 * (1.0 + math.cos(math.pi * progresso))

    agendador = optim.lr_scheduler.LambdaLR(otimizador, fator_lr)

    usar_amp = _DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=usar_amp)

    historico_loss_treino, historico_loss_val = [], []
    melhor_loss_val, melhor_epoca, melhor_estado = float("inf"), 0, None

    for epoca in range(1, n_epocas + 1):
        rede.train()
        perda_total = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(_DEVICE), y_batch.to(_DEVICE)
            otimizador.zero_grad()
            with torch.autocast(device_type=_DEVICE.type, enabled=usar_amp):
                saidas = rede(X_batch)
                perda = criterio(saidas, y_batch)
            scaler.scale(perda).backward()
            scaler.step(otimizador)
            scaler.update()
            agendador.step()
            perda_total += perda.item() * X_batch.size(0)
        historico_loss_treino.append(perda_total / len(train_loader.dataset))

        rede.eval()
        perda_val_total, acertos_val = 0.0, 0
        with torch.no_grad(), torch.autocast(device_type=_DEVICE.type, enabled=usar_amp):
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(_DEVICE), y_batch.to(_DEVICE)
                saidas = rede(X_batch)
                perda = criterio(saidas.float(), y_batch)
                perda_val_total += perda.item() * X_batch.size(0)
                acertos_val += (torch.argmax(saidas, dim=1) == y_batch).sum().item()
        loss_val = perda_val_total / len(val_loader.dataset)
        historico_loss_val.append(loss_val)

        if loss_val < melhor_loss_val:
            melhor_loss_val, melhor_epoca = loss_val, epoca
            melhor_estado = {k: v.detach().cpu().clone() for k, v in rede.state_dict().items()}

        print(f"      Época {epoca}/{n_epocas} - loss treino: {historico_loss_treino[-1]:.4f} | "
              f"loss val: {loss_val:.4f} | acurácia val: {acertos_val / len(val_loader.dataset):.4f}"
              f"{'  *' if melhor_epoca == epoca else ''}")

    rede.load_state_dict(melhor_estado)
    print(f"      Teste com a melhor época: {melhor_epoca} (loss val {melhor_loss_val:.4f})")

    rede.eval()
    y_true, y_pred, y_prob_matriz = [], [], []
    with torch.no_grad(), torch.autocast(device_type=_DEVICE.type, enabled=usar_amp):
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(_DEVICE)
            saidas = rede(X_batch).float()
            probs = torch.softmax(saidas, dim=1)
            preds = torch.argmax(saidas, dim=1)
            y_true.extend(y_batch.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob_matriz.extend(probs.cpu().numpy())

    del rede, melhor_estado
    if usar_amp:
        torch.cuda.empty_cache()

    return {
        "y_true": np.array(y_true),
        "y_pred": np.array(y_pred),
        "y_prob_matriz": np.array(y_prob_matriz),
        "ids_teste": ids_teste,
        "loss_treino_final": historico_loss_treino[melhor_epoca - 1],
    }


def rodar_dataset(source, combinacoes_desejadas, modelo, rec_plot_size=None, MTF_size=None, size_maxL=None,
                   mtf_q=8, tamanho_padrao=TAMANHO_PADRAO, batch_size=8, n_epocas=20, lr=2e-5,
                   seed=42, n_folds=5, pasta_saida="METRICAS", nomes_reshape=NOMES_RESHAPE_PADRAO,
                   pasta_splits_referencia=PASTA_SPLITS_CNN):
    """Roda combinação x fold pra um dataset: filtra combinações válidas
    (interpretar_combinacoes), fixa o split de cross-validation (universo =
    tudo que já existe em disco pra esse dataset, ver _universo_ids_labels),
    carrega os canais necessários e treina cada combinação x fold, salvando
    métricas em pasta_saida/metricas.csv e splits/predições em
    pasta_saida/<DATASET>/ (mesma grafia do dataset, ver pasta_modelo). Não plota
    (ver rodar_vit em vit.py).

    Resume: combinação x fold que já tem linha salva em predicoes.csv não é
    retreinada -- as métricas desse fold são recalculadas a partir do que
    já está no disco (ver _carregar_predicoes_cache). Pra forçar retreino,
    apague a linha (ou o arquivo) manualmente.

    MTF_size: se None (padrão) e o dataset já tiver pasta MTF, é descoberto
    automaticamente a partir do que existe em disco pra size_maxL + mtf_q
    (ver descobrir_mtf_size) -- não precisa mais digitar o N à mão nem
    manter em sincronia com size_maxL.

    Retorna {nome_combinacao: (media, desvio)}; {} se nenhuma combinação
    válida rodou.
    """
    source = Path(source)
    pasta_saida = Path(pasta_saida)
    pasta_saida.mkdir(parents=True, exist_ok=True)
    pasta_dataset = pasta_modelo(pasta_saida, source.name)
    print(f"\n{'=' * 60}\nDataset: {source.name} | modelo: {modelo}\n{'=' * 60}")

    combinacoes_validas = interpretar_combinacoes(source, combinacoes_desejadas, nomes_reshape)
    if not combinacoes_validas:
        print(f"  Nenhuma combinação válida para {source.name} -- pulando dataset.")
        return {}
    print(f"  Combinações válidas: {list(combinacoes_validas.keys())}")

    canais_reshape_existentes = descobrir_canais_reshape(source, nomes_reshape)

    if MTF_size is None and "MTF" in canais_reshape_existentes:
        MTF_size = descobrir_mtf_size(source, size_maxL, mtf_q=mtf_q)
        print(f"  MTF_size descoberto automaticamente pra size_maxL={size_maxL}: {MTF_size}")

    classes_dict = montar_classes_dict(source, canais_reshape_existentes, rec_plot_size, MTF_size, size_maxL)
    n_classes = len(classes_dict)
    print(f"  Classes: {list(classes_dict.keys())} ({n_classes})")

    ids_universo, labels_universo = _universo_ids_labels(
        classes_dict, canais_reshape_existentes, rec_plot_size, MTF_size
    )
    if not ids_universo:
        print(f"  Nenhuma imagem com todos os canais existentes em {source.name} -- pulando dataset.")
        return {}

    # StratifiedKFold precisa de >= n_folds amostras por classe
    contagem_classes = np.bincount(labels_universo)
    if contagem_classes.min() < n_folds:
        print(f"  Classe com só {contagem_classes.min()} imagens no universo fixo de {source.name} "
              f"(< n_folds={n_folds}) -- StratifiedKFold não consegue dividir, pulando dataset.")
        return {}

    # mesmos folds da ResNet: usa o splits da CNN se já existir (só leitura)
    nome_splits = f"splits_kfold_{source.name}_Q{mtf_q}_{size_maxL or 'auto'}.json"
    caminho_splits = pasta_dataset / nome_splits
    if pasta_splits_referencia is not None:
        splits_cnn = Path(pasta_splits_referencia) / source.name / nome_splits
        if splits_cnn.exists():
            caminho_splits = splits_cnn
    caminho_predicoes = caminho_cache_dataset(pasta_saida, source.name, size_maxL, mtf_q, modelo)
    folds = gerar_ou_carregar_splits(
        ids_universo, labels_universo, n_splits=n_folds, seed=seed, caminho_splits=caminho_splits
    )
    print(f"  Universo fixo: {len(ids_universo)} imagens | splits em {caminho_splits} ({len(folds)} folds)")

    # carrega os canais necessários pras combinações válidas e filtra pro universo fixo
    canais_uniao_lower = sorted({c.lower() for canais in combinacoes_validas.values() for c in canais if c != "img"})
    canais_uniao_exato = [_MAPA_CANAL_EXATO[c] for c in canais_uniao_lower]

    dados_canais = {"img": []}
    for canal in canais_uniao_lower:
        dados_canais[canal] = []
    labels_list, ids = [], []
    for nome_classe, cfg in classes_dict.items():
        dados_classe, labels_classe, ids_classe = carregar_classe(cfg, canais_uniao_exato, rec_plot_size, MTF_size)
        for canal_key, valores in dados_classe.items():
            dados_canais[canal_key].extend(valores)
        labels_list.extend(labels_classe)
        ids.extend(ids_classe)

    universo_set = set(ids_universo)
    mantidos = [i for i, id_ in enumerate(ids) if id_ in universo_set]
    dados_canais = {chave: [valores[i] for i in mantidos] for chave, valores in dados_canais.items()}
    labels = np.array([labels_list[i] for i in mantidos], dtype=np.int64)
    ids = [ids[i] for i in mantidos]

    print(f"  Imagens carregadas nessa rodada (após filtrar pro universo fixo): {len(ids)}")
    print(f"  Distribuição por classe (rótulo 0..N-1): {np.bincount(labels)}")

    resumo_combinacoes = {}

    for nome_combinacao, canais in combinacoes_validas.items():
        canais_lower = [c.lower() for c in canais]
        print(f"\n  {'-' * 50}\n  Combinação: {nome_combinacao}\n  {'-' * 50}")

        try:
            lista_inputs = montar_input_canais(dados_canais, canais_lower, tamanho_padrao)
        except Exception as erro:
            print(f"    [erro montando entradas de '{nome_combinacao}' em {source.name}, "
                  f"pulando essa combinação: {erro}]")
            continue
        n_canais = lista_inputs[0].shape[0]

        metricas_folds = []
        erro_na_combinacao = False
        # Resume: fold já salvo em predicoes.csv não é retreinado -- as
        # métricas são recalculadas a partir do que já está no disco.
        cache_existente = _carregar_predicoes_cache(caminho_predicoes, nome_combinacao)

        for i_fold, fold in enumerate(folds, start=1):
            if i_fold in cache_existente:
                dados_cache = cache_existente[i_fold]
                metricas = calcular_metricas(
                    dados_cache["y_true"], dados_cache["y_pred"], dados_cache["y_prob_matriz"], n_classes
                )
                metricas_folds.append(metricas)
                print(f"\n    -- Fold {i_fold}/{len(folds)} -- já processado, pulando (resume) --")
                continue

            print(f"\n    -- Fold {i_fold}/{len(folds)} --")
            try:
                idx_treino_full = indices_por_ids(ids, fold["treino_ids"])
                idx_teste = indices_por_ids(ids, fold["teste_ids"])

                idx_treino, idx_val = train_test_split(
                    idx_treino_full, test_size=0.15, random_state=seed, stratify=labels[idx_treino_full],
                )

                X_treino = [lista_inputs[i] for i in idx_treino]
                X_val = [lista_inputs[i] for i in idx_val]
                X_teste = [lista_inputs[i] for i in idx_teste]
                y_treino, y_val, y_teste = labels[idx_treino], labels[idx_val], labels[idx_teste]
                ids_teste = [ids[i] for i in idx_teste]

                resultado = treinar_e_avaliar_fold(
                    X_treino, y_treino, X_val, y_val, X_teste, y_teste, ids_teste,
                    n_canais, n_classes, modelo, batch_size=batch_size, n_epocas=n_epocas, lr=lr,
                )

                salvar_predicoes_csv(
                    resultado["ids_teste"], resultado["y_true"], resultado["y_prob_matriz"], resultado["y_pred"],
                    combinacao=nome_combinacao, fold=i_fold, caminho_csv=caminho_predicoes,
                )

                metricas = calcular_metricas(
                    resultado["y_true"], resultado["y_pred"], resultado["y_prob_matriz"], n_classes
                )

                salva_metrica_csv(
                    modelo=modelo, dataset=source.name, combinacao=nome_combinacao,
                    precisao=metricas["precisao"], recall=metricas["recall"], f1=metricas["f1"],
                    acuracia=metricas["acuracia"], auc=metricas["auc"],
                    perda_treino=resultado["loss_treino_final"], fold=i_fold,
                    mtf_q=mtf_q, size_maxL=size_maxL,
                    caminho_csv=pasta_saida / "metricas.csv",
                )

                metricas_folds.append(metricas)
                print(f"    Fold {i_fold}: acc={metricas['acuracia']:.4f} f1={metricas['f1']:.4f} "
                      f"auc={metricas['auc']:.4f}")
            except Exception as erro:
                print(f"    [erro no fold {i_fold} de '{nome_combinacao}' em {source.name}, "
                      f"pulando essa combinação: {erro}]")
                erro_na_combinacao = True
                break

        if erro_na_combinacao or not metricas_folds:
            continue

        media = {k: float(np.mean([m[k] for m in metricas_folds])) for k in metricas_folds[0]}
        desvio = {k: float(np.std([m[k] for m in metricas_folds])) for k in metricas_folds[0]}

        # Só regrava a linha de média se algum fold foi treinado de fato
        # nessa chamada -- combinação 100% resumida do cache não duplica a
        # linha "media" que uma rodada anterior já salvou.
        if len(cache_existente) < len(folds):
            salva_metrica_csv(
                modelo=modelo, dataset=source.name, combinacao=nome_combinacao,
                precisao=media["precisao"], recall=media["recall"], f1=media["f1"],
                acuracia=media["acuracia"], auc=media["auc"], fold="media",
                mtf_q=mtf_q, size_maxL=size_maxL,
                caminho_csv=pasta_saida / "metricas.csv",
            )

        resumo_combinacoes[nome_combinacao] = (media, desvio)
        print(f"\n  Média ± desvio padrão ({nome_combinacao}, {len(folds)} folds):")
        for k in media:
            print(f"    {k}: {media[k]:.4f} ± {desvio[k]:.4f}")

    return resumo_combinacoes


# ============================================================
# ABLAÇÃO POR COMBINAÇÃO DE SOFTMAX -- sem retreinar CNN
# ============================================================
# As 5 combinações de canal único (img/recplot/mtf/gasf/gadf) são as únicas
# treinadas de fato (rodar_dataset). Qualquer combinação com 2+ canais é
# obtida combinando (soft voting: média das probabilidades) as predições já
# salvas dessas 5 -- não treina nada, só lê predicoes.csv.

def mapear_canal_para_combinacao_base(combinacoes_base):
    """{canal: nome_combinacao} a partir das combinações de canal único de
    combinacoes_base (ex. {"img_original": ["img"], "mtf": ["mtf"]} ->
    {"img": "img_original", "mtf": "mtf"}). Combinações com 2+ canais em
    combinacoes_base são ignoradas aqui (não servem de base pra ablação)."""
    return {canais[0]: nome for nome, canais in combinacoes_base.items() if len(canais) == 1}


def rodar_ablacao_softmax(source, combinacoes_ablacao, combinacoes_base, modelo, n_folds=5,
                           pasta_saida="METRICAS", nomes_reshape=NOMES_RESHAPE_PADRAO,
                           size_maxL=None, mtf_q=8):
    """Calcula métricas de combinações multi-canal (2+ canais) combinando
    por soft voting (média das probabilidades softmax, id+fold a id+fold)
    as predições já salvas das combinações de canal único em
    combinacoes_base -- não treina nenhum modelo novo, só lê e agrega
    predicoes.csv. Resume: combinação já combinada antes (fold já salvo em
    predicoes.csv) não é recombinada.

    Pula (aviso no console, sem interromper as demais): combinação de canal
    ainda não gerado em disco (interpretar_combinacoes); combinação de
    canal único (é base, não ablação); canal sem combinação de canal único
    correspondente em combinacoes_base; combinação-base que ainda não tem
    os n_folds salvos em predicoes.csv (rodar rodar_dataset primeiro).

    Retorna {nome_combinacao: (media, desvio)}, mesmo formato de
    rodar_dataset -- dá pra juntar os dois dicts direto (ver rodar_vit).
    """
    source = Path(source)
    pasta_dataset = pasta_modelo(pasta_saida, source.name)
    caminho_predicoes = caminho_cache_dataset(pasta_saida, source.name, size_maxL, mtf_q, modelo)
    caminho_metricas = Path(pasta_saida) / "metricas.csv"

    mapa_canal_base = mapear_canal_para_combinacao_base(combinacoes_base)
    combinacoes_validas = interpretar_combinacoes(source, combinacoes_ablacao, nomes_reshape)
    resumo_combinacoes = {}

    for nome_combinacao, canais in combinacoes_validas.items():
        canais_lower = [c.lower() for c in canais]
        if len(canais_lower) < 2:
            continue  # combinação de canal único é base, não ablação

        nomes_base = []
        canal_faltando = None
        for canal in canais_lower:
            nome_base = mapa_canal_base.get(canal)
            if nome_base is None:
                canal_faltando = canal
                break
            nomes_base.append(nome_base)
        if canal_faltando:
            print(f"    [pulando ablação '{nome_combinacao}' em {source.name}: canal "
                  f"'{canal_faltando}' não tem combinação de canal único correspondente "
                  f"em combinacoes_base]")
            continue

        cache_por_base = {}
        base_faltando = None
        for nome_base in nomes_base:
            cache = _carregar_predicoes_cache(caminho_predicoes, nome_base)
            if len(cache) < n_folds:
                base_faltando = nome_base
                break
            cache_por_base[nome_base] = cache
        if base_faltando:
            print(f"    [pulando ablação '{nome_combinacao}' em {source.name}: combinação-base "
                  f"'{base_faltando}' ainda não tem os {n_folds} folds salvos em "
                  f"{caminho_predicoes} -- rodar rodar_dataset primeiro]")
            continue

        folds_prontos = _carregar_predicoes_cache(caminho_predicoes, nome_combinacao)
        metricas_folds = []

        for i_fold in range(1, n_folds + 1):
            if i_fold in folds_prontos:
                dados_cache = folds_prontos[i_fold]
                metricas_folds.append(calcular_metricas(
                    dados_cache["y_true"], dados_cache["y_pred"], dados_cache["y_prob_matriz"],
                    dados_cache["y_prob_matriz"].shape[1],
                ))
                continue

            linhas_base = [cache_por_base[nome_base][i_fold] for nome_base in nomes_base]
            ids_ref = linhas_base[0]["ids_teste"]
            if any(linha["ids_teste"] != ids_ref for linha in linhas_base[1:]):
                print(f"    [pulando fold {i_fold} de '{nome_combinacao}' em {source.name}: "
                      f"ids de teste divergem entre as combinações-base -- splits desalinhados]")
                continue

            y_prob_matriz = np.mean([linha["y_prob_matriz"] for linha in linhas_base], axis=0)
            y_pred = np.argmax(y_prob_matriz, axis=1)
            y_true = linhas_base[0]["y_true"]

            salvar_predicoes_csv(
                ids_ref, y_true, y_prob_matriz, y_pred,
                combinacao=nome_combinacao, fold=i_fold, caminho_csv=caminho_predicoes,
            )

            metricas = calcular_metricas(y_true, y_pred, y_prob_matriz, y_prob_matriz.shape[1])
            salva_metrica_csv(
                modelo=f"ensemble_softmax_{modelo}", dataset=source.name, combinacao=nome_combinacao,
                precisao=metricas["precisao"], recall=metricas["recall"], f1=metricas["f1"],
                acuracia=metricas["acuracia"], auc=metricas["auc"], fold=i_fold,
                mtf_q=mtf_q, size_maxL=size_maxL,
                caminho_csv=caminho_metricas,
            )
            metricas_folds.append(metricas)
            print(f"    [ablação] {nome_combinacao} fold {i_fold}: acc={metricas['acuracia']:.4f} "
                  f"f1={metricas['f1']:.4f} auc={metricas['auc']:.4f}")

        if len(metricas_folds) < n_folds:
            continue

        media = {k: float(np.mean([m[k] for m in metricas_folds])) for k in metricas_folds[0]}
        desvio = {k: float(np.std([m[k] for m in metricas_folds])) for k in metricas_folds[0]}

        if len(folds_prontos) < n_folds:
            salva_metrica_csv(
                modelo=f"ensemble_softmax_{modelo}", dataset=source.name, combinacao=nome_combinacao,
                precisao=media["precisao"], recall=media["recall"], f1=media["f1"],
                acuracia=media["acuracia"], auc=media["auc"], fold="media",
                mtf_q=mtf_q, size_maxL=size_maxL,
                caminho_csv=caminho_metricas,
            )

        resumo_combinacoes[nome_combinacao] = (media, desvio)
        print(f"\n  [ablação] Média ± desvio padrão ({nome_combinacao}, {n_folds} folds):")
        for k in media:
            print(f"    {k}: {media[k]:.4f} ± {desvio[k]:.4f}")

    return resumo_combinacoes
