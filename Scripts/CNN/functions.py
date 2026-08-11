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
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def indexar_por_numero(diretorio, padrao_glob="*.png"):
    """Indexa os arquivos de um diretório por índice numérico extraído do
    nome (ex. "Benign (5).png", "5_RecPlot_512x512.png"). padrao_glob
    filtra quais arquivos entram. Retorna {} se o diretório não existir."""
    diretorio = Path(diretorio)
    padrao_numero = re.compile(r"\d+")
    mapa = {}
    if not diretorio.is_dir():
        return mapa
    for arquivo in sorted(diretorio.glob(padrao_glob)):
        correspondencia = padrao_numero.search(arquivo.stem)
        if correspondencia is None:
            continue
        indice = int(correspondencia.group())
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


def criar_resnet50_nch(in_channels, pretrained=True, num_classes=2):
    """ResNet50 com conv1 adaptada pra in_channels (múltiplo de 3, um bloco
    RGB por canal empilhado) e fc adaptada pra num_classes.

    Pesos pré-treinados de conv1 são repetidos por bloco de 3 canais,
    divididos por n_blocos (inflação de canais, estilo I3D). Pra
    in_channels=3 fica idêntico ao pré-treinado original.
    """
    modelo = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)

    conv1_antigo = modelo.conv1
    conv1_novo = nn.Conv2d(
        in_channels=in_channels,
        out_channels=conv1_antigo.out_channels,
        kernel_size=conv1_antigo.kernel_size,
        stride=conv1_antigo.stride,
        padding=conv1_antigo.padding,
        bias=(conv1_antigo.bias is not None),
    )

    if pretrained:
        if in_channels % 3 != 0:
            raise ValueError(
                f"in_channels={in_channels} não é múltiplo de 3 -- espera-se um "
                "bloco RGB (3 canais) por canal empilhado (ver montar_input_canais)."
            )
        n_blocos = in_channels // 3
        with torch.no_grad():
            for b in range(n_blocos):
                conv1_novo.weight[:, b * 3:(b + 1) * 3, :, :] = conv1_antigo.weight / n_blocos

    modelo.conv1 = conv1_novo
    modelo.fc = nn.Linear(modelo.fc.in_features, num_classes)
    return modelo


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
                       caminho_csv="metricas.csv", sobrescrever=False):
    """Adiciona uma linha em metricas.csv (append; cria header se não
    existir). sobrescrever=True reescreve o arquivo do zero (id_run reinicia
    em 1).

    - id_run: automático (maior id_run salvo + 1)
    - data: hoje, formato yyyymmdd
    - combinacao: ex. "img_original", "img_original+mtf"
    - fold: 1..n_folds, ou "media" pra linha com a média dos folds
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


# ============================================================
# ORQUESTRAÇÃO: split fixo por dataset, filtro de combinações e
# execução completa (combinação x fold) de um dataset
# ============================================================

_MAPA_CANAL_EXATO = {nome.lower(): nome for nome in NOMES_RESHAPE_PADRAO}

TAMANHO_PADRAO = (224, 224)


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
                            n_canais, n_classes, batch_size=8, n_epocas=20, lr=1e-4):
    """Treina uma ResNet50 do zero (conv1 adaptada pra n_canais, fc pra
    n_classes) num fold e retorna as predições no conjunto de teste."""
    X_treino_t = torch.tensor(np.stack(X_treino), dtype=torch.float32) / 255.0
    X_val_t = torch.tensor(np.stack(X_val), dtype=torch.float32) / 255.0
    X_teste_t = torch.tensor(np.stack(X_teste), dtype=torch.float32) / 255.0
    y_treino_t = torch.tensor(y_treino, dtype=torch.long)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    y_teste_t = torch.tensor(y_teste, dtype=torch.long)

    train_loader = DataLoader(_HistologiaDataset(X_treino_t, y_treino_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(_HistologiaDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(_HistologiaDataset(X_teste_t, y_teste_t), batch_size=batch_size, shuffle=False)

    modelo = criar_resnet50_nch(in_channels=n_canais, pretrained=True, num_classes=n_classes).to(_DEVICE)
    criterio = nn.CrossEntropyLoss()
    otimizador = optim.Adam(modelo.parameters(), lr=lr)

    historico_loss_treino, historico_loss_val = [], []

    for epoca in range(1, n_epocas + 1):
        modelo.train()
        perda_total = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(_DEVICE), y_batch.to(_DEVICE)
            otimizador.zero_grad()
            saidas = modelo(X_batch)
            perda = criterio(saidas, y_batch)
            perda.backward()
            otimizador.step()
            perda_total += perda.item() * X_batch.size(0)
        historico_loss_treino.append(perda_total / len(train_loader.dataset))

        modelo.eval()
        perda_val_total, acertos_val = 0.0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(_DEVICE), y_batch.to(_DEVICE)
                saidas = modelo(X_batch)
                perda = criterio(saidas, y_batch)
                perda_val_total += perda.item() * X_batch.size(0)
                acertos_val += (torch.argmax(saidas, dim=1) == y_batch).sum().item()
        historico_loss_val.append(perda_val_total / len(val_loader.dataset))

        print(f"      Época {epoca}/{n_epocas} - loss treino: {historico_loss_treino[-1]:.4f} | "
              f"loss val: {historico_loss_val[-1]:.4f} | acurácia val: {acertos_val / len(val_loader.dataset):.4f}")

    modelo.eval()
    y_true, y_pred, y_prob_matriz = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(_DEVICE)
            saidas = modelo(X_batch)
            probs = torch.softmax(saidas, dim=1)
            preds = torch.argmax(saidas, dim=1)
            y_true.extend(y_batch.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob_matriz.extend(probs.cpu().numpy())

    return {
        "y_true": np.array(y_true),
        "y_pred": np.array(y_pred),
        "y_prob_matriz": np.array(y_prob_matriz),
        "ids_teste": ids_teste,
        "loss_treino_final": historico_loss_treino[-1],
    }


def rodar_dataset(source, combinacoes_desejadas, rec_plot_size=None, MTF_size=None, size_maxL=None,
                   tamanho_padrao=TAMANHO_PADRAO, batch_size=8, n_epocas=20, lr=1e-4,
                   seed=42, n_folds=5, pasta_saida=".", nomes_reshape=NOMES_RESHAPE_PADRAO):
    """Roda combinação x fold pra um dataset: filtra combinações válidas
    (interpretar_combinacoes), fixa o split de cross-validation (universo =
    tudo que já existe em disco pra esse dataset, ver _universo_ids_labels),
    carrega os canais necessários e treina cada combinação x fold, salvando
    métricas (pasta_saida/metricas.csv) e predições
    (pasta_saida/predicoes_<dataset>.csv). Não plota (ver rodar_resnet em
    resnet.py).

    Retorna {nome_combinacao: (media, desvio)}; {} se nenhuma combinação
    válida rodou.
    """
    source = Path(source)
    pasta_saida = Path(pasta_saida)
    print(f"\n{'=' * 60}\nDataset: {source.name}\n{'=' * 60}")

    combinacoes_validas = interpretar_combinacoes(source, combinacoes_desejadas, nomes_reshape)
    if not combinacoes_validas:
        print(f"  Nenhuma combinação válida para {source.name} -- pulando dataset.")
        return {}
    print(f"  Combinações válidas: {list(combinacoes_validas.keys())}")

    canais_reshape_existentes = descobrir_canais_reshape(source, nomes_reshape)
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

    caminho_splits = pasta_saida / f"splits_kfold_{source.name}.json"
    caminho_predicoes = pasta_saida / f"predicoes_{source.name}.csv"
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

        for i_fold, fold in enumerate(folds, start=1):
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
                    n_canais, n_classes, batch_size=batch_size, n_epocas=n_epocas, lr=lr,
                )

                salvar_predicoes_csv(
                    resultado["ids_teste"], resultado["y_true"], resultado["y_prob_matriz"], resultado["y_pred"],
                    combinacao=nome_combinacao, fold=i_fold, caminho_csv=caminho_predicoes,
                )

                metricas = calcular_metricas(
                    resultado["y_true"], resultado["y_pred"], resultado["y_prob_matriz"], n_classes
                )

                salva_metrica_csv(
                    modelo="resnet50", dataset=source.name, combinacao=nome_combinacao,
                    precisao=metricas["precisao"], recall=metricas["recall"], f1=metricas["f1"],
                    acuracia=metricas["acuracia"], auc=metricas["auc"],
                    perda_treino=resultado["loss_treino_final"], fold=i_fold,
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

        salva_metrica_csv(
            modelo="resnet50", dataset=source.name, combinacao=nome_combinacao,
            precisao=media["precisao"], recall=media["recall"], f1=media["f1"],
            acuracia=media["acuracia"], auc=media["auc"], fold="media",
            caminho_csv=pasta_saida / "metricas.csv",
        )

        resumo_combinacoes[nome_combinacao] = (media, desvio)
        print(f"\n  Média ± desvio padrão ({nome_combinacao}, {len(folds)} folds):")
        for k in media:
            print(f"    {k}: {media[k]:.4f} ± {desvio[k]:.4f}")

    return resumo_combinacoes