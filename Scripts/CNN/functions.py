import re
import csv
from datetime import date
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models

def indexar_por_numero(diretorio, padrao_glob="*.png"):
    """
    "SET": escaneia uma pasta uma única vez e monta um dicionário
    {indice: caminho}, identificando o índice de cada arquivo pelo
    primeiro número encontrado no nome (funciona com "Benign (5).png",
    "Benign_5.png", "5_RecPlot_512x512.png", "Malignant_5_gasf.png" etc,
    sem precisar saber o prefixo/convenção de nome usado).

    padrao_glob permite restringir quais arquivos entram no índice,
    por exemplo "*_gasf.png" ou f"*_{rec_plot_size}.png".
    """
    diretorio = Path(diretorio)
    padrao_numero = re.compile(r"\d+")
    mapa = {}
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


def carregar_classe(configuracao_classe, rec_plot_size, MTF_size):
    # "set": indexa cada pasta uma única vez, sem depender de prefixo de nome
    mapa_recplot = indexar_por_numero(configuracao_classe["recplot_dir"], f"*_{rec_plot_size}.png")
    mapa_mtf = indexar_por_numero(configuracao_classe["mtf_dir"], f"*_{MTF_size}.png")
    mapa_img = indexar_por_numero(configuracao_classe["img_dir"])
    mapa_gasf = indexar_por_numero(configuracao_classe["gasf_dir"], "*_gasf.png")
    mapa_gadf = indexar_por_numero(configuracao_classe["gadf_dir"], "*_gadf.png")

    indices = sorted(mapa_recplot.keys())

    list_img, list_recplot, list_mtf, list_gasf, list_gadf, labels = [], [], [], [], [], []

    for indice in indices:
        # "get": busca cada arquivo pelo índice já indexado
        img_path = obter_arquivo(mapa_img, indice, "imagem original")
        recplot_path = mapa_recplot[indice]
        mtf_path = obter_arquivo(mapa_mtf, indice, "MTF")
        gasf_path = obter_arquivo(mapa_gasf, indice, "GASF")
        gadf_path = obter_arquivo(mapa_gadf, indice, "GADF")

        list_img.append(np.array(Image.open(img_path).convert("L")))
        list_recplot.append(np.array(Image.open(recplot_path)))
        list_mtf.append(np.array(Image.open(mtf_path)))
        list_gasf.append(np.array(Image.open(gasf_path)))
        list_gadf.append(np.array(Image.open(gadf_path)))
        labels.append(configuracao_classe["label"])

    return list_img, list_recplot, list_mtf, list_gasf, list_gadf, labels


def padronizar(img_array, tamanho):
    if img_array.ndim != 2:
        img_array = img_array[:, :, 0]
    if img_array.shape != tamanho:
        img_array = np.array(Image.fromarray(img_array).resize(tamanho, Image.BILINEAR))
    return img_array


def montar_input(img, recplot, mtf, gasf, gadf, tamanho):
    entradas = []
    for im, r, m, g_s, g_d in zip(img, recplot, mtf, gasf, gadf):
        im = padronizar(im, tamanho)
        r = padronizar(r, tamanho)
        m = padronizar(m, tamanho)
        g_s = padronizar(g_s, tamanho)
        g_d = padronizar(g_d, tamanho)
        stacked = np.stack([im, r, m, g_s, g_d], axis=0)  # 5 canais
        entradas.append(stacked)
    return entradas


def criar_resnet50_nch(in_channels, pretrained=True, num_classes=2):
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
        with torch.no_grad():
            media_pesos = conv1_antigo.weight.mean(dim=1, keepdim=True)
            for c in range(in_channels):
                if c < 3:
                    conv1_novo.weight[:, c:c + 1, :, :] = conv1_antigo.weight[:, c:c + 1, :, :]
                else:
                    conv1_novo.weight[:, c:c + 1, :, :] = media_pesos

    modelo.conv1 = conv1_novo
    modelo.fc = nn.Linear(modelo.fc.in_features, num_classes)
    return modelo


# ============================================================
# REGISTRO DE MÉTRICAS
# ============================================================

CAMPOS_METRICAS_CSV = [
    "id_run",
    "data",
    "modelo",
    "dataset",
    "combinacao",
    "acuracia",
    "precisao",
    "recall",
    "f1",
    "auc",
    "perda_treino",
]


def _proximo_id_run(caminho_csv):
    """Lê o CSV existente e retorna max(id_run) + 1, ou 1 se o arquivo não existir/estiver vazio."""
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
                       acuracia=None, auc=None, perda_treino=None,
                       caminho_csv="metricas.csv", sobrescrever=False):
    """
    Adiciona uma linha de resultados em metricas.csv. Por padrão
    (sobrescrever=False) é sempre em modo append no mesmo arquivo,
    criando o header apenas se o arquivo ainda não existir.

    Se sobrescrever=True, o arquivo inteiro é reescrito do zero
    (apaga tudo que já estava lá) e passa a conter só o header +
    esta linha, com id_run reiniciando em 1.

    - id_run: gerado automaticamente (1, 2, 3, ...), lendo o maior
      id_run já salvo no CSV e somando 1 (ou reiniciando em 1 se
      sobrescrever=True)
    - data: data de hoje no formato yyyymmdd, gerada automaticamente
    - modelo: nome do modelo, ex. "resnet50"
    - dataset: nome do dataset usado, ex. "CR"
    - combinacao: combinação de entrada, ex. "img_original",
      "img_original+mtf", "img_original+recplot+mtf"
    - precisao, recall, f1: métricas obrigatórias
    - acuracia, auc: opcionais, ficam em branco no CSV se não informadas
    - perda_treino: valor de loss do treino (ex. da última época), opcional
    - sobrescrever: se True, reescreve o arquivo inteiro (default False)

    Retorna o id_run gerado para esta linha (útil para nomear outros
    arquivos, como o gráfico de métricas, com o mesmo id).
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