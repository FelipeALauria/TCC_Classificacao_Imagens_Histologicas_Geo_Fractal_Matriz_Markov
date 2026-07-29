import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from sklearn.model_selection import train_test_split


BASE_DIR = Path(r"C:/Users/felip/Desktop/Facul/TCC/Scripts_Datasets/Datasets/Imagens Histológicas/CR")

size_maxL = "maxL15"

CLASSES = {
    "Benign": {
        "label": 0,
        "recplot_dir": BASE_DIR / "Benign" / size_maxL,
        "mtf_dir": BASE_DIR / "Benign" / size_maxL,
        "gasf_dir": BASE_DIR / "Benign" / size_maxL,
        "gadf_dir": BASE_DIR / "Benign" / size_maxL,
        "gasf_prefix": "Benign",
        "gadf_prefix": "Benign",
    },
    "Malignant": {
        "label": 1,
        "recplot_dir": BASE_DIR / "Malignant" / size_maxL,
        "mtf_dir": BASE_DIR / "Malignant" / size_maxL,
        "gasf_dir": BASE_DIR / "Malignant" / size_maxL,
        "gadf_dir": BASE_DIR / "Malignant" / size_maxL,
        "gasf_prefix": "Malignant",
        "gadf_prefix": "Malignant",
    },
}

rec_plot_size = "RecPlot_512x512"
MTF_size = "MTF_Q8_N35"

def encontrar_arquivo(diretorio, candidatos):
    for nome in candidatos:
        caminho = diretorio / nome
        if caminho.exists():
            return caminho
    raise FileNotFoundError(f"Nenhum arquivo encontrado em {diretorio} para: {candidatos}")


def carregar_classe(configuracao_classe):
    indices = []
    for caminho in sorted(configuracao_classe["recplot_dir"].glob(f"*_{rec_plot_size}.png")):
        correspondencia = re.match(r"^(\d+)_", caminho.name)
        if correspondencia:
            indices.append(int(correspondencia.group(1)))

    list_recplot = []
    list_mtf = []
    list_gasf = []
    list_gadf = []
    labels = []

    for indice in indices:
        recplot_path = configuracao_classe["recplot_dir"] / f"{indice}_{rec_plot_size}.png"
        mtf_path = configuracao_classe["mtf_dir"] / f"{indice}_{MTF_size}.png"
        gasf_path = encontrar_arquivo(
            configuracao_classe["gasf_dir"],
            [
                f"{configuracao_classe['gasf_prefix']}_{indice}_gasf.png",
                f"Benign_{indice}_gasf.png",
                f"Malignant_{indice}_gasf.png",
            ],
        )
        gadf_path = encontrar_arquivo(
            configuracao_classe["gadf_dir"],
            [
                f"{configuracao_classe['gadf_prefix']}_{indice}_gadf.png",
                f"Benign_{indice}_gadf.png",
                f"Malignant_{indice}_gadf.png",
            ],
        )

        list_recplot.append(np.array(Image.open(recplot_path)))
        list_mtf.append(np.array(Image.open(mtf_path)))
        list_gasf.append(np.array(Image.open(gasf_path)))
        list_gadf.append(np.array(Image.open(gadf_path)))
        labels.append(configuracao_classe["label"])

    return list_recplot, list_mtf, list_gasf, list_gadf, labels


list_recplot_b, list_mtf_b, list_gasf_b, list_gadf_b, labels_b = carregar_classe(CLASSES["Benign"])
list_recplot_m, list_mtf_m, list_gasf_m, list_gadf_m, labels_m = carregar_classe(CLASSES["Malignant"])

list_recplot = list_recplot_b + list_recplot_m
list_mtf = list_mtf_b + list_mtf_m
list_gasf = list_gasf_b + list_gasf_m
list_gadf = list_gadf_b + list_gadf_m
labels = np.array(labels_b + labels_m, dtype=np.int64)

# print("Número de RecPlots carregados:", len(list_recplot))
# print("Número de MTFs carregados:", len(list_mtf))
# print("Número de GASFs carregados:", len(list_gasf))
# print("Número de GADFs carregados:", len(list_gadf))
# print("Número de rótulos:", len(labels))


TAMANHO_PADRAO = (224, 224) 

def padronizar(img_array, tamanho=TAMANHO_PADRAO):
    if img_array.ndim != 2:
        img_array = img_array[:, :, 0]
    if img_array.shape != tamanho:
        img_array = np.array(Image.fromarray(img_array).resize(tamanho, Image.BILINEAR))
    return img_array

def montar_input(recplot, mtf, gasf, gadf):
    entradas = []
    for r, m, g_s, g_d in zip(recplot, mtf, gasf, gadf):
        r = padronizar(r)
        m = padronizar(m)
        g_s = padronizar(g_s)
        g_d = padronizar(g_d)
        stacked = np.stack([r, m, g_s, g_d], axis=0) 
        entradas.append(stacked)
    return entradas

list_inputs = montar_input(list_recplot, list_mtf, list_gasf, list_gadf)
print("\nNúmero de amostras combinadas (4 canais):", len(list_inputs))

indices = np.arange(len(list_inputs))

idx_treino, idx_teste = train_test_split(
    indices,
    train_size=0.8,
    test_size=0.2,
    random_state=42,
    stratify=labels,
)

X_treino = [list_inputs[i] for i in idx_treino]
X_teste  = [list_inputs[i] for i in idx_teste]
y_treino = labels[idx_treino]
y_teste = labels[idx_teste]

print("\nDivisão treino/teste (80/20):")
print("Treino:", len(X_treino), "amostras")
print("Teste: ", len(X_teste), "amostras")
print("Treino benigno/maligno:", np.bincount(y_treino))
print("Teste benigno/maligno: ", np.bincount(y_teste))

X_treino_t = torch.tensor(np.stack(X_treino), dtype=torch.float32) / 255.0
X_teste_t  = torch.tensor(np.stack(X_teste), dtype=torch.float32) / 255.0
y_treino_t = torch.tensor(y_treino, dtype=torch.long)
y_teste_t = torch.tensor(y_teste, dtype=torch.long)

def criar_resnet50_4ch(pretrained=True):
    modelo = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)

    conv1_antigo = modelo.conv1
    conv1_novo = nn.Conv2d(
        in_channels=4,
        out_channels=conv1_antigo.out_channels,
        kernel_size=conv1_antigo.kernel_size,
        stride=conv1_antigo.stride,
        padding=conv1_antigo.padding,
        bias=(conv1_antigo.bias is not None)
    )

    if pretrained:
        with torch.no_grad():
            conv1_novo.weight[:, :3, :, :] = conv1_antigo.weight
            conv1_novo.weight[:, 3:4, :, :] = conv1_antigo.weight.mean(dim=1, keepdim=True)

    modelo.conv1 = conv1_novo
    return modelo

modelo = criar_resnet50_4ch(pretrained=True)
print("\nCamada conv1 adaptada para 4 canais:")
print(modelo.conv1)
