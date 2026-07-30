import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)


# ============================================================
# CONFIGURAÇÃO
# ============================================================

source = Path(r"C:/Users/felip/Desktop/Facul/TCC/Scripts_Datasets/Datasets/Imagens Histológicas/CR")

size_maxL = "maxL15"

MTF_sizes = ["MTF_Q8_N35", "MTF_Q16_N35", "MTF_Q32_N35", "MTF_Q64_N35"]
MTF_size = MTF_sizes[0] 

rec_plot_size = "RecPlot_512x512"

CLASSES = {
    "Benign": {
        "label": 0,
        "img_dir": source / "Benign", 
        "recplot_dir": source / "Benign" / size_maxL,
        "mtf_dir": source / "Benign" / size_maxL,
        "gasf_dir": source / "Benign" / size_maxL,
        "gadf_dir": source / "Benign" / size_maxL,
        "img_prefix": "Benign",
        "gasf_prefix": "Benign",
        "gadf_prefix": "Benign",
    },
    "Malignant": {
        "label": 1,
        "img_dir": source / "Malignant",
        "recplot_dir": source / "Malignant" / size_maxL,
        "mtf_dir": source / "Malignant" / size_maxL,
        "gasf_dir": source / "Malignant" / size_maxL,
        "gadf_dir": source / "Malignant" / size_maxL,
        "img_prefix": "Malignant",
        "gasf_prefix": "Malignant",
        "gadf_prefix": "Malignant",
    },
}

TAMANHO_PADRAO = (224, 224)
BATCH_SIZE = 8
N_EPOCAS = 20
LR = 1e-4


# ============================================================
# LEITURA DOS DADOS
# ============================================================

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

    list_img, list_recplot, list_mtf, list_gasf, list_gadf, labels = [], [], [], [], [], []

    for indice in indices:
        img_path = encontrar_arquivo(
            configuracao_classe["img_dir"],
            [
                f"{configuracao_classe['img_prefix']} ({indice}).png",
                f"{configuracao_classe['img_prefix']}_{indice}.png",
            ],
        )
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

        list_img.append(np.array(Image.open(img_path).convert("L")))
        list_recplot.append(np.array(Image.open(recplot_path)))
        list_mtf.append(np.array(Image.open(mtf_path)))
        list_gasf.append(np.array(Image.open(gasf_path)))
        list_gadf.append(np.array(Image.open(gadf_path)))
        labels.append(configuracao_classe["label"])

    return list_img, list_recplot, list_mtf, list_gasf, list_gadf, labels


list_img_b, list_recplot_b, list_mtf_b, list_gasf_b, list_gadf_b, labels_b = carregar_classe(CLASSES["Benign"])
list_img_m, list_recplot_m, list_mtf_m, list_gasf_m, list_gadf_m, labels_m = carregar_classe(CLASSES["Malignant"])

list_img = list_img_b + list_img_m
list_recplot = list_recplot_b + list_recplot_m
list_mtf = list_mtf_b + list_mtf_m
list_gasf = list_gasf_b + list_gasf_m
list_gadf = list_gadf_b + list_gadf_m
labels = np.array(labels_b + labels_m, dtype=np.int64)

print("Número de imagens originais carregadas:", len(list_img))
print("Número de RecPlots carregados:", len(list_recplot))
print("Número de MTFs carregados:", len(list_mtf))
print("Número de GASFs carregados:", len(list_gasf))
print("Número de GADFs carregados:", len(list_gadf))
print("Número de rótulos:", len(labels))


# ============================================================
# MONTAGEM DAS ENTRADAS (imagem original + todos os reshapes)
# ============================================================

def padronizar(img_array, tamanho=TAMANHO_PADRAO):
    if img_array.ndim != 2:
        img_array = img_array[:, :, 0]
    if img_array.shape != tamanho:
        img_array = np.array(Image.fromarray(img_array).resize(tamanho, Image.BILINEAR))
    return img_array


def montar_input(img, recplot, mtf, gasf, gadf):
    entradas = []
    for im, r, m, g_s, g_d in zip(img, recplot, mtf, gasf, gadf):
        im = padronizar(im)
        r = padronizar(r)
        m = padronizar(m)
        g_s = padronizar(g_s)
        g_d = padronizar(g_d)
        stacked = np.stack([im, r, m, g_s, g_d], axis=0)  # 5 canais
        entradas.append(stacked)
    return entradas


list_inputs = montar_input(list_img, list_recplot, list_mtf, list_gasf, list_gadf)
N_CANAIS = list_inputs[0].shape[0]
print(f"\nNúmero de amostras combinadas ({N_CANAIS} canais):", len(list_inputs))


# ============================================================
# SPLIT TREINO/VALIDAÇÃO/TESTE (70/15/15)
# ============================================================

indices = np.arange(len(list_inputs))

# 1º corte: separa o teste (15%) do restante (85%)
idx_resto, idx_teste = train_test_split(
    indices,
    test_size=0.15,
    random_state=42,
    stratify=labels,
)

# 2º corte: do restante (85%), separa treino (70% do total) e validação (15% do total)
# 0.15 / 0.85 ≈ 0.1765 -> fração do "resto" que vira validação
idx_treino, idx_val = train_test_split(
    idx_resto,
    test_size=0.15 / 0.85,
    random_state=42,
    stratify=labels[idx_resto],
)

X_treino = [list_inputs[i] for i in idx_treino]
X_val = [list_inputs[i] for i in idx_val]
X_teste = [list_inputs[i] for i in idx_teste]
y_treino = labels[idx_treino]
y_val = labels[idx_val]
y_teste = labels[idx_teste]

print("\nDivisão treino/validação/teste (70/15/15):")
print("Treino:     ", len(X_treino), "amostras")
print("Validação:  ", len(X_val), "amostras")
print("Teste:      ", len(X_teste), "amostras")
print("Treino benigno/maligno:    ", np.bincount(y_treino))
print("Validação benigno/maligno: ", np.bincount(y_val))
print("Teste benigno/maligno:     ", np.bincount(y_teste))

X_treino_t = torch.tensor(np.stack(X_treino), dtype=torch.float32) / 255.0
X_val_t = torch.tensor(np.stack(X_val), dtype=torch.float32) / 255.0
X_teste_t = torch.tensor(np.stack(X_teste), dtype=torch.float32) / 255.0
y_treino_t = torch.tensor(y_treino, dtype=torch.long)
y_val_t = torch.tensor(y_val, dtype=torch.long)
y_teste_t = torch.tensor(y_teste, dtype=torch.long)


# ============================================================
# DATASET / DATALOADER
# ============================================================

class HistologiaDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_ds = HistologiaDataset(X_treino_t, y_treino_t)
val_ds = HistologiaDataset(X_val_t, y_val_t)
test_ds = HistologiaDataset(X_teste_t, y_teste_t)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


# ============================================================
# MODELO — ResNet50 adaptada para 5 canais
# ============================================================

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsando dispositivo: {device}")

modelo = criar_resnet50_nch(in_channels=N_CANAIS, pretrained=True).to(device)
print("\nCamada conv1 adaptada:")
print(modelo.conv1)


# ============================================================
# TREINO
# ============================================================

criterio = nn.CrossEntropyLoss()
otimizador = optim.Adam(modelo.parameters(), lr=LR)

historico_loss_treino = []
historico_loss_val = []

for epoca in range(1, N_EPOCAS + 1):
    modelo.train()
    perda_total = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        otimizador.zero_grad()
        saidas = modelo(X_batch)
        perda = criterio(saidas, y_batch)
        perda.backward()
        otimizador.step()

        perda_total += perda.item() * X_batch.size(0)

    perda_media = perda_total / len(train_ds)
    historico_loss_treino.append(perda_media)

    # avaliação na validação (só para monitorar, não atualiza pesos)
    modelo.eval()
    perda_val_total = 0.0
    acertos_val = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            saidas = modelo(X_batch)
            perda = criterio(saidas, y_batch)
            perda_val_total += perda.item() * X_batch.size(0)
            acertos_val += (torch.argmax(saidas, dim=1) == y_batch).sum().item()

    perda_val_media = perda_val_total / len(val_ds)
    acuracia_val = acertos_val / len(val_ds)
    historico_loss_val.append(perda_val_media)

    print(f"Época {epoca}/{N_EPOCAS} - loss treino: {perda_media:.4f} | "
          f"loss val: {perda_val_media:.4f} | acurácia val: {acuracia_val:.4f}")


# ============================================================
# AVALIAÇÃO NO CONJUNTO DE TESTE
# ============================================================

modelo.eval()
y_true, y_pred, y_prob = [], [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        saidas = modelo(X_batch)
        probs = torch.softmax(saidas, dim=1)[:, 1]
        preds = torch.argmax(saidas, dim=1)

        y_true.extend(y_batch.numpy())
        y_pred.extend(preds.cpu().numpy())
        y_prob.extend(probs.cpu().numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)

acuracia = accuracy_score(y_true, y_pred)
precisao = precision_score(y_true, y_pred)
revocacao = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_prob)
matriz_confusao = confusion_matrix(y_true, y_pred)

print("\n===== Métricas no conjunto de teste =====")
print(f"Acurácia:           {acuracia:.4f}")
print(f"Precisão:           {precisao:.4f}")
print(f"Revocação (recall): {revocacao:.4f}")
print(f"F1-score:           {f1:.4f}")
print(f"AUC-ROC:            {auc:.4f}")
print("\nMatriz de confusão (linhas=real, colunas=predito):")
print(matriz_confusao)
print("\nRelatório de classificação completo:")
print(classification_report(y_true, y_pred, target_names=["Benign", "Malignant"]))


# ============================================================
# GRÁFICOS
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(range(1, N_EPOCAS + 1), historico_loss_val, marker="o")
axes[0].set_title("Curva de perda (treino)")
axes[0].set_xlabel("Época")
axes[0].set_ylabel("Loss")
axes[0].grid(alpha=0.3)

axes[1].imshow(matriz_confusao, cmap="Blues")
axes[1].set_title("Matriz de confusão")
axes[1].set_xticks([0, 1])
axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(["Benign", "Malignant"])
axes[1].set_yticklabels(["Benign", "Malignant"])
axes[1].set_xlabel("Predito")
axes[1].set_ylabel("Real")
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, matriz_confusao[i, j], ha="center", va="center",
                      color="white" if matriz_confusao[i, j] > matriz_confusao.max() / 2 else "black")

fpr, tpr, _ = roc_curve(y_true, y_prob)
axes[2].plot(fpr, tpr, label=f"AUC = {auc:.3f}")
axes[2].plot([0, 1], [0, 1], linestyle="--", color="gray")
axes[2].set_title("Curva ROC")
axes[2].set_xlabel("Taxa de falsos positivos")
axes[2].set_ylabel("Taxa de verdadeiros positivos")
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("metricas_resnet" + size_maxL + "_" + MTF_size + "_.png", dpi=150)
plt.show()