import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

source = r'C:\Users\felip\Desktop\Facul\TCC\Scripts_Datasets\Datasets\Imagens Histológicas\CR\Benign'
imagens = []

# Carrega as imagens em um vetor
base_path = Path(source)
files = sum(1 for file in base_path.glob('*.png') if file.is_file())

print(f"Número total de arquivos encontrados: {files}" )

for i in range(1, files + 1):
    img = Image.open(base_path / f'Benign ({i}).png')
    img_array = np.array(img)
    imagens.append(img_array)

print()
print("Imagens:")

qtd_visualizar = min(5, len(imagens))

if qtd_visualizar > 0:
    fig, axes = plt.subplots(1, qtd_visualizar, figsize=(4 * qtd_visualizar, 4))
    if qtd_visualizar == 1:
        axes = [axes]

    for i in range(qtd_visualizar):
        axes[i].imshow(imagens[i])
        axes[i].set_title(f"Imagem {i + 1}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()
else:
    print("Nenhuma imagem carregada para visualizacao")

#Leitura das imagens histólogicas

# Preserva a cor das imagens
def le_imagem_rgb():
    return 0

# Lê a imagem na escala de cinza
def le_imagem_cinza():
    return 3

# Leitura dos Reshapes feitos no matlab

# Recurrence Plot 
def le_recplot():
    return 2

# markov transition field
def le_mtf():
    return 1
