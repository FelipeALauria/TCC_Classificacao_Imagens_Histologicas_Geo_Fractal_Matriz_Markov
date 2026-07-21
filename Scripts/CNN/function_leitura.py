import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

def le_imagens(source):
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

    return imagens

# Leitura das imagens histólogicas

# Lê a imagem na escala de cinza
def trans_img_escala_cinza(lista_imagens):
    imagens_cinza = []
    for img in lista_imagens:
        img_cinza = img.convert('L')  # Converte para escala de cinza
        imagens_cinza.append(np.array(img_cinza))
    return imagens_cinza


## Leitura dos Reshapes feitos no matlab

# Recurrence Plot 
def le_recplot(source):
    recplot = []
    files = sum(1 for file in Path(source).glob('*.png') if file.is_file())
    for i in range(1, files + 1):
        img = Image.open(Path(source) / f'RecPlot ({i}).png')
        img_array = np.array(img)
        recplot.append(img_array)
    return recplot

# markov transition field
def le_mtf(source):
    mtf = []
    files = sum(1 for file in Path(source).glob('*.png') if file.is_file())
    for i in range(1, files + 1):
        img = Image.open(Path(source) / f'MTF ({i}).png')
        img_array = np.array(img)
        mtf.append(img_array)
    return mtf
