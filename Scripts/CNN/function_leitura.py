import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

def le_imagens(source):
    imagens = []

    base_path = Path(source)
    files = sum(1 for file in base_path.glob('*.png') if file.is_file())

    for i in range(1, files + 1):
        img = Image.open(base_path / f'Benign ({i}).png')
        img_array = np.array(img)
        imagens.append(img_array)

    return imagens

def trans_img_escala_cinza(lista_imagens):
    imagens_cinza = []
    for img in lista_imagens:
        img_cinza = img.convert('L') 
        imagens_cinza.append(np.array(img_cinza))
    return imagens_cinza

def le_recplot(source, recplot_size):
    recplot = []
    for i in range(1, 75):
        name = f'{i}_'+ str(recplot_size) + '.png'
        img = Image.open(Path(source) / name)
        img_array = np.array(img)
        recplot.append(img_array)
    return recplot

def le_mtf(source, x_maxL):
    mtf = []
    for i in range(1, 75):
        name = f'{i}_'+ str(x_maxL) + '.png'
        img = Image.open(Path(source) / name)
        img_array = np.array(img)
        mtf.append(img_array)
    return mtf

def le_gasf(source, class_name):
    gasf = []
    for i in range(1, 75):
        name = f'{class_name}_{i}_gasf.png'
        img = Image.open(Path(source) / name)
        img_array = np.array(img)
        gasf.append(img_array)
    return gasf

def le_gadf(source, class_name):
    gadf = []
    for i in range(1, 75):
        name = f'{class_name}_{i}_gadf.png'
        img = Image.open(Path(source) / name)
        img_array = np.array(img)
        gadf.append(img_array)
    return gadf
