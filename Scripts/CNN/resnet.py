import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from  function_leitura import trans_img_escala_cinza, le_imagens, le_recplot, le_mtf

source = 'C:/Users/felip/Desktop/Facul/TCC/Scripts_Datasets/Datasets/Imagens Histológicas/CR/Benign'
source_rec_mtf = 'C:/Users/felip/Desktop/Facul/TCC/Scripts_Datasets/Datasets/Imagens Histológicas/CR/Benign/maxL15'
rec_plot_size = "RecPlot_512x512"
MTF_size = "mtf_Q8_N35"

list_img = le_imagens(source)
print("Número de imagens carregadas: ", len(list_img))
print("\n")

list_mtf = le_mtf(source_rec_mtf, MTF_size)
print("Número de MTFs carregados: ", len(list_mtf))
print("\n")

list_recplot = le_recplot(source_rec_mtf, rec_plot_size)
print("Número de RecPlots carregados: ", len(list_recplot))


