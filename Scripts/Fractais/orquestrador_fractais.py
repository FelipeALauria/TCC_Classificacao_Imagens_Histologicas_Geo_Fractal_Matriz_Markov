"""Gera os atributos fractais (.mat) de todos os datasets na GPU, no mesmo
padrão de pastas do orquestradorFractais.m:
    <Dataset>/<Classe>/maxL<N>/<n>_maxL<N>.mat

Pula imagem/maxL que já existe (retomável). Os reshapes (GASF/GADF/MTF/RecPlot)
continuam no MATLAB: depois de rodar este script, o orquestradorFractais.m
encontra todos os .mat prontos, pula a extração e só gera os reshapes.

Como executar (dentro do .venv, com torch CUDA instalado):
    python Scripts/Fractais/orquestrador_fractais.py
"""
import os
from pathlib import Path

from fractal_features import escolher_device, extrair_classe

# Raiz do repositório (Scripts_Datasets), achada a partir deste arquivo: funciona
# no Windows e no servidor sem editar nada. Pra usar outra pasta de datasets:
#   export TCC_DATASETS="/caminho/para/Imagens Histológicas"
RAIZ_REPO = Path(__file__).resolve().parents[2]
BASE_DIR = Path(os.environ.get("TCC_DATASETS", RAIZ_REPO / "Datasets" / "Imagens Histológicas"))
DATASETS = ["CR-SameRes", "DIS", "LA", "LG", "NHL", "UCSB"]
MAXL_LIST = [65]
RESHAPES = {"GASF", "GADF", "MTF", "RecPlot"}

EXIGIR_GPU = True    # False deixa rodar na CPU (bem mais lento)
MEMORIA_GPU_GB = 1.5  # memória por bloco de caixas; aumente se a GPU tiver folga, diminua se der "out of memory"


def listar_classes(pasta_dataset):
    return sorted(p for p in pasta_dataset.iterdir()
                  if p.is_dir() and not p.name.startswith(".") and p.name not in RESHAPES)


if __name__ == "__main__":
    device = escolher_device(EXIGIR_GPU)
    erros = []
    for nome in DATASETS:
        pasta_dataset = BASE_DIR / nome
        print(f"\n===== Dataset: {nome} =====")
        try:
            for classe in listar_classes(pasta_dataset):
                print(f"--- Classe: {nome} / {classe.name} ---")
                destinos = {mL: classe / f"maxL{mL}" for mL in MAXL_LIST}
                try:
                    extrair_classe(classe, destinos, MAXL_LIST, device, MEMORIA_GPU_GB)
                except Exception as e:
                    print(f"ERRO nos fractais de {nome}/{classe.name}: {e}")
                    erros.append(f"{nome}/{classe.name}")
        except Exception as e:
            print(f"ERRO no dataset {nome}: {e}")
            erros.append(nome)

    print("\nOrquestrador de fractais concluído!")
    if erros:
        print(f"Com erro (pulados): {erros}")
