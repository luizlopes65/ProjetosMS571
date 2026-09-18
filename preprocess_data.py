import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_PATH = PROJECT_ROOT / "processed_data.csv"

labels_df = pd.read_csv(DATA_DIR / "labels.csv", header=None)
images_df = pd.read_csv(DATA_DIR / "images.csv", header=None)

X = []
y = []
for i in tqdm(range(0, 5000)):
    linha_df = images_df.values[i]
    label = labels_df.values[i]
   
    
    if linha_df.dtype == object:
        linha_df = np.char.replace(linha_df.astype(str), ",", ".").astype(float)
        
    X.append(linha_df)
    y.append(int(label[0]))

final_df = pd.DataFrame(X)
final_df["label"] = y  

final_df.to_csv(PROCESSED_DATA_PATH, index=False, header=False)
