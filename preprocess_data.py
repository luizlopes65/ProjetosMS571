import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

pasta_atual = os.getcwd()

labels_df = pd.read_csv(os.path.join(pasta_atual, "data", "labels.csv"), header=None)
images_df = pd.read_csv(os.path.join(pasta_atual,"data", "images.csv"), header=None)

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

final_df.to_csv(os.path.join(pasta_atual,"data", "processed_data.csv"), index=False, header=False)

