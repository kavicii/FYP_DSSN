import torch
from network import DSSN
from torchsummary import summary
from read_file import read_csv
import argparse
import os
import pandas as pd
from sklearn.metrics import f1_score
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import (
    get_loader,
    LOSO_sequence_generate
)
from train import evaluate
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from sklearn.metrics import classification_report
idx = 2
log_file = open(f"predict{idx}.log", "w")
PATH="C:/Users/Kin Chan/Documents/FYP/PyTorch-DSSN-MER/model/model.pt"
myseed = 5630
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
np.random.seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(myseed)
    device = torch.device('cuda')
    
else:
    device = torch.device('cpu')

model = DSSN(5,'add').to(device)
#summary(model,[(3,227,227),(3,227,227)])
data, label_mapping = read_csv('CASME2CSV.csv')
log_file.write(str(data))
log_file.write(str(label_mapping))
train_list, test_list = LOSO_sequence_generate(data,"Subject")
train_csv = train_list[idx]
test_csv = test_list[idx]
print(test_csv)
_, test_loader = get_loader(csv_file=test_csv,
                            label_mapping=label_mapping,
                            img_root="Dataset\Cropped",
                            mode=tuple(["F","G"]),
                            batch_size=len(test_csv),
                            catego="CASME")
temp_output,temp_labels, temp_test_accuracy, temp_f1_score = evaluate(test_loader=test_loader,
                                             model=model,
                                             device=device)
temp_output = np.append(temp_output,temp_output)
temp_labels = np.append(temp_labels,temp_labels)

print(temp_output)
print(temp_labels)
label_mapping = ['disgust', 'happiness', 'others', 'repression', 'surprise']
cm = metrics.confusion_matrix(temp_labels, temp_output)
plt.figure(figsize=(15,8))
sns.heatmap(cm,square=True,annot=True,fmt='d',linecolor='white',linewidths=1.5,cbar=False)
plt.xlabel('Predicted',fontsize=20)
plt.ylabel('True',fontsize=20)
plt.show()
plt.savefig('cm.png')
print(classification_report(temp_labels,temp_output))
print(f"In  Subject{idx}, test accuracy: {temp_test_accuracy}, f1-score: {temp_f1_score}")
log_file.write(f"In Subject{idx}: Accuracy: {temp_test_accuracy}, F1-Score: {temp_f1_score}\n")






























log_file.close()