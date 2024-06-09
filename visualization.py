import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pre_proccess_script import data_preprocess
import time

data_frame = data_preprocess(pd.read_csv('Dataset/train.csv'))
for column in data_frame.columns:
    sns.lmplot(x='SalePrice', y=column, data=data_frame)
    plt.savefig(f'Visualization_Output/{column}.png')

