import pandas as pd
import os

train = pd.DataFrame()
filename = []
label = []

for i in os.listdir('bookedge/JPEGImages/'):
    filename.append('JPEGImages/' + i)
    label.append('Annotations/' + i.split('.')[0] + '.png')

train['filename'] = filename
train['label'] = label
train.to_csv('bookedge/train_list.txt', index=False,header=None,sep=' ')
train.to_csv('bookedge/val_list.txt', index=False,header=None, sep=' ')
train.to_csv('bookedge/test_list.txt', index=False,header=None, sep=' ')