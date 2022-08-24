import torchvision.transforms as transforms
import torch
import open_clip
from torchvision.datasets import MNIST
import pandas as pd
import numpy as np
from tqdm import tqdm

modelname, pretrained = open_clip.list_pretrained()[12]
model, train_preprocess, val_preprocess = open_clip.create_model_and_transforms(modelname, pretrained=pretrained)

model = model.cuda()

ds = MNIST(root='~/',train=False,download=True, transform=val_preprocess)

mini_b = 4
d_loader = torch.utils.data.DataLoader(ds, batch_size=mini_b,num_workers=1)

img, fts,label = None, None, None
for x,y in tqdm(d_loader):
    first_dm = x.shape[0]
    if img is None:
        img = x.view(first_dm,-1)
    else:
        img = np.concatenate((img,x.view((first_dm,-1))),axis=0)
    ft = model.visual(x.cuda())
    ft = ft.detach().cpu()
    if fts is None:
        fts = ft.view((first_dm,-1))
    else:
        fts = np.concatenate((fts,ft.view((first_dm,-1))),axis=0)

    y = y.view((-1))
    if label is None:
        label = y
    else:
        label = np.concatenate((label,y))
    del x
    torch.cuda.empty_cache()
    
df = pd.DataFrame(data = {'img':[[i] for i in img],'label':[l for l in label],'features': [[ft] for ft in fts]})
df.to_pickle('data/mnist_'+modelname+pretrained+'.pkl')
