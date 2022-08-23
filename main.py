import streamlit as st
import torchvision.transforms as transforms
import torch
import open_clip
from torchvision.datasets import MNIST
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

@st.cache
def get_model(idx=0):
    modelname, pretrained = open_clip.list_pretrained()[idx]
    model, train_preprocess, val_preprocess = open_clip.create_model_and_transforms(modelname, pretrained=pretrained)
    return model, val_preprocess

@st.cache
def get_MNIST(train=False):
    _, val_preprocess = get_model()
    return MNIST(root='~/',train=train,download=False, transform=val_preprocess)

@st.cache
def get_loader(batch_size=32):
    ds = get_MNIST()
    return torch.utils.data.DataLoader(ds, batch_size=batch_size,num_workers=4)

@st.cache
def get_processed(b_size=32):
    model, _ = get_model()
    print("got model")
    data_loader = get_loader(b_size)
    print("got loader")
    img, label = [],[]
    for x,y in iter(data_loader):
        if len(label) >= 5000:
            break
        #x = x.cuda()
        x = model.visual(x)
        y = y.view((-1))
        img.extend(x.detach().cpu().view((b_size,-1)).numpy())
        label.extend(y.detach().numpy())
        del x
    print("encoded features")
    return np.array(img), np.array(label)
    
    
@st.cache
def get_PCATSNE(b_size=32,pca_components=50):
    subset_x, subset_y = get_processed()
    pca_k = PCA(n_components=pca_components).fit_transform(subset_x)
    print(type(pca_k),np.shape(pca_k))
    pca_tsne = TSNE(random_state=42,n_components=2,verbose=True,perplexity=40, n_iter=300).fit_transform(pca_k)
    print('fitted transform now creating df')
    vis_data = pd.DataFrame(data={'x':pca_tsne[:,0],'y':pca_tsne[:,1],'label':subset_y})
    return vis_data

@st.cache
def get_TSNE(b_size=32):
    subset_x, subset_y = get_processed()
    tsne = TSNE(random_state=42,n_components=2,verbose=True,perplexity=40, n_iter=300).fit_transform(subset_x)
    print('fitted transform now creating df')
    vis_data = pd.DataFrame(data={'x':tsne[:,0],'y':tsne[:,1],'label':subset_y})
    return vis_data


st.write('MNIST embedding visualization experiment')    
b_size=32
vis_data = get_TSNE(b_size=b_size)

fig = plt.figure()
ax = fig.subplots()
ch_plt = sns.scatterplot(data=vis_data,x="x",y="y",hue="label",ax=ax,palette='Spectral')
ch_plt.set(title='TSNE without PCA')
print('sending to streamlit')
st.pyplot(fig, use_container_width=True)


vis_data = get_PCATSNE(b_size=b_size,pca_components=100)
fig = plt.figure()
ax = fig.subplots()
ch_plt = sns.scatterplot(data=vis_data,x="x",y="y",hue="label",ax=ax,palette='Spectral')
ch_plt.set(title='TSNE after PCA to 100 components')
print('sending to streamlit')
st.pyplot(fig, use_container_width=True)


vis_data = get_PCATSNE(b_size=b_size,pca_components=25)
fig = plt.figure()
ax = fig.subplots()
ch_plt = sns.scatterplot(data=vis_data,x="x",y="y",hue="label",ax=ax,palette='Spectral')
ch_plt.set(title='TSNE after PCA to 25 components')
print('sending to streamlit')
st.pyplot(fig, use_container_width=True)
