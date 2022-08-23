import streamlit as st
import torchvision.transforms as transforms
import torch
from torchvision.datasets import MNIST
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache
def get_MNIST(train=True):
    return MNIST(root='~/',train=train,download=True, transform=transforms.Compose([transforms.ToTensor()]))

@st.cache
def get_PCATSNE(b_size=64,subset=10000,pca_components=50):
    ds = get_MNIST()
    print(len(ds))
    data_loader = torch.utils.data.DataLoader(ds, batch_size=b_size)
    img, label = next(iter(data_loader))
    img = img.view((b_size,-1))
    label = label.view((-1))
    subset_x = img[:subset]
    subset_y = label[:subset]
    pca_k = PCA(n_components=pca_components).fit_transform(subset_x)
    pca_tsne = TSNE(random_state=42,n_components=2,verbose=True,perplexity=40, n_iter=300).fit_transform(pca_k)
    print('fitted transform now creating df')
    vis_data = pd.DataFrame(data={'x':pca_tsne[:,0],'y':pca_tsne[:,1],'label':subset_y})
    return vis_data

@st.cache
def get_TSNE(b_size=64,subset=10000):
    ds = get_MNIST()
    print(len(ds))
    data_loader = torch.utils.data.DataLoader(ds, batch_size=b_size)
    img, label = next(iter(data_loader))
    img = img.view((b_size,-1))
    label = label.view((-1))
    subset_x = img[:subset]
    subset_y = label[:subset]
    tsne = TSNE(random_state=42,n_components=2,verbose=True,perplexity=40, n_iter=300).fit_transform(subset_x)
    print('fitted transform now creating df')
    vis_data = pd.DataFrame(data={'x':tsne[:,0],'y':tsne[:,1],'label':subset_y})
    return vis_data


st.write('MNIST embedding visualization experiment')    
b_size=60000
subset= 10000
vis_data = get_TSNE(b_size=b_size,subset=subset)

fig = plt.figure()
ax = fig.subplots()
ch_plt = sns.scatterplot(data=vis_data,x="x",y="y",hue="label",ax=ax,palette='Spectral')
ch_plt.set(title='TSNE without PCA')
print('sending to streamlit')
st.pyplot(fig, use_container_width=True)


vis_data = get_PCATSNE(b_size=b_size,subset=subset,pca_components=100)
fig = plt.figure()
ax = fig.subplots()
ch_plt = sns.scatterplot(data=vis_data,x="x",y="y",hue="label",ax=ax,palette='Spectral')
ch_plt.set(title='TSNE after PCA to 100 components')
print('sending to streamlit')
st.pyplot(fig, use_container_width=True)


vis_data = get_PCATSNE(b_size=b_size,subset=subset,pca_components=25)
fig = plt.figure()
ax = fig.subplots()
ch_plt = sns.scatterplot(data=vis_data,x="x",y="y",hue="label",ax=ax,palette='Spectral')
ch_plt.set(title='TSNE after PCA to 25 components')
print('sending to streamlit')
st.pyplot(fig, use_container_width=True)