import streamlit as st
import torchvision.transforms as transforms
import torch
from torchvision.datasets import MNIST
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_MNIST(train=True):
    return MNIST(root='~/',train=train,download=True, transform=transforms.Compose([transforms.ToTensor()]))

def get_data(b_size):
    ds = get_MNIST()
    data_loader = torch.utils.data.DataLoader(ds, batch_size=b_size)
    img, label = next(iter(data_loader))
    img = img.view((b_size,-1))
    label = label.view((-1))
    return img, label

def get_TSNE(b_size=10000,pca=False,pca_components=50):
    img, label = get_data(b_size)
    if pca:
        img = PCA(n_components=pca_components).fit_transform(img)
    tsne = TSNE(n_components=2,verbose=True,perplexity=40,metric='cosine', n_iter=300).fit_transform(img)
    print('fitted transform now creating df')
    vis_data = pd.DataFrame(data={'x':tsne[:,0],'y':tsne[:,1],'label':label})
    return vis_data


st.write('MNIST embedding visualization experiment')    
subset= 20000
vis_data = get_TSNE(b_size=subset)

fig = plt.figure()
ax = fig.subplots()
ch_plt = sns.scatterplot(data=vis_data,x="x",y="y",hue="label",ax=ax,palette='Spectral')
ch_plt.set(title='TSNE without PCA')
print('sending to streamlit')
st.pyplot(fig, use_container_width=True)

vis_data = get_TSNE(b_size=subset,pca=True,pca_components=400)
fig = plt.figure()
ax = fig.subplots()
ch_plt = sns.scatterplot(data=vis_data,x="x",y="y",hue="label",ax=ax,palette='Spectral')
ch_plt.set(title='TSNE after PCA to 400 components')
print('sending to streamlit')
st.pyplot(fig, use_container_width=True)

vis_data = get_TSNE(b_size=subset,pca=True,pca_components=200)
fig = plt.figure()
ax = fig.subplots()
ch_plt = sns.scatterplot(data=vis_data,x="x",y="y",hue="label",ax=ax,palette='Spectral')
ch_plt.set(title='TSNE after PCA to 200 components')
print('sending to streamlit')
st.pyplot(fig, use_container_width=True)

vis_data = get_TSNE(b_size=subset,pca=True,pca_components=100)
fig = plt.figure()
ax = fig.subplots()
ch_plt = sns.scatterplot(data=vis_data,x="x",y="y",hue="label",ax=ax,palette='Spectral')
ch_plt.set(title='TSNE after PCA to 100 components')
print('sending to streamlit')
st.pyplot(fig, use_container_width=True)

vis_data = get_TSNE(b_size=subset,pca=True,pca_components=50)
fig = plt.figure()
ax = fig.subplots()
ch_plt = sns.scatterplot(data=vis_data,x="x",y="y",hue="label",ax=ax,palette='Spectral')
ch_plt.set(title='TSNE after PCA to 50 components')
print('sending to streamlit')
st.pyplot(fig, use_container_width=True)

vis_data = get_TSNE(b_size=subset,pca_components=25)
fig = plt.figure()
ax = fig.subplots()
ch_plt = sns.scatterplot(data=vis_data,x="x",y="y",hue="label",ax=ax,palette='Spectral')
ch_plt.set(title='TSNE after PCA to 25 components')
print('sending to streamlit')
st.pyplot(fig, use_container_width=True)