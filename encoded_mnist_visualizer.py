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

mnist_df = pd.read_pickle('data/mnist_RN50x64openai.pkl')
im = np.squeeze(mnist_df['img'].to_list())
ft = np.squeeze(mnist_df['features'].to_list())
label = np.squeeze(mnist_df['label'].to_list())
print('read data')

pca = PCA(n_components=200).fit_transform(im)
tsne = TSNE(n_components=3,verbose=True,perplexity=40,metric='cosine', n_iter=300).fit_transform(pca)
vis_data = pd.DataFrame(data={'x':tsne[:,0],'y':tsne[:,1],'label':label})
fig = plt.figure()
ax = fig.subplots()
ax.scatter3D(vis_data['x'],vis_data['y'],vis_data['z'],c=vis_data['label'],cmap='Spectral')
#ch_plt = sns.scatterplot(data=vis_data,x="x",y="y",hue="label",ax=ax,palette='Spectral')
ch_plt.set(title='TSNE raw encoding')
print('sending to streamlit')
st.pyplot(fig, use_container_width=True)

print('next')
pca = PCA(n_components=200).fit_transform(ft)
tsne = TSNE(n_components=3,verbose=True,perplexity=40,metric='cosine', n_iter=300).fit_transform(pca)
vis_data = pd.DataFrame(data={'x':tsne[:,0],'y':tsne[:,1],'label':label})
fig = plt.figure()
ax = fig.subplots()
ax.scatter3D(vis_data['x'],vis_data['y'],vis_data['z'],c=vis_data['label'],cmap='Spectral')
#ch_plt = sns.scatterplot(data=vis_data,x="x",y="y",hue="label",ax=ax,palette='Spectral')
ch_plt.set(title='TSNE MNIST embeddings')
print('sending to streamlit')
st.pyplot(fig, use_container_width=True)
