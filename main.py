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
def get_MNIST(train=False):
    return MNIST(root='~/',train=train,download=True, transform=transforms.Compose([transforms.ToTensor()]))

@st.cache
def get_PCATSNE(data_loader,b_size=64,subset=1000,pca_components=50):
    img, label = next(iter(data_loader))
    img = img.view((b_size,-1))
    label = label.view((-1))
    subset_x = img[:subset]
    subset_y = label[:subset]
    pca_k = PCA(n_components=pca_components).fit_transform(subset_x)
    pca_tsne = TSNE(random_state=40,n_components=2,verbose=True).fit_transform(pca_k)

    vis_data = pd.DataFrame(data={'x':pca_tsne[:,0],'y':pca_tsne[:,1],'label':subset_y})
    return vis_data

st.write('MNIST embedding visualization experiment')    

@st.cache
def plot_tsne(ds,b_size,cn):
    data_loader = torch.utils.data.DataLoader(ds, batch_size=b_size)
    vis_data = get_PCATSNE(data_loader,b_size=b_size,pca_components=cn)
    fig = plt.gcf()
    ax = fig.subplots()
    ch_plt = sns.scatterplot(data=vis_data,x="x",y="y",hue="label",ax=ax)
    return fig

ds = get_MNIST()
b_size=st.sidebar.slider("Number of samples", min_value=1,max_value=len(ds),value=128)
cn = st.sidebar.slider("Number of features",min_value=1,max_value=min(b_size,28*28),value=b_size)

#if st.sidebar.button("Plot"):
fig = plot_tsne(ds,b_size=b_size,cn=cn)
st.pyplot(fig, use_container_width=True)




