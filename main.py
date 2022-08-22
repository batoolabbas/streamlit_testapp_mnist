import streamlit as st
import torchvision.transforms as transforms
import torch
from torchvision.datasets import MNIST
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import altair as alt

b_size = 64
mnist_test = torch.utils.data.DataLoader(MNIST(root='/home/ubuntu/datasets',train=False,download=True, transform=transforms.Compose([transforms.ToTensor()])), batch_size=b_size)

img, label = next(iter(mnist_test))
img = img.view((b_size,-1))
label = label.view((-1))
print(img.shape)
print(label.shape)


subset_x = img[:1000]
subset_y = label[:1000]

pca_50 = PCA(n_components=50).fit_transform(subset_x)
pca_tsne = TSNE(random_state=40,n_components=2,verbose=True).fit_transform(pca_50)

print(label.shape)
vis_data = pd.DataFrame(data={'x':pca_tsne[:,0],'y':pca_tsne[:,1],'label':subset_y})

ch_alt = alt.Chart(vis_data).mark_point().encode(
            x='x', 
            y='y',
            color=alt.Color('label:O')
        ).properties(width=800, height=800)
st.altair_chart(ch_alt, use_container_width=True)

