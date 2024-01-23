from datetime import datetime

import cv2
import os
import numpy as np
import torch
import torchvision
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torchvision.transforms import transforms

from utils import var_to_np
from tqdm import tqdm
import openTSNE

import sys
from utils import downsample_to_N, imscatter, date_string
from step_0_specify_dataset import SSLDataset, transform_at_prediction

# Initially I tried to use UMAP bot got error:
#import umap # UMAP (currently experiencing segfault - will be fixed soon (?) https://github.com/numba/numba/pull/7625/files)

#

#################### INSTRUCTION #######################

# path_to_pretrained_model = '/nr/samba/jo/pro/iari/usr/anders/code/PyTorch_BYOL/runs/Jan18_09-32-57_jo1.ad.nr.no/checkpoints/model.pth'
pretraing_type = 'DINOv2'  # What pretraining did you use? ('BYOL', 'SimCLR', 'ImageNet' (no path needed), 'DINOv2' (no path needed))

# PARAMETERS (that does not need to be changed)
GPU_NO = 1
N_pca_components_in_tsne = 100 # Number of components after PCA transform
tsne_perplexity = None #If you do not want to search for a good value, set this.
PATH = '.' #Where everything is stored


########################################################

## Load model and run prediction
if pretraing_type=='BYOL':
    model_conv = torchvision.models.resnet18(pretrained=False)
    weights = torch.load(path_to_pretrained_model)
    weights = {k.replace('encoder.',''):v for k,v in weights['online_network_state_dict'].items() if 'proj' not in k}
    model = torch.nn.Sequential(*list(model_conv.children())[:-1])
    model.load_state_dict(weights)
elif pretraing_type=='SimCLR':
    model_conv = torchvision.models.resnet18(pretrained=False)
    weights = torch.load(path_to_pretrained_model)['state_dict']
    weights = {k.replace('backbone.',''):v for k,v in weights.items()}
    weights = {k: v for k, v in weights.items() if 'fc.' not in k}
    model_conv.load_state_dict(weights,strict=False)
    model = model_conv
elif pretraing_type=='ImageNet':
    model_conv = torchvision.models.resnet18(pretrained=True)
    model = torch.nn.Sequential(*list(model_conv.children())[:-1]) #Remove ResNet head
    transform_at_prediction.transforms.append(        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
)
elif pretraing_type == 'DINOv2':
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
else:
    raise NotImplementedError(pretraing_type)


dataset = SSLDataset(transform_at_prediction)
now = datetime.now().strftime("%m.%d.%Y, %H:%M:%S")
OUT_FILE = os.path.join(PATH, 'tsne_{}_{}_{}.npz'.format(pretraing_type, date_string(), dataset.name))
train_loader = torch.utils.data.DataLoader( dataset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

preds = []
model.eval()
model.cuda(GPU_NO)
for batch in tqdm(train_loader, desc='Applying trained network to images', total=len(train_loader)):
    features = model(batch[0].cuda(GPU_NO))
    preds.append(var_to_np(features.squeeze()))
preds = np.concatenate(preds,0)


## Feature centering and withening + PCA transform

pca = PCA(n_components=preds.shape[1])

preds_normed = (preds-np.mean(preds,0, keepdims=True))/(np.std(preds,1, keepdims=True) + 10**-6)
preds_pca = pca.fit_transform(preds_normed)


# #Explore feature importance
# plt.plot(np.cumsum(pca.explained_variance_ratio_));
# plt.grid()
# plt.show()
#
# # Visualize some PCA features
# N = 5
# for i in range(N):
#     for j in range(N):
#         plt.subplot(N,N,i+1 + j*N)
#         plt.scatter(preds_pca[::N,i], preds_pca[::N,j])
# plt.show()



## Find best choice for perplexity

if tsne_perplexity is None:
    plt.figure(figsize=[30,15])
    for i, perplexity in enumerate([2, 5, 10, 30, 50, 100, 125, 150, 200, 300, 500, 1000]):
        tsne = openTSNE.TSNE(n_components=2, perplexity=perplexity, verbose=True, n_jobs=40, negative_gradient_method="fft", )
        embedding = tsne.fit(preds_pca[:,:N_pca_components_in_tsne])

        plt.subplot(2,6,i+1)
        plt.scatter(
            downsample_to_N(embedding[:, 0], 5000),
            downsample_to_N(embedding[:, 1], 5000),
            marker='.')
        plt.title('Perplexity {} - {}'.format(perplexity, dataset.name))
    plt.show()



    while tsne_perplexity is None:
        ask = input('Select your preferred perplexity (int)')
        try:
            tsne_perplexity = int(ask)
            print('Selected ')
            break
        except ValueError:
            print('Could not parse "{}" as int'.format(ask))
## Save clustered-points to a file that can be read by icat
tsne = openTSNE.TSNE(n_components=2, perplexity=tsne_perplexity, verbose=True, n_jobs=18, negative_gradient_method="fft", )
embedding = tsne.fit(preds_pca[:, :N_pca_components_in_tsne])


np.savez(OUT_FILE, xy=embedding, files=dataset.files, perplexity=tsne_perplexity)

## Visualize cluster
f = np.load(OUT_FILE)
xy = f['xy']
files = f['files']



scatter_img = imscatter(xy[:,0], xy[:,1], images = files, n_imgs_pr_axis=100, image_resolution=100)
cv2.imwrite(OUT_FILE.split('.npz')[0] + '.jpg', cv2.cvtColor(scatter_img, cv2.COLOR_RGB2BGR))
plt.figure(figsize=[15,15])
plt.imshow(scatter_img);
plt.tight_layout()
plt.show()


##

