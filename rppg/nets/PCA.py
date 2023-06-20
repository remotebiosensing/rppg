import torch
from sklearn.decomposition import PCA as skpca

class PCA(torch.nn.Module):
    def __init__(self):
        super(PCA, self).__init__()


    def forward(self,x):
        batch, _, _, _, _ = x.shape
        x = torch.mean(x,dim=(3,4))
        bvp = []
        for i in range(batch):
            X = x[i]
            pca = skpca(n_components=3)
            pca.fit(X.cpu().numpy())
            bvp.append(torch.from_numpy(pca.components_[1] * pca.explained_variance_[1]).to(x.device))
        bvp = torch.stack(bvp)
        return bvp