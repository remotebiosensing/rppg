import torch
import torch.nn as nn

class SSR(nn.Module):
    """
    This class implements the Spatial Subspace Rotation for Remote Photoplethysmography

    It is based on the work published in "A Novel Algorithm for Remote Photoplethysmography - Spatial Subspace Rotation",
    Wenjin Wang, Sander Stuijk, and Gerard de Haan, IEEE TRANSACTIONS ON BIOMEDICAL ENGINEERING, VOL. 63, NO. 9, SEPTEMBER 2016
    Source code for bob.rppg.ssr.ssr_utils
    """

    def __init__(self):
        super(SSR, self).__init__()

    def __build_p(self,τ, k, l, U, Λ):
        """
        builds P
        Parameters
        ----------
        k: int
            The frame index
        l: int
            The temporal stride to use
        U: torch.Tensor
            The eigenvectors of the c matrix (for all frames up to counter).
        Λ: torch.Tensor
            The eigenvalues of the c matrix (for all frames up to counter).
        Returns
        -------
        p: torch.Tensor
            The p signal to add to the pulse.
        """
        # SR'
        SR = torch.zeros((U.size(0), l), dtype=U.dtype, device=U.device)  # dim: 3xl
        z = 0

        for t in range(τ, k, 1):  # 6, 7
            a = Λ[0, t]
            b = Λ[1, τ]
            c = Λ[2, τ]
            d = U[:, 0, t]
            e = U[:, 1, τ]
            f = U[:, 2, τ]
            g = U[:, 1, τ]
            h = U[:, 2, τ]
            x1 = a / b
            x2 = a / c
            x3 = e.unsqueeze(1) * g.unsqueeze(0)
            x4 = torch.matmul(d, x3.unsqueeze(-1)).squeeze(-1)
            x5 = f.unsqueeze(1) * h.unsqueeze(0)
            x6 = torch.matmul(d, x5.unsqueeze(-1)).squeeze(-1)
            x7 = torch.sqrt(x1)
            x8 = torch.sqrt(x2)
            x9 = x7 * x4
            x10 = x8 * x6
            x11 = x9 + x10
            SR[:, z] = x11  # 8 | dim: 3
            z += 1

        # build p and add it to the final pulse signal
        s0 = SR[0, :]  # dim: l
        s1 = SR[1, :]  # dim: l
        p = s0 - ((torch.std(s0) / torch.std(s1)) * s1)  # 10 | dim: l
        p = p - torch.mean(p)  # 11
        return p  # dim: l

    def __build_correlation_matrix(self,V):
        # V dim: (W×H)x3
        V = V.view(-1,3)
        V_T = V.transpose(0, 1)  # dim: (W×H)x3
        N = V.size(0)
        # build the correlation matrix
        C = torch.matmul(V_T, V)  # dim: 3x3
        C = C / N

        return C

    def __eigs(self, C):
        """
        get eigenvalues and eigenvectors, sort them.
        Parameters
        ----------
        C: torch.Tensor
            The RGB values of skin-colored pixels.
        Returns
        -------
        Λ: torch.Tensor
            The eigenvalues of the correlation matrix
        U: torch.Tensor
            The (sorted) eigenvectors of the correlation matrix
        """
        # get eigenvectors and sort them according to eigenvalues (largest first)
        L, U = torch.linalg.eigh(C)  # dim Λ: 3 | dim U: 3x3
        idx = torch.argsort(L)  # dim: 3x1
        idx = idx.flip(0)  # dim: 1x3
        L_ = L[idx]  # dim: 3
        U_ = U[:, idx]  # dim: 3x3

        return L_, U_

    def forward(self, images, fps = 30):
        """
        Parameters
        ----------
        images: torch.Tensor | dim: BxHxWx3
            The images to elaborate

        fps: int
            Frame per seconds

        Returns
        -------
        k : int
            The number of frame elaborated

        P: torch.Tensor | dim: BxK
            The pulse signal
        """
        fps = fps

        raw_sig = images
        raw_sig = torch.permute(raw_sig,(0,2,3,4,1))
        B, K, h, w, c = raw_sig.size()
        l = int(fps)

        P = torch.zeros(B, K, dtype=raw_sig.dtype, device=raw_sig.device)  # 1 | dim: BxK
        # store the eigenvalues Λ and the eigenvectors U at each frame
        L = torch.zeros(B, 3, K, dtype=torch.float32, device=raw_sig.device)  # dim: Bx3xK
        U = torch.zeros(B, 3, 3, K, dtype=torch.float32, device=raw_sig.device)  # dim: Bx3x3xK

        for b in range(B):
            for k in range(K):
                n_roi = len(raw_sig[b, k])
                VV = []
                V = raw_sig[b, k].float()
                idx = V != 0
                idx2 = torch.logical_and(torch.logical_and(idx[:, :, 0], idx[:, :, 1]), idx[:, :, 2])
                # V_skin_only = torch.masked_select(V,idx2)
                idx2_expanded = idx2.unsqueeze(-1).expand(-1, -1, 3)
                V_skin_only = torch.masked_select(V, idx2_expanded).view(h,w,c)
                VV.append(V_skin_only)

                VV = torch.vstack(VV)

                C = self.__build_correlation_matrix(VV)  # dim: 3x3

                # get: eigenvalues Λ, eigenvectors U
                L[b, :, k], U[b, :, :, k] = self.__eigs(C)  # dim Λ: 3 | dim U: 3x3

                # build p and add it to the pulse signal P
                if k >= l:  # 5
                    tau = k - l  # 5
                    p = self.__build_p(tau, k, l, U[b], L[b])  # 6, 7, 8, 9, 10, 11 | dim: l
                    P[b, tau:k] += p  # 11

                if torch.isnan(torch.sum(P[b])):
                    print('NAN')
                    print(raw_sig[b, k])

        # bvp = P.unsqueeze(2)

        return P
