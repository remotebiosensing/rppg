import torch
import torch.nn as nn

class SSR(nn.Module):
    """
    This class implements the Spatial Subspace Rotation for Remote Photoplethysmography

    It is based on the work published in "A Novel Algorithm for Remote Photoplethysmography - Spatial Subspace Rotation",
    Wenjin Wang, Sander Stuijk, and Gerard de Haan, IEEE TRANSACTIONS ON BIOMEDICAL ENGINEERING, VOL. 63, NO. 9, SEPTEMBER 2016
    """

    def __init__(self):
        super(SSR, self).__init__()

        self.skin_filter = SkinColorFilter()

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
        images = torch.permute(images,(0,2,3,4,1))
        B, K, H, W, _ = images.size()
        l = fps  # The temporal stride to use

        # the pulse signal
        P = torch.zeros(B, K, dtype=torch.float64)  # dim: BxK

        # store the eigenvalues Λ and the eigenvectors U at each frame
        Λ = torch.zeros(B, 3, K, dtype=torch.float64)  # dim: Bx3xK
        U = torch.zeros(B, 3, 3, K, dtype=torch.float64)  # dim: Bx3x3xK


        for b in range(B):
            k = 0  # the number of frame elaborated

            for i in range(K):
                img = images[b, i]  # dim: HxWx3

                # detects objects of different sizes in the input image. The detected objects are returned as a list of rectangles.
                        # get: skin pixels
                V = self.__get_skin_pixels(img, k == 0)  # 3 | dim: (W×H)x3

                # build the correlation matrix
                C = self.__build_correlation_matrix(V)  # 3 | dim: 3x3

                # get: eigenvalues Λ, eigenvectors U
                Λ[b, :, k], U[b, :, :, k] = self.__eigs(C)  # 4 | dim Λ: 3 | dim U: 3x3

                # build p and add it to the pulse signal P
                if k >= l:  # 5
                    τ = k - l  # 5
                    p = self.__build_p(τ, k, l, U[b], Λ[b])  # 6, 7, 8, 9, 10, 11 | dim: l
                    P[b, τ:k] += p  # 11

                k = k + 1


        return P#k, P

    def __build_correlation_matrix(self, V):
        # V dim: (W×H)x3
        V_T = V.transpose(0, 1)  # dim: 3x(W×H)

        N = V.shape[0]

        # build the correlation matrix
        C = torch.matmul(V_T, V)  # dim: 3x3
        C = C / N

        return C

    def __get_skin_pixels(self, img, do_skininit):
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

        if do_skininit:
            self.skin_filter.estimate_gaussian_parameters(img)

        skin_mask = self.skin_filter.get_skin_mask(img)  # dim: wxh

        V = img[skin_mask]  # dim: (w×h)x3
        V = V  / 255.0

        return V

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
        Λ, U = torch.linalg.eig(C)  # dim Λ: 3 | dim U: 3x3

        idx = torch.argsort(Λ.abs())  # dim: 3x1
        idx = idx.flip(0)  # dim: 1x3

        Λ_ = Λ[idx]  # dim: 3
        U_ = U[:, idx]  # dim: 3x3

        return Λ_, U_

    def __build_p(self, τ, k, l, U, Λ):
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
        SR = torch.zeros((3, l), dtype=torch.float64)  # dim: 3xl
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
            x3 = torch.outer(e, g)
            x4 = d @ x3
            x5 = torch.outer(f, h)
            x6 = d @ x5
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

class SkinColorFilter():
    def __init__(self):
        self.mean = torch.tensor([0.0, 0.0])
        self.covariance = torch.zeros((2, 2), dtype=torch.float64)
        self.covariance_inverse = torch.zeros((2, 2), dtype=torch.float64)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __generate_circular_mask(self, image, radius_ratio=0.4):
        w = image.shape[0]
        h = image.shape[1]

        x_center = w / 2
        y_center = h / 2

        X = torch.arange(w, device=self.device).view(-1, 1).expand(w, h)
        Y = torch.arange(h, device=self.device).view(1, -1).expand(w, h)

        X = X -  x_center
        Y = Y -  y_center

        radius = radius_ratio * h

        cm = (X ** 2 + Y ** 2) < (radius ** 2)
        self.circular_mask = cm.to(self.device)

    def __remove_luma(self, image):
        R = 0.299 * image[..., 0]
        G = 0.587 * image[..., 1]
        B = 0.114 * image[..., 2]

        luma = R + G + B

        m = torch.mean(luma)
        s = torch.std(luma)

        lm = torch.logical_and((luma > (m - 1.5 * s)), (luma < (m + 1.5 * s)))
        self.luma_mask = lm.to(self.device)

    def __RG_Mask(self, image):
        channel_sum = image[..., 0].to(torch.float64) + image[..., 1] + image[..., 2]

        nonzero_mask = torch.logical_or(torch.logical_or(image[..., 0] > 0, image[..., 1] > 0), image[..., 2] > 0)

        R = torch.zeros_like(image[..., 0], dtype=torch.float64)
        R[nonzero_mask] = image[nonzero_mask][:, 0] / channel_sum[nonzero_mask]

        G = torch.zeros_like(image[..., 0], dtype=torch.float64)
        G[nonzero_mask] = image[nonzero_mask][:, 1] / channel_sum[nonzero_mask]

        return R, G

    def estimate_gaussian_parameters(self, image):
        self.__generate_circular_mask(image)
        self.__remove_luma(image)

        mask = torch.logical_and(self.luma_mask, self.circular_mask)

        R, G = self.__RG_Mask(image)

        self.mean = torch.tensor([torch.mean(R[mask]), torch.mean(G[mask])], dtype=torch.float64, device=self.device)

        R_minus_mean = R[mask] - self.mean[0]
        G_minus_mean = G[mask] - self.mean[1]

        samples = torch.stack((R_minus_mean, G_minus_mean), dim=1)

        cov = torch.matmul(samples.T, samples) / float(samples.shape[0] - 1)
        self.covariance = cov.to(torch.float64)

        if torch.det(self.covariance) != 0:
            self.covariance_inverse = torch.inverse(self.covariance)
        else:
            self.covariance_inverse = torch.zeros_like(self.covariance)

    def get_skin_mask(self, image, threshold=0.5):
        R, G = self.__RG_Mask(image)

        R_minus_mean = R - self.mean[0]
        G_minus_mean = G - self.mean[1]

        V = torch.stack((R_minus_mean, G_minus_mean), dim=2)
        V = V.view(-1, 2)

        probs = torch.matmul(V, torch.matmul(self.covariance_inverse, V.t())).diag()
        probs = probs.view(R.shape)

        skin_map = torch.exp(-0.5 * probs)

        return skin_map > threshold
