import torch
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64  #Change format for Cholesky stability


"""
For sequential update of kernels for ReLu acivation has a closed form provided by Cho and Saul [2009]. 
Please refer to the readme file for more details.
"""

def relu_nngp_kernel(X1, X2, depth, sigma_w2, sigma_b2, d_in):
    K = K0(X1, X2, sigma_w2, sigma_b2, d_in)

    K11_diag = torch.diag(K0(X1, X1, sigma_w2, sigma_b2, d_in))
    K22_diag = torch.diag(K0(X2, X2, sigma_w2, sigma_b2, d_in))

    for _ in range(depth):
        norm_mat = torch.sqrt(torch.outer(K11_diag, K22_diag)).clamp_min(1e-12)
        cos_theta = (K / norm_mat).clamp(-1.0, 1.0)
        theta = torch.arccos(cos_theta)

        K = sigma_b2 + (sigma_w2 / (2.0 * math.pi)) * norm_mat * (
            torch.sin(theta) + (math.pi - theta) * torch.cos(theta)
        ) 

        K11_diag = sigma_b2 + 0.5 * sigma_w2 * K11_diag
        K22_diag = sigma_b2 + 0.5 * sigma_w2 * K22_diag

    return K