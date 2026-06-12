import torch
import math
import urllib.request

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

dtype = torch.float64 


urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/RezoanoorRahman/Infinitely-Wide-Deep-Neural-Network-in-Torch/main/bilinear_interpolation.py",
    "bilinear_interpolation.py"
)


from bilinear_interpolation import linear_interpolation_F_diag, bilinear_interpolation_F


###################################################################################


# Initial kernel assuming row-wise independence: K0(x,x') = sigma_b^2 + sigma_w^2 * <x,x'> / d_in

def K0(X1, X2, sigma_w2, sigma_b2, d_in):
    return sigma_b2 + sigma_w2 * (X1 @ X2.T) / d_in

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


"""
A closed form for iterative update of the kernels when using a tanh activation is not available. 
Lee et al.[2018] proposed a bilinear method to compute it. Please refer to the readme file for details.
"""

def tanh_nngp_kernel(
    X1, X2, depth,
    sigma_w2, sigma_b2,
    d_in,
    s_vec, c_vec,
    F_grid, F_diag
):
    device = X1.device # Using the default device used in X

    # Kernel for the initial layer
    def K0(Xa, Xb):
        return sigma_b2 + sigma_w2 * (Xa @ Xb.T) / d_in

    K = K0(X1, X2)

    # Layer-to-layer updater

    for _ in range(depth):

        n1, n2 = K.shape

        # Diagonal terms
        s1 = torch.diag(K0(X1, X1)) if n1 == X1.shape[0] else torch.diag(K) # For training X
        s2 = torch.diag(K0(X2, X2)) if n2 == X2.shape[0] else torch.diag(K.T) # For test X


        # Diagonal update using linear interpolation
        F_diag_vals1 = linear_interpolation_F_diag(s1, s_vec, F_diag)
        F_diag_vals2 = linear_interpolation_F_diag(s2, s_vec, F_diag)

        # --- prepare output ---
        K_updated = torch.zeros_like(K)

        # --- handle square case (most common) ---
        if n1 == n2:
            n = n1
            s_diag = torch.diag(K)

            # diagonal
            F_diag_vals = linear_interpolation_F_diag(s_diag, s_vec, F_diag)
            K_updated[range(n), range(n)] = F_diag_vals

            # off-diagonal
            S = s_diag.unsqueeze(1)
            C = K / (S + 1e-12)

            mask = ~torch.eye(n, dtype=torch.bool, device=device)

            s_mat = S.expand_as(K)[mask]
            c_mat = C[mask]

            F_vals = bilinear_interpolation_F(s_mat, c_mat, s_vec, c_vec, F_grid)

            K_updated[mask] = F_vals


        # Rectangular case needed to get the cross pairwise covariance between the training and test inputs (X),
        # since the input numbers are often different.
        else:
            # Kernel for test X
            S = s1.unsqueeze(1)                    # (n1,1)
            T = s2.unsqueeze(0)                    # (1,n2)

            norm = torch.sqrt(S * T).clamp_min(1e-12)
            C = K / norm

            s_mat = S.expand_as(K).reshape(-1)
            c_mat = C.reshape(-1)

            F_vals = bilinear_interpolation_F(s_mat, c_mat, s_vec, c_vec, F_grid)

            K_updated = F_vals.view(n1, n2)

        # Final update
        K = sigma_b2 + sigma_w2 * K_updated

    return K



#----------------------------------------------------------------------------------------#

"""
Chunked update for tanh.
The previous function works perfectly if the available GPU memory is enough, which is often not the case.
Here, we shall update the kernels chunk-wise.
This update is slower compared to the last one.
One can tweak 'chunk_size' according to available vRAM.
"""


def tanh_nngp_kernel_square_chunked(
    X,
    depth,
    sigma_w2,
    sigma_b2,
    d_in,
    s_vec,
    c_vec,
    F_grid,
    F_diag,
    chunk_size=512
):

    device = X.device # set the default device

    s_vec = s_vec.to(device)
    c_vec = c_vec.to(device)
    F_grid = F_grid.to(device)
    F_diag = F_diag.to(device)

    K = K0(X, X, sigma_w2, sigma_b2, d_in)

    n = K.shape[0]

    for _ in range(depth):

        s_diag = torch.diag(K)

        K_updated = torch.zeros_like(K)

        # diagonal update: c = 1 case
        diag_vals = linear_interpolation_F_diag(s_diag, s_vec, F_diag)
        idx = torch.arange(n, device=device)
        K_updated[idx, idx] = diag_vals

        # off-diagonal update in chunks
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)

            K_block = K[start:end, :]                 # (chunk, n)
            s_block = s_diag[start:end].unsqueeze(1)  # (chunk, 1)

            C_block = K_block / (s_block + 1e-12)

            mask = torch.ones_like(K_block, dtype=torch.bool, device=device)

            row_idx = torch.arange(end - start, device=device)
            col_idx = torch.arange(start, end, device=device)
            mask[row_idx, col_idx] = False

            s_mat = s_block.expand_as(K_block)[mask]
            c_mat = C_block[mask]

            F_vals = bilinear_interpolation_F(
                s_mat, c_mat, s_vec, c_vec, F_grid
            )

            block_updated = K_updated[start:end, :]
            block_updated[mask] = F_vals
            K_updated[start:end, :] = block_updated

            del K_block, s_block, C_block, mask, s_mat, c_mat, F_vals, block_updated

        # Apply the final affine transformation
        K = sigma_b2 + sigma_w2 * K_updated

    return K



'''
Now for the rectangle part
'''

def tanh_nngp_kernel_rect_chunked(
    X1, X2,
    depth,
    sigma_w2, sigma_b2,
    d_in,
    s_vec, c_vec,
    F_grid, F_diag,
    chunk_size=256
):
    device = X1.device

    # initial kernel 
    K = sigma_b2 + sigma_w2 * (X1 @ X2.T) / d_in

    # initial diagonals (fixed per dataset, then update layer wise) 
    s1 = torch.diag(sigma_b2 + sigma_w2 * (X1 @ X1.T) / d_in)  # dimension: (n1,) n1: row number for training
    s2 = torch.diag(sigma_b2 + sigma_w2 * (X2 @ X2.T) / d_in)  # (n2,)

    n1, n2 = K.shape

    for _ in range(depth):

        K_updated = torch.zeros_like(K)

        # Chunk-wise update over rows of X1
        for start in range(0, n1, chunk_size):
            end = min(start + chunk_size, n1)

            K_block = K[start:end, :]                 # (chunk, n2)
            s_block = s1[start:end].unsqueeze(1)      # (chunk, 1)

            # correlation
            norm = torch.sqrt(s_block * s2.unsqueeze(0)).clamp_min(1e-12)
            C_block = K_block / norm

            # Flatten for interpolation
            s_mat = s_block.expand_as(K_block).reshape(-1)
            c_mat = C_block.reshape(-1)

            # use lookup table to update
            F_vals = bilinear_interpolation_F(
                s_mat, c_mat, s_vec, c_vec, F_grid
            )

            K_updated[start:end, :] = F_vals.view(end - start, n2)

            # free up the memory
            del K_block, s_block, norm, C_block, s_mat, c_mat, F_vals

        # Affine transformation
        K = sigma_b2 + sigma_w2 * K_updated

        # Update diagonals
        s1 = sigma_b2 + sigma_w2 * linear_interpolation_F_diag(s1, s_vec, F_diag)
        s2 = sigma_b2 + sigma_w2 * linear_interpolation_F_diag(s2, s_vec, F_diag)

    return K