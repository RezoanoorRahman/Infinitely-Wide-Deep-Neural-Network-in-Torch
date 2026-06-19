import torch
import math

#from bilinear_interpolation import linear_interpolation_F_diag, bilinear_interpolation_F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64  #Change format for Cholesky stability


####################################################################################################################################

### Bilinear interpolation codes

def bilinear_interpolation_F(s,c, s_vec, c_vec, F_grid):
 
    # For using the specific device
    s=s.to(device)
    c=c.to(device)
    s_vec=s_vec.to(device)
    c_vec=c_vec.to(device)
    F_grid=F_grid.to(device)

    # Find the approximate positin of s inside s_vec
    # Same for c and c_vec
    i = torch.searchsorted(s_vec, s) - 1
    j = torch.searchsorted(c_vec, c) - 1
    
    # Safety layer: making the indices nonnegative    
    i = torch.clamp(i, 0, len(s_vec) - 2)
    j = torch.clamp(j, 0, len(c_vec) - 2)
    
    # Selecting indices to interpolate
    s0, s1 = s_vec[i], s_vec[i+1]
    c0, c1 = c_vec[j], c_vec[j+1]

    # Get corresponding values from the F lookup table which 
    F00 = F_grid[i, j]
    F10 = F_grid[i+1, j]
    F01 = F_grid[i, j+1]
    F11 = F_grid[i+1, j+1]

    # Get linear interpolation weighrs
    alpha = (s - s0) / (s1 - s0 + 1e-12) # Forcing the denominator to be positive in case s1==s0
    beta = (c - c0) / (c1 - c0 + 1e-12)

    # Compute F-values
    F_val = ((1 - alpha) * (1 - beta) * F00 +
             alpha * (1 - beta) * F10 +
             (1 - alpha) * beta * F01 +
             alpha * beta * F11
    )
    
    return F_val


### Linear interpolation for diagonal terms of the kernel. Same logic like before.
## Bilinear interpolation is not valid if c==1

def linear_interpolation_F_diag(s, s_vec, F_diag):

    s=s.to(device)
    s_vec=s_vec.to(device)
    F_diag=F_diag.to(device)

    i = torch.searchsorted(s_vec, s) - 1 
    i = torch.clamp(i, 0, len(s_vec) - 2)
    
    s0, s1 = s_vec[i], s_vec[i+1]
    
    F0 = F_diag[i]
    F1 = F_diag[i+1]
    
    alpha = (s - s0) / (s1 - s0 + 1e-12)
 
    F_val = (1 - alpha)  * F0 + alpha * F1
    
    return F_val

###################################################################################














# Initial kernel assuming row-wise independence: K0(x,x') = sigma_b^2 + sigma_w^2 * <x,x'> / d_in

def K0(X1, X2, sigma_w2, sigma_b2, d_in):
    return sigma_b2 + sigma_w2 * (X1 @ X2.T) / d_in


#----------------------------------------------------------------------------------------#

"""
Chunked update for tanh.
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

        # Apply the final linear transformation
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

        # --- chunk over rows of X1 ---
        for start in range(0, n1, chunk_size):
            end = min(start + chunk_size, n1)

            K_block = K[start:end, :]                 # (chunk, n2)
            s_block = s1[start:end].unsqueeze(1)      # (chunk, 1)

            # correlation
            norm = torch.sqrt(s_block * s2.unsqueeze(0)).clamp_min(1e-12)
            C_block = K_block / norm

            # flatten for interpolation
            s_mat = s_block.expand_as(K_block).reshape(-1)
            c_mat = C_block.reshape(-1)

            # lookup
            F_vals = bilinear_interpolation_F(
                s_mat, c_mat, s_vec, c_vec, F_grid
            )

            K_updated[start:end, :] = F_vals.view(end - start, n2)

            # free memory
            del K_block, s_block, norm, C_block, s_mat, c_mat, F_vals

        # --- scale ---
        K = sigma_b2 + sigma_w2 * K_updated

        # --- update diagonals (CRITICAL for correctness) ---
        s1 = sigma_b2 + sigma_w2 * linear_interpolation_F_diag(s1, s_vec, F_diag)
        s2 = sigma_b2 + sigma_w2 * linear_interpolation_F_diag(s2, s_vec, F_diag)

    return K