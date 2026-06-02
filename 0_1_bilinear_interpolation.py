import math
import torch
import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64  #Change format for Cholesky stability

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