import math
import torch
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda') # Can use 'mps' if you are using apple silicone, default id 'cpu'


print(device) # Check whether using the right hardware or not


# Density of the grid. Of course, a denser grid would make things more accurate.
n_g=1001 # For numerical approximation, make it dense for more accuracy.
n_v= 1001
n_c=1000

# Range values suggested by the authors
u_max= 11
s_max=100

# Create grids, u_vec is used for numerical approximation
u_vec = torch.linspace(-u_max, u_max, steps=n_g, device=device)
s_vec = torch.linspace(0.05, s_max, steps=n_v, device=device)
c_vec = torch.linspace(-0.999, 0.999, steps=n_c, device=device)

F_grid = torch.zeros(len(s_vec), len(c_vec), device=device)


# For parallal computation in gpu

u_vec = u_vec.to(device)

F_diag = torch.zeros(len(s_vec))
F_grid = torch.zeros(len(s_vec), len(c_vec))

F_diag = F_diag.to(device)
F_grid = F_grid.to(device)
pairs = torch.cartesian_prod(u_vec, u_vec) # Create grids of pairs of U as we need pairwise inputs for covariance values


ua, ub = torch.meshgrid(u_vec, u_vec, indexing='ij')

for i in range(len(s_vec)):
    s = s_vec[i]
    weights_diag = torch.exp(-.5* u_vec**2/s + 1e-12)

    expectation_approx_diag = torch.sum(torch.tanh(u_vec)**2 * weights_diag)
    density_approx_diag = torch.sum(weights_diag)

    F_diag[i]=expectation_approx_diag/density_approx_diag
    
    for j in range(len(c_vec)):
        sc = s * c_vec[j]

        Sigma = torch.stack([
            torch.stack([s, sc]),
            torch.stack([sc, s])
        ])

        inv_matrix = torch.linalg.inv(Sigma)

        quad = (
            inv_matrix[0,0]*ua**2 +
            2*inv_matrix[0,1]*ua*ub +
            inv_matrix[1,1]*ub**2
        )

        weights = torch.exp(-0.5 * quad)

        expectation_approx = torch.sum(torch.tanh(ua) * torch.tanh(ub) * weights)
        density_approx     = torch.sum(weights)

        F_grid[i,j] = expectation_approx / density_approx

# Save the final tables and values used as a .pt file for future use.
torch.save({
    'F_grid': F_grid,
    'F_diag': F_diag,
    's_vec' : s_vec,
    'c_vec' : c_vec
}, 'lookup_table_tanh.pt')