# 1 Overview

We implemented the Neural Network Gaussian Process (NNGP) method proposed by Lee et al. (2018) using *PyTorch*. Additionally, we implemented a chunk-wise kernel update strategy to bypass memory allocation bottlenecks during large-scale kernel computations.

## 1.1 Neural Networks as Gaussian Processes

Neural network Gaussian processes (NNGPs) are based on a surprising connection between deep neural networks and Gaussian processes. Neal (1996) showed that a single-hidden-layer neural network converges to a Gaussian process as the number of hidden neurons approaches infinity. Lee et al. (2018) extended this result to deep neural networks, showing that the same behavior holds when the width of every hidden layer tends to infinity.

Intuitively, instead of learning a single function by optimizing network weights, an NNGP defines a probability distribution over functions. The network architecture and activation function determine the covariance structure of this distribution.

A key advantage of NNGPs over standard multilayer perceptrons (MLPs) is that they naturally provide predictive uncertainty estimates. Lee et al. (2018) showed that these uncertainty estimates are strongly correlated with prediction error. A limitation, however, is that model performance depends heavily on hyperparameter choices, making external validation essential.

For a given set of hyperparameters ($\sigma_w^2$, $\sigma_b^2$, $\sigma_\epsilon^2$), the overall process can be summarized as follows:

- Normalize each input vector so that $||x_i|| = 1$ for all $i$.
- Construct the initial kernel matrix using

$$
K^{0}(x,x')=\sigma_b^2 + \sigma_w^2\frac{x^\top x'}{d_{\text{in}}}.
$$

- For each layer $l=1,\dots,L$ and each pair of inputs $(x,x')$, update the kernel recursively as

$$
K^{l}(x,x')=\sigma_b^2 + \sigma_w^2 \mathbb{E}_{(u,v)\sim\mathcal N(0,\Sigma)} \big[\phi(u)\phi(v)\big].
$$

- Repeat this process to compute the training-training, training-test, and test-test kernels:

$$K_{DD}=K^L(X_{\mathrm{train}}, X_{\mathrm{train}})$$
$$K_{*D}=K^L(X*, X_{\mathrm{train}})$$
$$K_{**}=K^L(X*, X*).$$

- Use standard Gaussian process regression formulas to compute the predictive mean and covariance for the test set:



$$
\bar \mu = K_{*D}
(K_{DD}+\sigma_\epsilon^2 I)^{-1} t, \text{ and}
$$


$$
\bar K = K_{**} - K_{*D} (K_{DD} + \sigma^2_{\epsilon} I)^{-1} K_{*D}^T
$$

![Infinite-width neural network](images/flow.png)

*Figure 1. Flow of iterative update of Kernels for Gaussian Process.*

## 1.2 Numerical Calculation of the Expectation

For the ReLU activation function, Cho and Saul (2009) derived a closed-form expression for the layer-to-layer kernel update. For most other activation functions, however, this expectation cannot be computed analytically and must instead be evaluated numerically.

A naive implementation computes this expectation separately for every pair of inputs at every layer, leading to a computational complexity of

$$
\mathcal{O}\left(n_g^2 L (n_{\mathrm{train}}^2+n_{\mathrm{train}}n_{\mathrm{test}})\right),
$$

where $n_{\mathrm{train}}$, $n_{\mathrm{test}}$, and $n_g$ denote the number of training samples, test samples, and integration grid points, respectively.

To address this issue, Lee et al. (2018) proposed a bilinear interpolation-based lookup method that reduces the computational complexity to

$$
\mathcal{O}\left(n_g^2 n_v n_c + L(n_{\mathrm{train}}^2+n_{\mathrm{train}}n_{\mathrm{test}})\right),
$$

where $n_v$ and $n_c$ are the grid sizes for variance and correlation values. The key idea is that the expensive Gaussian expectations are computed only once and then reused throughout the kernel recursion.

In this project, we precomputed the lookup table for the `tanh()` activation function, further reducing the online computational cost to

$$
\mathcal{O}\left(n_v n_c + L(n_{\mathrm{train}}^2+n_{\mathrm{train}}n_{\mathrm{test}})\right).
$$

The implementation is available in [grid_calculation.py](https://raw.githubusercontent.com/RezoanoorRahman/Infinitely-Wide-Deep-Neural-Network-in-Torch/main/scripts/grid_calculation.py), which can be adapted to generate lookup tables for other activation functions.

For a detailed theoretical treatment, see [Deep_NNGP.pdf](https://github.com/RezoanoorRahman/Infinitely-Wide-Deep-Neural-Network-in-Torch/blob/main/Deep_NNGP.pdf).




# 2 Step by step implementation

## 2.1 Load required libraries and download pre-computed lookup table

First load the required libraries and set up the device. If *CUDA* is available, we can speed-up the implementation process by using it.

```python
import torch
import math
import urllib.request


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

dtype = torch.float64  #Change format, can use float32 or float16 also

print(device)
```


Then download pre-tained lookup table and the corresponding gird of variance and correlations if $Tanh()$ is used as the activation function. One can skip this step if they are using the $Sigmoid()$ activation function.


```python
url = "https://raw.githubusercontent.com/RezoanoorRahman/Infinitely-Wide-Deep-Neural-Network-in-Torch/main/lookup_table_tanh.pt"

data = torch.hub.load_state_dict_from_url(url,
    map_location=torch.device(device))

F_grid = data['F_grid']
F_diag = data['F_diag']
s_vec = data['s_vec']
c_vec = data['c_vec']

```


Now let's load the functions to update iterative update kernels and 


```python
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/RezoanoorRahman/Infinitely-Wide-Deep-Neural-Network-in-Torch/main/kernel_updater.py",
    "kernel_updater.py"
)

from kernel_updater import relu_nngp_kernel, tanh_nngp_kernel_square_chunked, tanh_nngp_kernel_rect_chunked

```
A short description of the kernel updater functions are provided in [Appendix](#appendix).




## 2.2 Training 

Using these functions, we can write the following function that takes Training data $D_{Train}=(X_{Train}, y_{Train})$ and Test inputs $X_{Test}$ for fixed hyperparameters:  $\sigma^2_w, \sigma^2_b, \sigma^2_e$ and $L (Depth)$ for $ReLu()$ activation and provides predicted mean and variances.



```python
def get_mean_cov_relu(sigma_w2, sigma_b2, noise, 
    X_train, X_test, 
    y_train):

    d_in = X_train.shape[1]

    # Train
    K_DD = relu_nngp_kernel(X_train, X_train, depth, sigma_w2, sigma_b2, d_in)

    # add noise + jitter
    K_DD = K_DD + (noise + 1e-3) * torch.eye(K_DD.shape[0], device=device)

    # Solve
    L = torch.linalg.cholesky(K_DD)

    T_train = make_targets(y_train, dtype=dtype)  # shape (n_train, 10)
    alpha = torch.cholesky_solve(T_train, L)   # (n_train, 10)

    # Test
    K_starD = relu_nngp_kernel(X_test, X_train, depth, sigma_w2, sigma_b2, d_in)

    mu_test = K_starD @ alpha # shape (n_test, 10)

    K_star_star = relu_nngp_kernel(X_test, X_test, depth, sigma_w2, sigma_b2, d_in)

    # V = torch.linalg.inv(L) @ K_starD.T

    # Don't use that cause not numerically stable and slower

    V = torch.linalg.solve_triangular(L, K_starD.T,
                                      upper=False)

    variances = torch.diag(K_star_star - (V.T @ V)) 

    return mu_test, variances

```


Same for $Tanh()$ kernel. The core difference between this function and the last one is that during the Kernel building process, we used one extra argument `chunk_size` which indicates how many components of the kernels will get updated at a single time.




```python
def get_mean_cov_tanh(sigma_w2, sigma_b2, noise, 
    X_train, X_test, 
    y_train):

    d_in = X_train.shape[1]

    # Train
    K_DD = tanh_nngp_kernel_square_chunked(X_train, depth, sigma_w2, sigma_b2, d_in,
        s_vec, c_vec, F_grid, F_diag,
        chunk_size=chunk_size)


    # add noise + jitter
    K_DD = K_DD + (noise + 1e-3) * torch.eye(K_DD.shape[0], device=device)

    # Solve
    L = torch.linalg.cholesky(K_DD)

    T_train = make_targets(y_train, dtype=dtype)  # shape (n_train, 10)
    alpha = torch.cholesky_solve(T_train, L)   # (n_train, 10)

    # Test
    chunk = 256
    mu_list = []

    for i in range(0, X_test.shape[0], chunk):
        X_chunk = X_test[i:i+chunk]

        K_starD_chunk = tanh_nngp_kernel_rect_chunked(X_chunk, X_train, depth, sigma_w2, sigma_b2, d_in,
            s_vec, c_vec, F_grid, F_diag, chunk_size=chunk_size)

        mu_chunk = K_starD_chunk @ alpha
        mu_list.append(mu_chunk)

    mu_test = torch.cat(mu_list, dim=0)

    K_star_star = K_DD = tanh_nngp_kernel_square_chunked(X_test, depth, sigma_w2, sigma_b2, d_in,
        s_vec, c_vec, F_grid, F_diag,
        chunk_size=chunk_size)

    V = torch.linalg.solve_triangular(L, K_starD.T,
                                      upper=False)

    variances = torch.diag(K_star_star - (V.T @ V)) 

    return mu_test, variances
```


# 3 Conclusion

The performance of this method is heavily dependent on the choice of hyperparameters as this method doesn't implicitly learn/update them. As a result, external validation to choose the values of hyperparameters is necessary. For example, we implemented **10-Fold cross validation** method to choose the best set for implementing a **multi-class classifier on the MNIST dataset**. The full process is available in the notebook [Test_MNIST.ipynb](https://github.com/RezoanoorRahman/Infinitely-Wide-Deep-Neural-Network-in-Torch/blob/main/Test_MNIST.ipynb). Eventually, it yields an accuracy of $96.1\%$ for $Sigmoid()$ activation and number of hidden layers $L=20$. For $Tanh()$ activation, the accuracy drops to $95.6\%$








# References

- Lee, J., Bahri, Y., Novak, R., Schoenholz, S. S., Pennington, J., & Sohl-Dickstein, J. (2018). *Deep neural networks as Gaussian processes*. In *International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/1711.00165

- Neal, R. M. (1996). *Bayesian Learning for Neural Networks*. Springer. https://doi.org/10.1007/978-1-4612-0745-0


-------


# Appendix

**relu_nngp_kernel(X1, X2, depth, sigma_w2, sigma_b2, d_in):**


- Can be used to build $K_{DD}$, $K_{*D}$ and $K_{**}$ for $Sigmoid()$ as activation.

- Parameters:

| Argument | Type | Description |
|-----------|------|-------------|
| `X1` | `torch.Tensor` or `numpy.ndarray` | Test input matrix of shape `(n1, d_in)`. |
| `X2` | `torch.Tensor` or `numpy.ndarray` | Training input matrix of shape `(n2, d_in)`. |
| `depth` | `int` | Number of hidden layers in the corresponding neural network. |
| `sigma_w2` | `float` | Variance of the weight initialization. |
| `sigma_b2` | `float` | Variance of the bias initialization. |
| `d_in` | `int` | Input dimension used to normalize the first-layer kernel. |

- Return the final kernel of shape `(n1 , n1)`.


**tanh_nngp_kernel_square_chunked(X, depth, sigma_w2, sigma_b2, d_in, s_vec, c_vec, F_grid, F_diag, chunk_size=512)**


- Can be used to build $K_{DD}$, and $K_{**}$ for $Tanh()$ as activation.

- Parameters: 

| Argument | Type | Description |
|-----------|------|-------------|
| `X` | `torch.Tensor` or `numpy.ndarray` | Input matrix of shape `(n, d_in)`. |
| `depth` | `int` | Number of hidden layers in the corresponding neural network. |
| `sigma_w2` | `float` | Variance of the weight initialization. |
| `sigma_b2` | `float` | Variance of the bias initialization. |
| `d_in` | `int` | Input dimension used to normalize the first-layer kernel. |
| `s_vec` | `torch.Tensor` or `numpy.ndarray` | One-dimensional grid of marginal variances used for lookup interpolation. |
| `c_vec` | `torch.Tensor` or `numpy.ndarray` | One-dimensional grid of correlations in the interval `[-1, 1]` used for lookup interpolation. |
| `F_grid` | `torch.Tensor` or `numpy.ndarray` | Precomputed lookup table containing values of the Gaussian expectation $F_{\phi}(s,c)$ for the `tanh` activation. Shape `(len(s_vec), len(c_vec))`. |
| `F_diag` | `torch.Tensor` or `numpy.ndarray` | Precomputed lookup table for diagonal kernel updates corresponding to the case `x = x'`. Shape `(len(s_vec),)`. |
| `chunk_size` | `int`, optional | Number of rows processed simultaneously during pairwise kernel computation. Default is `512`. |

- Return the final kernel of shape `(n , n)`.





**tanh_nngp_kernel_rect_chunked(X1, X2, depth, sigma_w2, sigma_b2, d_in, s_vec, c_vec, F_grid, F_diag, chunk_size=256):**

- Can be used to build $K_{*D}$ for $Tanh()$ as activation.

- Parameters:


| Argument | Type | Description |
|-----------|------|-------------|
| `X1` | `torch.Tensor` or `numpy.ndarray` | First input matrix of shape `(n_1, d_in)`. |
| `X2` | `torch.Tensor` or `numpy.ndarray` | Second input matrix of shape `(n_2, d_in)`. |
| `depth` | `int` | Number of hidden layers in the corresponding neural network. |
| `sigma_w2` | `float` | Variance of the weight initialization. |
| `sigma_b2` | `float` | Variance of the bias initialization. |
| `d_in` | `int` | Input dimension used to normalize the first-layer kernel. |
| `s_vec` | `torch.Tensor` or `numpy.ndarray` | One-dimensional grid of marginal variances used for lookup interpolation. |
| `c_vec` | `torch.Tensor` or `numpy.ndarray` | One-dimensional grid of correlations in the interval `[-1, 1]` used for lookup interpolation. |
| `F_grid` | `torch.Tensor` or `numpy.ndarray` | Precomputed lookup table containing values of the Gaussian expectation $F_{\phi}(s,c)$ for the `tanh` activation. Shape `(len(s_vec), len(c_vec))`. |
| `F_diag` | `torch.Tensor` or `numpy.ndarray` | Precomputed lookup table for diagonal kernel updates corresponding to the case `x = x'`. Shape `(len(s_vec),)`. |
| `chunk_size` | `int`, optional | Number of rows processed simultaneously during pairwise kernel computation. Default is `256`. |


- Return the final kernel of shape `(n1 , n2)`.


