def tanh_nngp_kernel(
    X1, X2, depth,
    sigma_w2, sigma_b2,
    d_in,
    s_vec, c_vec,
    F_grid, F_diag
):
    device = X1.device

    ### Initial kernel
    def K0(Xa, Xb):
        return sigma_b2 + sigma_w2 * (Xa @ Xb.T) / d_in

    K = K0(X1, X2)

    for _ in range(depth):

        n1, n2 = K.shape

        s1 = torch.diag(K0(X1, X1)) if n1 == X1.shape[0] else torch.diag(K)
        s2 = torch.diag(K0(X2, X2)) if n2 == X2.shape[0] else torch.diag(K.T)


        F_diag_vals1 = linear_interpolation_F_diag(s1, s_vec, F_diag)
        F_diag_vals2 = linear_interpolation_F_diag(s2, s_vec, F_diag)

        K_updated = torch.zeros_like(K)

        if n1 == n2:
            n = n1
            s_diag = torch.diag(K)

            F_diag_vals = linear_interpolation_F_diag(s_diag, s_vec, F_diag)
            K_updated[range(n), range(n)] = F_diag_vals

            S = s_diag.unsqueeze(1)
            C = K / (S + 1e-12)

            mask = ~torch.eye(n, dtype=torch.bool, device=device)

            s_mat = S.expand_as(K)[mask]
            c_mat = C[mask]

            F_vals = bilinear_interpolation_F(s_mat, c_mat, s_vec, c_vec, F_grid)

            K_updated[mask] = F_vals

        else:
            S = s1.unsqueeze(1)                    # (n1,1)
            T = s2.unsqueeze(0)                    # (1,n2)

            norm = torch.sqrt(S * T).clamp_min(1e-12)
            C = K / norm

            s_mat = S.expand_as(K).reshape(-1)
            c_mat = C.reshape(-1)

            F_vals = bilinear_interpolation_F(s_mat, c_mat, s_vec, c_vec, F_grid)

            K_updated = F_vals.view(n1, n2)

        K = sigma_b2 + sigma_w2 * K_updated

    return K