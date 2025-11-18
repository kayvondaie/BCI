import numpy as np
import matplotlib.pyplot as plt

# clear

labels = ['Prep+strong', 'Prep', 'RNN']
for ii in range(1, 2):  # for ii = 1:1;
    for fit_rep in range(1, 2):  # for fit_rep = 1:1;

        # keep ii labels L_mat fit_rep
        plt.figure(1)
        plt.clf()

        N = 100
        dt = 0.01
        T = np.arange(0, 2 + dt, dt)  # 0:dt:2;
        tau = 0.3
        etas = np.array([0.01, 0.05])

        pre_cells = np.arange(1, 21)   # 1:20
        trl_cells = np.arange(21, 81)  # 21:80
        Nt = len(trl_cells)
        Np = len(pre_cells)

        fun = np.tanh
        der = lambda x: 1 - np.tanh(x) ** 2

        # fun = @(x) x;
        # der = @(x) 1;

        in_idx = np.arange(1, N + 1)  # in = 1:N;
        # Use 1D arrays for simplicity; MATLAB had (N,1)
        w_in = np.zeros(N)
        # w_in(in) = randn(1,length(in))*0;
        w_in[in_idx - 1] = np.random.randn(len(in_idx)) * 0

        in2 = np.arange(1, N + 1)  # in2 = 1:N;
        w_in2 = np.zeros(N)
        w_in2[0:20] = 0.1  # w_in2(1:20) = .1;
        w_in[76] = 0       # w_in(77) = 0;
        w_out = np.random.randn(1, N)
        x = np.arange(1, N + 1)  # x = 1:N;
        mask = np.random.rand(N, N) > 0.00

        T_trl = np.array([[0, 10],
                          [0, 0]])
        # U_trl = Gram_Schmidt_Process([linspace(-1,1,Nt)' ones(Nt,1)]);
        # --- Improvisation: Gram-Schmidt via QR (matches intent) ---
        def Gram_Schmidt_Process(A):
            Q, _ = np.linalg.qr(A)
            return Q

        U_trl = Gram_Schmidt_Process(np.column_stack((np.linspace(-1, 1, Nt), np.ones(Nt))))
        W_trl = (U_trl @ T_trl @ U_trl.T).T
        W_pre = np.random.rand(Np, Np)
        W_pre = W_pre / np.max(np.real(np.linalg.eigvals(W_pre))) * (-1)
        W_tp = np.random.rand(Nt, Np) / 5
        W = np.zeros((N, N))
        # W(trl_cells,trl_cells) = W_trl;
        # W(trl_cells,pre_cells) = 1;
        # Use np.ix_ to emulate MATLAB W(rows, cols)
        W[np.ix_(trl_cells - 1, trl_cells - 1)] = W_trl
        W[np.ix_(trl_cells - 1, pre_cells - 1)] = 1
        W[59:80, 0:20] = 0.2    # W(60:80,1:20) = .2;
        W[80:100, 59:80] = 0.2  # W(81:N,60:80) = .2;
        if ii == 1:
            W[96, np.arange(84, 100)] = 10  # W(97,[85:100]) = 10;

        w = W.copy()
        if ii == 3:
            w = np.random.randn(N, N) / np.sqrt(N)
        w = w / 20
        # imagesc(W);colormap(bluewhitered)

        iters = 20
        learning_rule = 1
        eta = etas[learning_rule - 1]  # eta = etas(learning_rule);
        rules = ['backprop', 'Hebbian']
        print(rules[learning_rule - 1])  # disp(rules{learning_rule})
        real_cn = 1

        # --- Improvisation: preallocate histories like MATLAB grows them ---
        R = np.zeros((N, len(T), iters))        # R(:,:,iter) = h;
        W_hist = np.zeros((N, N, iters))        # W(:,:,iter) = w; (renamed to avoid clobbering W)

        for iter in range(1, iters + 1):  # for iter = 1:iters;
            print(iter)
            x = np.zeros_like(T); x2 = x.copy()    # x = 0*T;x2=x;
            x[99:] = 1                              # x(100:end) = 1;
            x2[9:100] = 1                           # x2(10:100) = 1;
            h = np.zeros((N, 1))
            target = x.copy()
            target = target[:-1] * 1
            target = target / np.max(target)

            # Forward Pass (Equation 1 from Murray elife 2019)
            y = np.zeros(len(T) - 1)
            h = np.zeros((N, len(T)))  # allocate so h[:, i+1] valid
            for i in range(len(T) - 1):  # for i = 1:length(T)-1;
                y[i] = w_out @ h[:, i]
                h[:, i + 1] = (
                    h[:, i]
                    + dt / tau
                    * (
                        -h[:, i]
                        + fun(w @ h[:, i] + x[i + 1] * w_in)
                        + x2[i + 1] * w_in2
                        + np.random.randn(N) * 0.01
                    )
                )

            if (iter > 10) and (iter < iters):  # if iter > 10 & iter < iters

                # Backward Pass (Equation 20 from Murray elife 2019);
                w_back = w_out.copy()
                e = target - y
                e2 = target / 2 - h[real_cn - 1, 0:len(T) - 1]
                e2[0:100] = 0
                z = np.zeros((N, len(T)))
                z[:, len(T) - 1] = w_back.flatten() * e[-1]  # z(:,length(T)) = w_back'*e(end);

                for t in range(len(T) - 2, -1, -1):  # for t = length(T)-1:-1:1;
                    u = w @ h[:, t] + x[t + 1] * w_in
                    for i in range(N):  # for i = 1:N;
                        # z(i,t) = (1-dt/tau)*z(i,t+1) + dt/tau * sum((z(:,t+1).*der(u))'*w(:,i)) + w_back(i)'*e(t)*10 + w_back(i)'*e2(t)*0;
                        z[i, t] = (
                            (1 - dt / tau) * z[i, t + 1]
                            + dt / tau * np.sum(z[:, t + 1] * der(u) * w[:, i])
                            + w_back[0, i] * e[t] * 10
                            + w_back[0, i] * e2[t] * 0
                        )

                # for t = 2:length(T);
                #     u = w*h(:,t-1) + x(t)*w_in;
                #     for i = 1:N;
                #         for j = 1:N;
                #             if learning_rule == 1;
                #                 dw(i,j,t) = z(i,t)*der(u(i))*h(j,t-1);
                #             elseif learning_rule == 2
                #                 dw(i,j,t) = h(j,t-1)*h(i,t-1)*(target(t-1)-y(t-1));
                #                 %dw(:,j,t) = h(j,t-1)*(target(t-1)-y(t-1));
                #                 %dw(i,:,t) = h(i,t-1)*(target(t-1)-y(t-1));
                #             end
                #         end
                #     end

                # --- Literal translation of the above (uncommented) ---
                dw = np.zeros((N, N, len(T)))
                for t in range(1, len(T)):  # 2:length(T)  (1-based) → range starts at 1
                    u = w @ h[:, t - 1] + x[t] * w_in
                    for i in range(N):
                        for j in range(N):
                            if learning_rule == 1:
                                dw[i, j, t] = z[i, t] * der(u[i]) * h[j, t - 1]
                            elif learning_rule == 2:
                                dw[i, j, t] = h[j, t - 1] * h[i, t - 1] * (target[t - 1] - y[t - 1])
                                # dw[:,j,t] = h(j,t-1)*(target(t-1)-y(t-1));
                                # dw(i,:,t) = h(i,t-1)*(target(t-1)-y(t-1));

                # tuno = nanmean(R(:,1:100,1)');
                # tun  = nanmean(R(:,1:100,end)');
                # dtun = tun - tuno;
                tuno = np.nanmean(R[:, 0:100, 0], axis=1)
                tun  = np.nanmean(R[:, 0:100, iter - 1], axis=1)  # end' → current iter
                dtun = tun - tuno

                # for i = 1:N;
                #     dww(:,i) = dtun(i)*mean(e2(end-20:end))*(iter>20);
                # end
                dww = np.zeros((N, N))
                tail_mean = np.mean(e2[-21:]) if len(e2) >= 21 else np.mean(e2)
                scale_iter = 1.0 if (iter > 20) else 0.0
                for i in range(N):
                    dww[:, i] = dtun[i] * tail_mean * scale_iter

                # for i = 1:N;
                #     for j = 1:N;
                #         dwww(i,j) = tun(j)*dtun(i)*mean(e2(end-20:end))*(iter>20); %from neuron j onto neuron i
                #     end
                # end
                dwww = np.zeros((N, N))
                for i in range(N):
                    for j in range(N):
                        dwww[i, j] = tun[j] * dtun[i] * tail_mean * scale_iter  # from j onto i

                # cc = corr(h',h(real_cn,:)');
                # for i = 1:N;
                #     dwwww(i,:) = cc(i)*mean(e2(end-20:end))*(iter>20);
                # end
                # --- Improvisation: vectorized corr with a reference row ---
                # cc: correlation between each row h[i,:] and h[real_cn-1,:]
                ref = h[real_cn - 1, :]
                # Avoid NaNs; compute Pearson r manually
                dwwww = np.zeros((N, h.shape[1]))
                # If you intended a single scalar per i (like MATLAB cc(i)), take correlation over entire time:
                cc_vec = np.zeros(N)
                ref0 = ref - ref.mean()
                denom_ref = np.sqrt((ref0 ** 2).sum())
                for i in range(N):
                    xi = h[i, :]
                    xi0 = xi - xi.mean()
                    denom_xi = np.sqrt((xi0 ** 2).sum())
                    cc = (xi0 @ ref0) / (denom_xi * denom_ref + 1e-12)
                    cc_vec[i] = cc
                dwwww = (cc_vec[:, None]) * tail_mean * scale_iter

                # dw = eta*tau/length(T)*sum(dw,3);
                dw = eta * tau / len(T) * np.sum(dw, axis=2)

                # dw = dw + dww*0.035 + dwww*0.025 + dwwww*0.025;
                # dw = dw.*mask;
                w = w + dw

            if iter == 1:
                # [a,b] = sort(mean(h'.^2));
                cn = 97
                real_cn = cn
                # [a,b] = sort(mean(h'),'descend');
                w_out = np.zeros((1, N))
                w_out[0, cn - 1] = 1  # w_out(cn) = 1;
                w_out = w_out + np.random.randn(1, N) / 10
                #         ind = 1:N;ind(cn) = [];
                #         w_out(ind) = randn(length(ind),1)/10;

            if iter == 20:
                # [a,b] = sort(mean(h(:,100:end)'.^2) + mean(h(:,1:100)'.^2));
                # real_cn = b(1);
                pass

            # W(:,:,iter) = w;
            W_hist[:, :, iter - 1] = w  # --- Improvisation: avoid overwriting W

            figure = plt.figure(1)
            plt.cla()
            plt.plot(y, label='y')
            plt.plot(target, label='target')
            if iter > 20:
                plt.plot(h[real_cn - 1, :], label='h(real_cn)')
            plt.legend(loc='best')
            plt.draw()
            plt.pause(0.001)
            #     ylim([-2 2])

            # R(:,:,iter) = h;
            R[:, :, iter - 1] = h
            
            
            # --- Visualization summary (inside ii loop, after fit_rep loop) ---
        plt.figure(100)
        marg = 0.4
        
        # --- Improvisation: simple KDsubplot replacement ---
        def KDsubplot(nrows, ncols, idx_pair, marg):
            """Mimics KDsubplot(nrows, ncols, [row col], marg)"""
            row, col = idx_pair
            plt.subplot(nrows, ncols, (row - 1) * ncols + col)
            plt.subplots_adjust(wspace=marg, hspace=marg)
        
        KDsubplot(2, 4, [learning_rule, 1], marg)
        plt.plot(np.squeeze(np.mean(R[cn - 1, :, [0, -1]], axis=1)))
        KDsubplot(2, 4, [learning_rule, 2], marg)
        plt.imshow(W_hist[:, :, -1] - W_hist[:, :, 0], vmin=-1, vmax=1)
        plt.colorbar()
        plt.colormaps['viridis']  # parula ≈ viridis
        KDsubplot(2, 4, [learning_rule, 3], marg)
        plt.imshow(R[:, :, -1] - R[:, :, 0])
        plt.colorbar()
        KDsubplot(2, 4, [learning_rule, 4], marg)
        del_var = np.sum(R[:, 99:, -1] - R[:, 99:, 10], axis=1)
        plt.scatter(np.abs(np.arange(1, N + 1) - np.mean(cn)), del_var, c='k', marker='o', facecolors='none')
        plt.scatter(np.zeros(len(np.atleast_1d(cn))), del_var[cn - 1], c='m', marker='o')
        
        cn = real_cn
        learning_rule = 2
        KDsubplot(2, 4, [learning_rule, 1], marg)
        plt.plot(np.squeeze(np.mean(R[cn - 1, :, [19, -1]], axis=1)))
        KDsubplot(2, 4, [learning_rule, 2], marg)
        plt.imshow(W_hist[:, :, -1] - W_hist[:, :, 19], vmin=-1, vmax=1)
        plt.colorbar()
        plt.colormaps['viridis']
        KDsubplot(2, 4, [learning_rule, 3], marg)
        plt.imshow(R[:, :, -1] - R[:, :, 19])
        plt.colorbar()
        KDsubplot(2, 4, [learning_rule, 4], marg)
        del_var = np.sum(R[:, 99:, -1] - R[:, 99:, 19], axis=1)
        plt.scatter(np.abs(np.arange(1, N + 1) - np.mean(cn)), del_var, c='k', marker='o', facecolors='none')
        plt.scatter(np.zeros(len(np.atleast_1d(cn))), del_var[cn - 1], c='m', marker='o')
        
        plt.show()
        
        # clear y
        y = np.array([])

        for ei in range(1, 3):  # for ei = 1:2
            # clear Y
            Y = np.array([])
            ww = W_hist[:, :, -1].copy()  # ww = W(:,:,end);
            if ei == 1:
                ww = W_hist[:, :, 0].copy()  # ww = W(:,:,1);

            Y = np.zeros((len(T), N, N))
            for ci in range(N):  # for ci = 1:N;
                h = np.zeros((N, len(T)))
                pert = w_in * 0
                pert[ci] = 0.01
                for i in range(len(T) - 1):
                    h[:, i + 1] = h[:, i] + dt / tau * (
                        -h[:, i] + fun(ww @ h[:, i] + x[i + 1] * pert)
                    )
                Y[:, :, ci] = h.T
            y_tmp = np.squeeze(np.mean(Y, axis=0)).T  # squeeze(mean(Y))
            if ei == 1:
                y = y_tmp[:, :, np.newaxis]
            else:
                y = np.concatenate((y, y_tmp[:, :, np.newaxis]), axis=2)

        dy = y[:, :, 1] - y[:, :, 0]  # dy = y(:,:,2) - y(:,:,1);

        plt.clf()
        # del = sum(mean(R(:,:,end-10:end),3)-R(:,:,1),2);
        del_var = np.sum(np.mean(R[:, :, -10:], axis=2) - R[:, :, 0], axis=1)
        tun = np.sum(R[:, :, 0], axis=1)  # tun = sum(R(:,:,1),2);
        non = np.ones_like(del_var)

        dw = W_hist[:, :, -1] - W_hist[:, :, 0]  # dw = W(:,:,end) - W(:,:,1);

        # --- Improvisation: define mean_bin_plot equivalent ---
        def mean_bin_plot(x, y, bins=20):
            """Mimic mean_bin_plot(x,y): plot mean of y in bins of x."""
            x = np.asarray(x).flatten()
            y = np.asarray(y).flatten()
            inds = np.digitize(x, np.linspace(np.min(x), np.max(x), bins))
            xb, yb = [], []
            for b in range(1, bins + 1):
                mask = inds == b
                if np.any(mask):
                    xb.append(np.mean(x[mask]))
                    yb.append(np.mean(y[mask]))
            plt.plot(xb, yb, 'ko-')

        mean_bin_plot(del_var, np.mean(dw, axis=0))
        plt.xlabel(r'$\Delta$ activity')
        plt.ylabel(r'$\Delta$ outputs')
        plt.show()


        plt.clf()
        
        dyt = dy.T  # dyt = dy';
        
        a = np.ones((len(del_var), 1)) @ del_var.reshape(1, -1)
        
        plt.subplot(1, 2, 1)
        x_flat = a.ravel()
        y_flat = dyt.ravel()
        m = min(x_flat.size, y_flat.size)
        mean_bin_plot(x_flat[:m], y_flat[:m])
        
        plt.subplot(1, 2, 2)
        a = np.ones((len(del_var), 1)) @ del_var.reshape(1, -1)
        a = del_var.reshape(-1, 1) @ tun.reshape(1, -1)
        x_flat = a.ravel()
        y_flat = dyt.ravel()
        m = min(x_flat.size, y_flat.size)
        mean_bin_plot(x_flat[:m], y_flat[:m])
        
        for jj in range(1, 3):
            if jj == 2:
                dy_local = y[:, :, 1] - y[:, :, 0]
            else:
                dy_local = y[:, :, 0]
        
            dyt = dy_local.T
        
            plt.figure(jj + ii)
            # figure_initialize (no direct equivalent)
        
            tstrt = 100
            dtrl = np.sum(np.mean(R[:, tstrt-1:, -10:], axis=2) - R[:, tstrt-1:, 0], axis=1)
            dpre = np.sum(np.mean(R[:, 0:tstrt, -10:], axis=2) - R[:, 0:tstrt, 0], axis=1)
            trl = np.sum(R[:, tstrt-1:, 0], axis=1)
            pre = np.sum(R[:, 0:tstrt, 0], axis=1)
            non = np.ones_like(del_var)
        
            if jj == 2:
                vec = [non, pre, dpre, trl, dtrl]
            else:
                vec = [non, pre, trl]
        
            L = len(vec)
            I = 0
            OP = []
            X_cols = []
            for i in range(L):
                for j in range(L):
                    I += 1
                    op = np.outer(vec[i], vec[j])
                    a_flat = op.ravel()
                    a_flat = (a_flat - np.nanmean(a_flat)) / (np.nanstd(a_flat) + 1e-12)
                    X_cols.append(a_flat)
                    OP.append(op)
        
            X = np.column_stack(X_cols)
            Y = dyt.ravel()
            X[:, 0] = 0
        
            beta = np.linalg.pinv(X) @ Y
            beta_matrix = np.zeros((L, L))
            I = 0
            for i in range(L):
                for j in range(L):
                    I += 1
                    beta_matrix[i, j] = beta[I - 1]
        
            # --- MATLAB-style lasso replacement ---
            from sklearn.linear_model import LassoCV
            from sklearn.preprocessing import StandardScaler
        
            scaler_X = StandardScaler(with_mean=True, with_std=True)
            scaler_Y = StandardScaler(with_mean=True, with_std=False)
        
            X_scaled = scaler_X.fit_transform(X)
            Y_scaled = scaler_Y.fit_transform(Y.reshape(-1, 1)).ravel()
        
            n_samples = X_scaled.shape[0]
            lambda_vals = np.logspace(-8, -4, 100)
            alpha_vals = lambda_vals / n_samples
        
            lcv = LassoCV(
                alphas=alpha_vals,
                cv=5,
                fit_intercept=True,
                max_iter=10000,
                tol=1e-4,
                n_jobs=-1,
            )
            lcv.fit(X_scaled, Y_scaled)
        
            # Recover unscaled coefficients
            B2_best = lcv.coef_ / scaler_X.scale_
        
            lasso_matrix = np.zeros((L, L))
            I = 0
            for i in range(L):
                for j in range(L):
                    I += 1
                    lasso_matrix[i, j] = B2_best[I - 1]
        
            a_lim = np.max(np.abs(lasso_matrix.ravel()) * 1.2) * 1e2 * 2
        
            plt.clf()
            marg = 0.5
            fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        
            im0 = axes[0].imshow((lasso_matrix[0, 1:] * 1e2)[np.newaxis, :],
                                 vmin=-a_lim, vmax=a_lim, aspect='auto', cmap='bwr')
            axes[0].set_xticks([]); axes[0].set_yticks([])
        
            im1 = axes[1].imshow((lasso_matrix[1:, 0][:, np.newaxis] * 1e2),
                                 vmin=-a_lim, vmax=a_lim, aspect='auto', cmap='bwr')
            axes[1].set_xticks([]); axes[1].set_yticks([])
        
            im2 = axes[2].imshow(lasso_matrix[1:, 1:] * 1e2,
                                 vmin=-a_lim, vmax=a_lim, aspect='auto', cmap='bwr')
            axes[2].set_xticks([]); axes[2].set_yticks([])
        
            fig.colorbar(im2, ax=axes.ravel().tolist(), shrink=0.6)
        
            if jj == 2:
                if 'L_mat_store' not in globals():
                    L_mat_store = {}
                L_mat_store[(fit_rep, ii)] = lasso_matrix.copy()
        
            plt.tight_layout()
            plt.show()
            
            import numpy as np
            import matplotlib.pyplot as plt
            
            
    # === Build 4D array from L_mat_store ===
    if 'L_mat_store' in globals() and len(L_mat_store) > 0:
        unique_fit_reps = sorted({k[0] for k in L_mat_store.keys()})
        unique_iis = sorted({k[1] for k in L_mat_store.keys()})
    
        L = next(iter(L_mat_store.values())).shape[0]
        n_fit_rep = len(unique_fit_reps)
        n_ii = len(unique_iis)
    
        L_mat_store_array = np.zeros((L, L, n_fit_rep, n_ii))
        for f_idx, fit_rep in enumerate(unique_fit_reps):
            for i_idx, ii_val in enumerate(unique_iis):
                L_mat_store_array[:, :, f_idx, i_idx] = L_mat_store[(fit_rep, ii_val)]
    else:
        raise ValueError("No L_mat_store found — check loop execution.")
    
            
    plt.figure(ii * 10)
    # figure_initialize (no direct equivalent)
    # MATLAB 'set(gcf, "position", [33 3 4 4])' sets figure size in inches;
    # in matplotlib we can approximate:
    plt.gcf().set_size_inches(4, 4)
    
    # lasso_matrix = mean(L_mat(:,:,:,ii),3);
    # In Python: average across axis=2 (third dimension)
    lasso_matrix = np.mean(L_mat_store_array[:, :, :, ii - 1], axis=2)
    
    marg = 0.5
    a = np.max(np.abs(lasso_matrix)) * 1e2  # inferred from previous context
    
    # --- Improvisation: KDsubplot(4,1.5,[1.5 1.5],marg)
    # KDsubplot likely creates custom subplot positions with overlap control.
    # We'll emulate this with three horizontally arranged axes.
    
    fig, axes = plt.subplots(1, 3, figsize=(8, 3))
    
    im0 = axes[0].imshow((lasso_matrix[0, 1:] * 1e2)[np.newaxis, :],
                         vmin=-a, vmax=a, aspect='auto', cmap='bwr')
    axes[0].set_xticks([]); axes[0].set_yticks([])
    
    im1 = axes[1].imshow((lasso_matrix[1:, 0][:, np.newaxis] * 1e2),
                         vmin=-a, vmax=a, aspect='auto', cmap='bwr')
    axes[1].set_xticks([]); axes[1].set_yticks([])
    
    im2 = axes[2].imshow(lasso_matrix[1:, 1:] * 1e2,
                         vmin=-a, vmax=a, aspect='auto', cmap='bwr')
    axes[2].set_xticks([]); axes[2].set_yticks([])
    
    fig.colorbar(im2, ax=axes.ravel().tolist(), shrink=0.6)
    plt.tight_layout()
    
    # colormap(bluewhitered) -> 'bwr' above
    # % title(labels{ii});
    plt.suptitle(labels[ii - 1])
    
    plt.show()

