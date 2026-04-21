% =========================================================================
%  PART 3A — CRACK FILLING: Order Statistics Filters
%             Modified Trimmed Mean (MTM) — Paper Section IV-A
% =========================================================================

fprintf('\n--- PART 3A: Crack Filling — Order Statistics (MTM) Filter ---\n');

%% ===============================
% Convert to double for filtering
% ===============================
gray_d     = double(gray);
crack_mask = crack_clean;
non_crack  = ~crack_mask;

if size(img,3) == 3
    img_d    = double(img_rgb);
    channels = 3;
else
    img_d    = double(gray);
    img_d    = repmat(img_d,[1 1 3]);
    channels = 1;
end

%% ===============================
% MTM Filter — window 1 px wider than widest crack (paper §IV-A)
% ===============================
win  = 5;    % filter window size (must be odd)
half = floor(win/2);

filled_mtm = img_d;

for ch = 1:channels
    channel_in  = img_d(:,:,ch);
    channel_out = channel_in;

    for r = 1:nRows
        for c = 1:nCols
            if crack_mask(r,c)
                r1 = max(1, r-half);  r2 = min(nRows, r+half);
                c1 = max(1, c-half);  c2 = min(nCols, c+half);
                win_vals  = channel_in(r1:r2, c1:c2);
                win_mask  = non_crack(r1:r2, c1:c2);
                valid_pix = win_vals(win_mask);
                if ~isempty(valid_pix)
                    channel_out(r,c) = mean(valid_pix);
                else
                    channel_out(r,c) = mean(win_vals(:));
                end
            end
        end
    end
    filled_mtm(:,:,ch) = channel_out;
end

filled_mtm = uint8(filled_mtm);
if channels == 1
    filled_mtm = filled_mtm(:,:,1);
end

%% ===============================
% Part 3A — Visualisation
% ===============================
if channels == 3
    orig_show = img_rgb;
else
    orig_show = gray;
end

figure('Name','PART 3A — Crack Filling: MTM Order Statistics Filter', ...
       'NumberTitle','off','Units','normalized','OuterPosition',[0 0 1 1]);

subplot(2,3,1);
imshow(gray);
title('1. Original Grayscale');

subplot(2,3,2);
imshow(crack_clean);
title('2. Final Crack Mask');

subplot(2,3,3);
overlay = img_rgb;
for ch = 1:3
    tmp = overlay(:,:,ch);
    if ch == 1,  tmp(crack_clean) = 255;
    else,        tmp(crack_clean) = 0;
    end
    overlay(:,:,ch) = tmp;
end
imshow(overlay);
title('3. Cracks Highlighted (red)');

subplot(2,3,4);
imshow(filled_mtm);
title(['4. Filled Image (MTM, win=',num2str(win),')']);

subplot(2,3,5);
diff_map = abs(double(orig_show) - double(filled_mtm));
diff_map = uint8(diff_map * 5);
imshow(diff_map);
title('5. Difference Map (×5)');

subplot(2,3,6);
imshowpair(orig_show, filled_mtm, 'montage');
title('6. Original | MTM Filled');

sgtitle('PART 3A — Crack Filling via Modified Trimmed Mean (Order Statistics) Filter', ...
        'FontSize',13,'FontWeight','bold');


% =========================================================================
%  PART 3B — CRACK FILLING: Orientation-Sensitive Controlled Anisotropic
%             Diffusion (Perona-Malik) — Paper Section IV-B
% =========================================================================

fprintf('\n--- PART 3B: Crack Filling — Controlled Anisotropic Diffusion ---\n');

%% ===============================
% Hough Transform — find dominant crack orientations
% ===============================
[H_hough, theta, rho] = hough(crack_clean);
P = houghpeaks(H_hough, 10, 'Threshold', ceil(0.3 * max(H_hough(:))));
lines = houghlines(crack_clean, theta, rho, P, ...
                   'FillGap', 10, 'MinLength', 7);

% Per-pixel orientation map (angle in degrees, 0 = horizontal)
orient_map = zeros(nRows, nCols);
for k = 1:length(lines)
    pt1 = lines(k).point1;
    pt2 = lines(k).point2;
    ang = lines(k).theta;
    x_range = min(pt1(1),pt2(1)):max(pt1(1),pt2(1));
    for xi = x_range
        t = (xi - pt1(1)) / max(pt2(1)-pt1(1), 1);
        yi = round(pt1(2) + t*(pt2(2)-pt1(2)));
        if yi >= 1 && yi <= nRows && xi >= 1 && xi <= nCols
            orient_map(yi, xi) = ang;
        end
    end
end

%% ===============================
% Anisotropic Diffusion Parameters (Perona-Malik)
% ===============================
lambda    = 0.25;   % stability: must be <= 0.25
K         = 30;     % conduction coefficient threshold (paper eq.20)
num_iter  = 20;     % number of diffusion iterations

g_func = @(grad_mag) 1 ./ (1 + (grad_mag / K).^2);

%% ===============================
% Apply diffusion on each channel
% ===============================
if channels == 3
    img_float = double(img_rgb);
else
    img_float = repmat(double(gray),[1 1 3]);
end

filled_ad = img_float;

for ch = 1:3
    I = img_float(:,:,ch);
    I_out = I;

    for iter = 1:num_iter
        DN = [I_out(1,:);   I_out(1:end-1,:)] - I_out;
        DS = [I_out(2:end,:); I_out(end,:)]   - I_out;
        DE = [I_out(:,2:end), I_out(:,end)]   - I_out;
        DW = [I_out(:,1), I_out(:,1:end-1)]   - I_out;

        cN = g_func(abs(DN));
        cS = g_func(abs(DS));
        cE = g_func(abs(DE));
        cW = g_func(abs(DW));

        is_horiz = abs(orient_map) < 45;
        is_vert  = ~is_horiz;

        cE(crack_clean & is_horiz) = 0;
        cW(crack_clean & is_horiz) = 0;
        cN(crack_clean & is_vert)  = 0;
        cS(crack_clean & is_vert)  = 0;

        delta = lambda * (cN.*DN + cS.*DS + cE.*DE + cW.*DW);
        update_mask = crack_clean;
        I_out(update_mask) = I_out(update_mask) + delta(update_mask);
    end

    filled_ad(:,:,ch) = I_out;
end

filled_ad = uint8(filled_ad);
if channels == 1
    filled_ad_show = filled_ad(:,:,1);
else
    filled_ad_show = filled_ad;
end

%% ===============================
% Part 3B — Visualisation
% ===============================
figure('Name','PART 3B — Crack Filling: Anisotropic Diffusion', ...
       'NumberTitle','off','Units','normalized','OuterPosition',[0 0 1 1]);

subplot(2,3,1);
imshow(orig_show);
title('1. Original Image');

subplot(2,3,2);
imshow(crack_clean);
title('2. Final Crack Mask');

subplot(2,3,3);
hough_overlay = repmat(gray, [1 1 3]);
for k = 1:length(lines)
    hough_overlay = insertShape(hough_overlay, 'Line', ...
        [lines(k).point1, lines(k).point2], ...
        'Color','red','LineWidth',1);
end
imshow(hough_overlay);
title(['3. Hough Lines (',num2str(length(lines)),' detected)']);

subplot(2,3,4);
imshow(filled_ad_show);
title(['4. Filled Image (Anisotropic Diffusion, ',num2str(num_iter),' iters)']);

subplot(2,3,5);
diff_ad = abs(double(orig_show) - double(filled_ad_show));
diff_ad = uint8(diff_ad * 5);
imshow(diff_ad);
title('5. Difference Map (×5)');

subplot(2,3,6);
imshowpair(orig_show, filled_ad_show, 'montage');
title('6. Original | Anisotropic Diffusion Filled');

sgtitle('PART 3B — Crack Filling via Orientation-Sensitive Controlled Anisotropic Diffusion', ...
        'FontSize',13,'FontWeight','bold');


% =========================================================================
%  FINAL COMPARISON — All Methods Side by Side
% =========================================================================
figure('Name','FINAL — All Methods Comparison','NumberTitle','off', ...
       'Units','normalized','OuterPosition',[0 0 1 1]);

subplot(1,4,1);
imshow(orig_show);
title('Original','FontSize',12,'FontWeight','bold');

subplot(1,4,2);
imshow(crack_clean);
title('Crack Mask (Part 2)','FontSize',12,'FontWeight','bold');

subplot(1,4,3);
imshow(filled_mtm);
title({'MTM Order Statistics','(Part 3A)'},'FontSize',12,'FontWeight','bold');

subplot(1,4,4);
imshow(filled_ad_show);
title({'Anisotropic Diffusion','(Part 3B)'},'FontSize',12,'FontWeight','bold');

sgtitle('Final Comparison: Original vs Crack Mask vs MTM vs Anisotropic Diffusion', ...
        'FontSize',14,'FontWeight','bold');

fprintf('\nDone! All parts executed successfully.\n');
fprintf('Summary:\n');
fprintf('  Part 1  — Crack detection threshold T = %d\n', T);
fprintf('  Part 2A — Grassfire from %d seed(s)\n', numel(seed_rows));
fprintf('  Part 2B — MRBF classifier (H & S features)\n');
fprintf('  Part 3A — MTM filter window = %dx%d\n', win, win);
fprintf('  Part 3B — Anisotropic diffusion: K=%d, lambda=%.2f, iters=%d\n', ...
        K, lambda, num_iter);


%% =========================================================================
% EXTRA COMPARISON — Part 3 Applied DIRECTLY on Raw Crack Map (Without Part 2)
% Shows necessity of brush-stroke removal in Part 2
% =========================================================================

raw_mask = crack_map;
filled_raw = img_d;

for ch = 1:channels
    channel_in  = img_d(:,:,ch);
    channel_out = channel_in;

    for r = 1:nRows
        for c = 1:nCols
            if raw_mask(r,c)
                r1 = max(1,r-half);  r2 = min(nRows,r+half);
                c1 = max(1,c-half);  c2 = min(nCols,c+half);
                vals = channel_in(r1:r2, c1:c2);
                msk  = ~raw_mask(r1:r2, c1:c2);
                valid_pix = vals(msk);
                if ~isempty(valid_pix)
                    channel_out(r,c) = mean(valid_pix);
                else
                    channel_out(r,c) = mean(vals(:));
                end
            end
        end
    end
    filled_raw(:,:,ch) = channel_out;
end

filled_raw = uint8(filled_raw);

figure('Name','Necessity of Part 2','NumberTitle','off', ...
       'Units','normalized','OuterPosition',[0 0 1 1]);

subplot(1,3,1);
imshow(orig_show);
title('Original');

subplot(1,3,2);
imshow(filled_raw);
title('Part 3 on Raw Crack Map (no Part 2)');

subplot(1,3,3);
imshow(filled_mtm);
title('Part 3 after Part 2 (brush strokes removed)');

sgtitle('Necessity of Part 2: Brush Stroke Removal Before Filling', ...
        'FontSize',14,'FontWeight','bold');


% =========================================================================
%  LOCAL FUNCTION — Manual MRBF Classifier
%  Implements equations (4)–(9) from Giakoumis et al. 2006
%  Used as fallback when fitcnet (Statistics Toolbox) is unavailable
%
%  Architecture:
%    Input   : 2-D  [Hue (degrees),  Saturation]
%    Hidden  : L_per_class Gaussian RBF kernels per class
%              Centre  : marginal median LVQ          — eq.(6)
%              Width   : MAD / 0.6745                 — eq.(7)
%    Output  : 2 nodes  (crack=1 / brush stroke=2)
%    Weights : backprop on MSE                        — eq.(8)
% =========================================================================
function y_pred = mrbf_manual(X_train, y_train, X_query)

    % ---- Hyper-parameters ----
    L_per_class = 3;          % paper: 3 hidden units per class
    classes     = [1, 2];
    n_classes   = numel(classes);
    L           = L_per_class * n_classes;
    eta         = 0.01;       % output-layer learning rate — eq.(8)
    W_buf       = 500;        % moving-window size for median — paper notation W

    [N_train, D] = size(X_train);

    % ---- Step 1: Initialise kernel centres from random class samples ----
    rng(0);
    mu  = zeros(L, D);
    sig = ones(L, D);

    k = 1;
    for c_idx = 1:n_classes
        idx_c = find(y_train == classes(c_idx));
        picks = idx_c(randperm(numel(idx_c), min(L_per_class, numel(idx_c))));
        for p = 1:numel(picks)
            mu(k,:) = X_train(picks(p),:);
            k = k + 1;
        end
    end

    kernel_class = repelem(classes, L_per_class)';   % [L x 1]

    % ---- Step 2: Unsupervised hidden-layer training (marginal median LVQ) ----
    buf_count = zeros(L,1,'int32');
    buf       = cell(L,1);
    for kk = 1:L,  buf{kk} = zeros(W_buf, D);  end

    for epoch = 1:50
        perm = randperm(N_train);
        for ii = 1:N_train
            x  = X_train(perm(ii),:);
            cx = y_train(perm(ii));

            % Winner kernel of same class — eq.(5)
            same_class = find(kernel_class == cx);
            dists = sum((repmat(x, numel(same_class), 1) - mu(same_class,:)).^2, 2);
            [~, local_w] = min(dists);
            winner = same_class(local_w);

            % Add to circular buffer
            pos = mod(buf_count(winner), W_buf) + 1;
            buf{winner}(pos,:) = x;
            buf_count(winner)  = buf_count(winner) + 1;

            n_buf = min(buf_count(winner), W_buf);
            data  = buf{winner}(1:n_buf,:);

            % Update centre via marginal median — eq.(6)
            mu(winner,:) = median(data, 1);

            % Update width via MAD / 0.6745 — eq.(7)
            mad_val = median(abs(data - repmat(mu(winner,:), n_buf, 1)), 1);
            sig(winner,:) = max(mad_val / 0.6745, 1e-6);
        end
    end

    % ---- Step 3: RBF activations — eq.(4) ----
    % phi_j(x) = exp(-sum( (x - mu_j)^2 / sig_j^2 ))
    function phi = rbf_act(X, mu_, sig_)
        N_ = size(X,1);
        L_ = size(mu_,1);
        phi = zeros(N_, L_);
        for kk_ = 1:L_
            d = X - repmat(mu_(kk_,:), N_, 1);
            phi(:,kk_) = exp(-sum((d ./ repmat(sig_(kk_,:), N_, 1)).^2, 2));
        end
    end

    Phi_train = rbf_act(X_train, mu, sig);    % [N_train x L]

    % ---- Step 4: Supervised output-weight training — eq.(8) ----
    W_out = randn(L, n_classes) * 0.1;

    % Desired output F_k(X) — eq.(9): 1 if X in class k, else 0
    F_train = zeros(N_train, n_classes);
    for c_idx = 1:n_classes
        F_train(:,c_idx) = double(y_train == classes(c_idx));
    end

    for epoch = 1:100
        perm = randperm(N_train);
        for ii = 1:N_train
            phi_i = Phi_train(perm(ii),:)';          % [L x 1]
            raw   = W_out' * phi_i;                  % [n_classes x 1]
            % Softmax output
            e_raw = exp(raw - max(raw));
            Y_i   = e_raw / sum(e_raw);
            F_i   = F_train(perm(ii),:)';            % [n_classes x 1]
            % Gradient — eq.(8)
            grad  = phi_i * (Y_i - F_i)';            % [L x n_classes]
            W_out = W_out - eta * grad;
        end
    end

    % ---- Step 5: Classify query pixels (winner-takes-all) ----
    Phi_query = rbf_act(X_query, mu, sig);    % [N_q x L]
    scores    = Phi_query * W_out;            % [N_q x n_classes]
    [~, pred_idx] = max(scores, [], 2);
    y_pred = classes(pred_idx)';

end   % mrbf_manual