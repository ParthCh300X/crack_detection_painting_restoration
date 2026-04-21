clc;
clear;
close all;

%% ===============================
% Load Image
% ===============================
img = imread('input_img.png');
if size(img,3) == 3
    gray = rgb2gray(img);
else
    gray = img;
end
gray = im2uint8(gray);

% Keep original RGB for hue/saturation use in Part 2
if size(img,3) == 3
    img_rgb = img;
else
    img_rgb = cat(3, gray, gray, gray);
end

% =========================================================================
%  PART 1 — CRACK DETECTION (Top-Hat Transform)
% =========================================================================

%% ===============================
% Structuring Element (Paper: square 3x3, 2 dilations)
% ===============================
B  = strel('square', 3);
SE = B;
for k = 1:2
    SE = imdilate(SE.Neighborhood, B.Neighborhood);
    SE = strel(SE);
end

%% ===============================
% Morphological Operations
% ===============================
dilated      = imdilate(gray, SE);
eroded       = imerode(gray, SE);
opened       = imopen(gray, SE);
closed       = imclose(gray, SE);
white_tophat = imsubtract(gray, opened);
black_tophat = imsubtract(closed, gray);

%% ===============================
% Crack Extraction via Thresholding
% ===============================
T         = 18;           % threshold — adjust between 15–30
crack_map = black_tophat > T;
crack_map = bwareaopen(crack_map, 5);   % remove tiny specks
BW_Crack_img = ~crack_map;

%% ===============================
% Part 1 — Visualisation
% ===============================
figure('Name','PART 1 — Crack Detection','NumberTitle','off', ...
       'Units','normalized','OuterPosition',[0 0 1 1]);

subplot(3,4,1);  imshow(gray);                         title('1. Original Grayscale');
subplot(3,4,2);  imshow(SE.Neighborhood);              title('2. Final Structuring Element');
subplot(3,4,3);  imshow(dilated);                      title('3. Dilation');
subplot(3,4,4);  imshow(eroded);                       title('4. Erosion');
subplot(3,4,5);  imshow(opened);                       title('5. Opening');
subplot(3,4,6);  imshow(closed);                       title('6. Closing');
subplot(3,4,7);  imshow(white_tophat, []);             title('7. White Top-Hat');
subplot(3,4,8);  imshow(black_tophat, []);             title('8. Black Top-Hat (BTH)');
subplot(3,4,9);  imshow(crack_map);                    title(['9. Binary Crack Map (T=',num2str(T),')']);
subplot(3,4,10); imshow(BW_Crack_img);                 title('10. BW Crack Image');
subplot(3,4,11); imshowpair(white_tophat, black_tophat,'montage'); title('11. WTH vs BTH');
subplot(3,4,12); imshowpair(gray, BW_Crack_img,'montage');         title('12. Original vs Crack Image');

sgtitle('PART 1 — Crack Detection via Morphological Top-Hat Transform', ...
        'FontSize',14,'FontWeight','bold');


% =========================================================================
%  PART 2A — BRUSH STROKE REMOVAL (Semi-Automatic Crack Separation)
%             Region-Growing (Grassfire) from user-selected crack seeds
% =========================================================================

fprintf('\n--- PART 2A: Semi-Automatic Crack Separation (Grassfire) ---\n');
fprintf('Please select seed points ON ACTUAL CRACKS in the figure.\n');
fprintf('Press ENTER when done selecting.\n\n');

%% ===============================
% Display crack map for seed selection
% ===============================
figure('Name','PART 2A — Select Crack Seeds','NumberTitle','off');
imshow(crack_map);
title({'Select seed points ON actual cracks (not brush strokes)'; ...
       'Left-click to add points — press ENTER when done'}, ...
      'FontSize',11);

% Collect seed points interactively
[seed_cols, seed_rows] = getpts();   % returns (x,y) = (col,row)
seed_rows = round(seed_rows);
seed_cols = round(seed_cols);

close(gcf);

%% ===============================
% Region-Growing (Grassfire Algorithm) — 8-connectivity
% ===============================
[nRows, nCols] = size(crack_map);

if isempty(seed_rows)
    warning('No seeds selected. Skipping brush-stroke separation.');
    crack_clean_2a = crack_map;
else
    valid = seed_rows >= 1 & seed_rows <= nRows & ...
            seed_cols >= 1 & seed_cols <= nCols;
    seed_rows = seed_rows(valid);
    seed_cols = seed_cols(valid);

    % Snap seeds to nearest crack pixel if needed
    for s = 1:numel(seed_rows)
        if ~crack_map(seed_rows(s), seed_cols(s))
            rMin = max(1, seed_rows(s)-2);  rMax = min(nRows, seed_rows(s)+2);
            cMin = max(1, seed_cols(s)-2);  cMax = min(nCols, seed_cols(s)+2);
            patch = crack_map(rMin:rMax, cMin:cMax);
            [pr, pc] = find(patch, 1);
            if ~isempty(pr)
                seed_rows(s) = rMin + pr - 1;
                seed_cols(s) = cMin + pc - 1;
            end
        end
    end

    % BFS / Grassfire
    visited   = false(nRows, nCols);
    queue_r   = zeros(nnz(crack_map), 1, 'int32');
    queue_c   = zeros(nnz(crack_map), 1, 'int32');
    head = 1;  tail = 0;

    for s = 1:numel(seed_rows)
        r = seed_rows(s);  c = seed_cols(s);
        if crack_map(r,c) && ~visited(r,c)
            visited(r,c) = true;
            tail = tail + 1;
            queue_r(tail) = r;
            queue_c(tail) = c;
        end
    end

    dr = [-1,-1,-1, 0, 0, 1, 1, 1];
    dc = [-1, 0, 1,-1, 1,-1, 0, 1];

    while head <= tail
        r = queue_r(head);
        c = queue_c(head);
        head = head + 1;
        for d = 1:8
            nr = r + dr(d);
            nc = c + dc(d);
            if nr >= 1 && nr <= nRows && nc >= 1 && nc <= nCols
                if crack_map(nr,nc) && ~visited(nr,nc)
                    visited(nr,nc) = true;
                    tail = tail + 1;
                    queue_r(tail) = nr;
                    queue_c(tail) = nc;
                end
            end
        end
    end

    crack_clean_2a = visited;
end

brush_strokes_removed_2a = crack_map & ~crack_clean_2a;

%% ===============================
% Part 2A — Visualisation
% ===============================
figure('Name','PART 2A — Semi-Auto Grassfire Brush Stroke Removal','NumberTitle','off', ...
       'Units','normalized','OuterPosition',[0 0 1 1]);

subplot(2,3,1);
imshow(gray);
title('1. Original Grayscale');

subplot(2,3,2);
imshow(crack_map);
title('2. Raw Crack Map (from Part 1)');

subplot(2,3,3);
seed_img = repmat(uint8(crack_map)*255, [1 1 3]);
for s = 1:numel(seed_rows)
    r = seed_rows(s);  c = seed_cols(s);
    rr = max(1,r-3):min(nRows,r+3);
    cc = max(1,c-3):min(nCols,c+3);
    seed_img(rr, cc, 1) = 255;
    seed_img(rr, cc, 2) = 0;
    seed_img(rr, cc, 3) = 0;
end
imshow(seed_img);
title('3. Crack Map with Seeds (red)');

subplot(2,3,4);
imshow(brush_strokes_removed_2a);
title('4. Removed Brush Strokes (2A)');

subplot(2,3,5);
imshow(crack_clean_2a);
title('5. Cleaned Crack Map (2A)');

subplot(2,3,6);
comp = [crack_map, ones(nRows,2,'logical'), crack_clean_2a];
imshow(comp);
title('6. Before | After (2A)');

sgtitle('PART 2A — Semi-Automatic Brush Stroke Removal (Region-Growing / Grassfire)', ...
        'FontSize',14,'FontWeight','bold');

fprintf('Part 2A — Brush strokes removed : %d pixels\n', nnz(brush_strokes_removed_2a));
fprintf('Part 2A — Crack pixels remaining: %d pixels\n', nnz(crack_clean_2a));


% =========================================================================
%  PART 2B — BRUSH STROKE REMOVAL (MRBF Neural Network)
%             Discrimination on the Basis of Hue and Saturation
%             Paper Section III-B — Giakoumis et al., IEEE TIP 2006
%
%  Statistical basis from paper (47 paintings):
%    Crack hue        : 0°–60°     Crack saturation     : 0.30–0.70
%    Brush-stroke hue : 0°–360°   Brush-stroke sat.    : 0.00–0.40
% =========================================================================

fprintf('\n--- PART 2B: MRBF Neural Network Brush-Stroke Separation ---\n');

%% ===============================
% Extract H and S for all crack-map pixels
% ===============================
hsv_img = rgb2hsv(img_rgb);          % H in [0,1], S in [0,1], V in [0,1]
H_full  = hsv_img(:,:,1);
S_full  = hsv_img(:,:,2);

crack_idx   = find(crack_map);
H_crack_px  = H_full(crack_idx) * 360;   % degrees [0, 360]
S_crack_px  = S_full(crack_idx);          % [0, 1]

if isempty(crack_idx)
    warning('No crack pixels — skipping Part 2B, using Part 2A result.');
    crack_clean_2b = crack_clean_2a;
else

    %% ===============================
    % Build synthetic training data from paper's reported distributions
    % Class 1 = Crack,  Class 2 = Brush Stroke
    % ===============================
    rng(42);
    n_per_class = 600;

    % Class 1: Cracks
    H_c = rand(n_per_class,1) * 60;
    S_c = 0.30 + rand(n_per_class,1) * 0.40;
    X_crack_tr = [H_c, S_c];
    y_crack_tr = ones(n_per_class, 1);

    % Class 2: Brush Strokes (two sub-groups — overlap + full gamut)
    n_overlap  = round(n_per_class * 0.40);
    n_fullgam  = n_per_class - n_overlap;
    H_ba = rand(n_overlap,1) * 60;
    S_ba = rand(n_overlap,1) * 0.40;
    H_bb = 60 + rand(n_fullgam,1) * 300;
    S_bb = rand(n_fullgam,1) * 0.40;
    X_brush_tr = [[H_ba; H_bb], [S_ba; S_bb]];
    y_brush_tr = 2 * ones(n_per_class, 1);

    X_train = [X_crack_tr; X_brush_tr];
    y_train = [y_crack_tr; y_brush_tr];

    %% ===============================
    % Feature matrix for crack pixels
    % ===============================
    X_query = [H_crack_px, S_crack_px];

    %% ===============================
    % Train classifier — fitcnet (R2021b+) or manual MRBF fallback
    % ===============================
    use_fitcnet = (exist('fitcnet','file') == 2);

    if use_fitcnet
        fprintf('[MRBF] Using fitcnet (Statistics & ML Toolbox)...\n');
        net_mrbf = fitcnet(X_train, y_train, ...
            'LayerSizes',    [6, 4], ...
            'Activations',   'relu', ...
            'Standardize',   true,   ...
            'Lambda',        1e-4,   ...
            'IterationLimit',500);
        y_pred_2b = predict(net_mrbf, X_query);
    else
        fprintf('[MRBF] fitcnet not found — using manual MRBF (eqs 4-9)...\n');
        y_pred_2b = mrbf_manual(X_train, y_train, X_query);
    end

    %% ===============================
    % Build cleaned crack map — class 1 = crack, class 2 = brush stroke
    % ===============================
    is_crack_2b = (y_pred_2b == 1);
    crack_clean_2b = false(nRows, nCols);
    crack_clean_2b(crack_idx(is_crack_2b)) = true;
end

brush_strokes_removed_2b = crack_map & ~crack_clean_2b;

%% ===============================
% Part 2B — Visualisation
% ===============================
figure('Name','PART 2B — MRBF Neural Network Brush Stroke Removal','NumberTitle','off', ...
       'Units','normalized','OuterPosition',[0 0 1 1]);

subplot(2,3,1);
imshow(img_rgb);
title('1. Original RGB Image');

subplot(2,3,2);
imshow(crack_map);
title('2. Raw Crack Map (from Part 1)');

subplot(2,3,3);
% H-S scatter coloured by MRBF prediction
scatter(H_crack_px(y_pred_2b==1), S_crack_px(y_pred_2b==1), 4, [0 0.6 0], 'filled');
hold on;
scatter(H_crack_px(y_pred_2b==2), S_crack_px(y_pred_2b==2), 4, [0.8 0 0], 'filled');
xlabel('Hue (degrees)');  ylabel('Saturation');
legend('Crack','Brush Stroke','Location','NE','FontSize',8);
title('3. H–S Classification (green=crack, red=brush)');
xlim([0 360]);  ylim([0 1]);  grid on;
hold off;

subplot(2,3,4);
imshow(brush_strokes_removed_2b);
title('4. Removed Brush Strokes (MRBF)');

subplot(2,3,5);
imshow(crack_clean_2b);
title('5. Cleaned Crack Map (MRBF)');

subplot(2,3,6);
comp_2b = [crack_map, ones(nRows,2,'logical'), crack_clean_2b];
imshow(comp_2b);
title('6. Before | After MRBF');

sgtitle('PART 2B — MRBF Neural Network Brush Stroke Removal (Hue & Saturation)', ...
        'FontSize',14,'FontWeight','bold');

fprintf('Part 2B — Brush strokes removed : %d pixels\n', nnz(brush_strokes_removed_2b));
fprintf('Part 2B — Crack pixels remaining: %d pixels\n', nnz(crack_clean_2b));

%% ===============================
% Part 2 Combined Comparison — 2A vs 2B
% ===============================
figure('Name','PART 2 — Method Comparison','NumberTitle','off', ...
       'Units','normalized','OuterPosition',[0 0 1 1]);

subplot(1,4,1);
imshow(crack_map);
title('Raw Crack Map','FontSize',11,'FontWeight','bold');

subplot(1,4,2);
imshow(crack_clean_2a);
title({'2A: Semi-Auto','(Grassfire)'},'FontSize',11,'FontWeight','bold');

subplot(1,4,3);
imshow(crack_clean_2b);
title({'2B: MRBF Neural Net','(Hue & Saturation)'},'FontSize',11,'FontWeight','bold');

subplot(1,4,4);
% Combined: keep only pixels flagged by BOTH methods (conservative union)
crack_clean_combined = crack_clean_2a & crack_clean_2b;
imshow(crack_clean_combined);
title({'Combined: 2A AND 2B','(Most Conservative)'},'FontSize',11,'FontWeight','bold');

sgtitle('PART 2 — Comparison: Raw | Grassfire (2A) | MRBF (2B) | Combined', ...
        'FontSize',13,'FontWeight','bold');

% Choose which result feeds into Part 3.
% Default: use Part 2A result (grassfire) as it is seeded by user.
% You can change to crack_clean_2b or crack_clean_combined.
crack_clean = crack_clean_2a;
fprintf('\nPart 3 will use Part 2A (Grassfire) result as crack_clean.\n');
fprintf('To change, set crack_clean = crack_clean_2b or crack_clean_combined.\n');


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