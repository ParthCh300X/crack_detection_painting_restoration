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

