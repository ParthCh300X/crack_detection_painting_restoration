clc;
clear;
close all;

%% ===============================
% Load Image
% ===============================
img = imread('input_img.png');

if size(img,3)==3
    gray = rgb2gray(img);
else
    gray = img;
end

gray = im2uint8(gray);

%% ===============================
% Structuring Element
% Square 3x3 with 2 dilations
% ===============================
B = strel('square',3);

SE = B;
for k = 1:2
    SE = imdilate(SE.Neighborhood,B.Neighborhood);
    SE = strel(SE);
end

%% ===============================
% Morphological Operations
% ===============================
dilated       = imdilate(gray,SE);
eroded        = imerode(gray,SE);
opened        = imopen(gray,SE);
closed        = imclose(gray,SE);

white_tophat  = imsubtract(gray,opened);
black_tophat  = imsubtract(closed,gray);

%% ===============================
% Crack Extraction
% ===============================
T = 18;   % adjust 15 to 30

crack_map = black_tophat > T;
crack_map = bwareaopen(crack_map,5);

BW_Crack_img = ~crack_map;

%% ===============================
% Displaying the outputs of each morphological operations
% ===============================
figure('Name','Complete Morphological Processing','NumberTitle','off');

subplot(3,4,1);
imshow(gray);
title('Original');

subplot(3,4,2);
imshow(SE.Neighborhood);
title('Final SE');

subplot(3,4,3);
imshow(dilated);
title('Dilation');

subplot(3,4,4);
imshow(eroded);
title('Erosion');

subplot(3,4,5);
imshow(opened);
title('Opening');

subplot(3,4,6);
imshow(closed);
title('Closing');

subplot(3,4,7);
imshow(white_tophat,[]);
title('White Top Hat');

subplot(3,4,8);
imshow(black_tophat,[]);
title('Black Top Hat');

subplot(3,4,9);
imshow(crack_map);
title('Binary Crack Map');

subplot(3,4,10);
imshow(BW_Crack_img);
title('BW Crack Image');

subplot(3,4,11);
imshowpair(white_tophat,black_tophat,'montage');
title('WTH / BTH');

subplot(3,4,12);
imshowpair(gray,BW_Crack_img,'montage');
title('Original / Final Crack Image');
