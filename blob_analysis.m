img = imread('dataset\sim\sim_6.jpg');

% subplot(1, 3, 1);
% imshow(img);


%% Threshold Image

% Convert RGB image to chosen color space
I = rgb2hsv(img);

% Define thresholds for channel 1 based on histogram settings
channel1Min = 0.092;
channel1Max = 0.128;

% Define thresholds for channel 2 based on histogram settings
channel2Min = 0.579;
channel2Max = 0.723;

% Define thresholds for channel 3 based on histogram settings
channel3Min = 0.354;
channel3Max = 0.487;

% Create mask based on chosen histogram thresholds
BW = (I(:,:,1) >= channel1Min ) & (I(:,:,1) <= channel1Max) & ...
    (I(:,:,2) >= channel2Min ) & (I(:,:,2) <= channel2Max) & ...
    (I(:,:,3) >= channel3Min ) & (I(:,:,3) <= channel3Max);

% subplot(1, 3, 2);
% imshow(BW);

%% Noise Removal: Morphological Operations

diskElem = strel('disk', 4);
IBWOpen = imopen(BW, diskElem);

% subplot(1, 3, 2);
imshow(IBWOpen);

%% Blob Analysis