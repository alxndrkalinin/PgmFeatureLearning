clear all
close all

% Load the data, you may put the following code within a for loop for
% iterating over all data
load Tub.mat
x = Tub{1};
[m,n,p] = size(x);

% View the original image (For your debugging eyes only. Comment this and 
% all the succeeding imshow3D for iterations to avoid crazy shit!)
figure(1); imshow3D(x); title('Original Image');

%% Translation
xy_min = -50; xy_max = 50; % range of translation about x and y axis
z_min = -6; z_max = 6; % range of translation about z axis

% Generate Random translations along the 3 axis
t = [randi([xy_min xy_max],1) randi([xy_min xy_max],1) ...
    randi([z_min z_max],1)];
t_obj = zeros([size(x) 3]);

% Create the displacement matrix for imwarp function
for i=1:3
    D(:,:,:,i) = ones(size(x))*t(i);
end
x_trans = imwarp(x,D);

% View the translation output
figure(2); imshow3D(x_trans); title('After Translation');
x = x_trans;

%% Rotation (Currently only in the x-y plane)
min_angle = -pi; max_angle = pi;
az =  min_angle + (max_angle-min_angle).*rand(1,1); % Rotation about the z axis
% ax =  min_angle + (max_angle-min_angle).*rand(1,1); % Rotation about the x axis
% ay =  min_angle + (max_angle-min_angle).*rand(1,1); % Rotation about the y axis

% Make rotation transformation matrix and apply 
Rz = [ cos(az) sin(az) 0 0;
      -sin(az) cos(az) 0 0;
        0   0   1   0;
       0    0   0   1];
R_obj = affine3d(Rz); % Only rotationg about the z-axis (in X-Y plane)
x_rot = imwarp(x,R_obj);

% View the Rotation output
figure(3); imshow3D(x_rot); title('After Translation + Rotation');
x = x_rot;
%% Scaling (log-uniform) 
smin = log(1/2) ; smax = log(2); % Change scale range within the log 
s = exp(smin + (smax-smin) * rand(1)); % Random Scale factor 
S = [ s 0 0 0 ;
      0 s 0 0 ;
      0 0 s 0 ;
      0 0 0 1];
S_obj = affine3d(S);
x_scaled = imwarp(x,S_obj);

% Bring the image back to the dimensions of the original image
[ms,ns,ps] = size(x_scaled);
if s > 1
    x_scaled = x_scaled(ceil(ms/2)-ceil(m/2-1):ceil(ms/2)+floor(m/2),...
                  ceil(ns/2)-ceil(n/2-1):ceil(ns/2)+floor(n/2),...
                  ceil(ps/2)-ceil(p/2-1):ceil(ps/2)+floor(p/2));
elseif s < 1
    x_scaled = padarray(x_scaled,[ceil((m-ms)/2),ceil((n-ns)/2),...
        ceil((p-ps)/2)],0,'pre');
    x_scaled = padarray(x_scaled,[floor((m-ms)/2),floor((n-ns)/2),...
        floor((p-ps)/2)],0,'post');
end

% View the Scaling output
figure(4); imshow3D(x_scaled); 
title('After Translation + Rotation + Scaling');

       