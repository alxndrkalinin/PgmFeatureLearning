function [H HP Hc HPc imdata_v0] = crbm_v2_response(gpuMode, im2, CRBM, sigma, spacing)
%
if ~exist('sigma','var') || isempty(sigma), 
    sigma = 1; 
end
if ~exist('noiselevel', 'var'), 
    noiselevel = 0.5; 
end

% im2 = im;
for k = 1:size(im2, 4)
    im2(:,:,:,k) = im2(:,:,:,k) - mean(mean(mean(im2(:,:,:,k))));
    im2(:,:,:,k) = im2(:,:,:,k) / sqrt(mean(mean(mean(im2(:,:,:,k) .^ 2))));
end

if ndims(CRBM.W) == 3,
    ws = sqrt(size(CRBM.W, 1));
elseif ndims(CRBM.W) == 4,
    ws = size(CRBM.W, 1);
elseif ndims(CRBM.W) == 5,
    ws = size(CRBM.W, 1);
end

for k = 1:size(im2, 4)
   im2(:,:,:,k) = trim_image(im2(:,:,:,k), ws, spacing);
end
imdata_v0 = im2 / 1.5;

%%% compute response
[H, HP, Hc, HPc] = crbm_inference_response(imdata_v0, CRBM, sigma, spacing, gpuMode);

return

function im2 = trim_image(im2, ws, spacing)
% % Trim image so that it matches the spacing.
if mod(size(im2, 1) - ws + 1, spacing) ~= 0
    n = mod(size(im2, 1) - ws + 1, spacing);
    im2(1 : floor(n / 2), : , :) = [];
    im2(end - ceil(n / 2) + 1 : end, : ,:) = [];
end
if mod(size(im2, 2) - ws + 1, spacing) ~= 0
    n = mod(size(im2, 2) - ws + 1, spacing);
    im2(:, 1 : floor(n / 2), :) = [];
    im2(:, end - ceil(n / 2) + 1 : end, :) = [];
end
if mod(size(im2, 3) - ws + 1, spacing)~=0
    n = mod(size(im2, 3) - ws + 1, spacing);
    im2(:, :, 1 : floor(n / 2)) = [];
    im2(:, :, end - ceil(n / 2) + 1 : end) = [];
end
return
