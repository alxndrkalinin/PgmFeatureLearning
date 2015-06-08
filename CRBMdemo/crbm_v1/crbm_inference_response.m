%%% hidden unit inference 
%%% of convolutional restricted Boltzmann machine
%%% with probabilistic max-pooling

function [H, HP, Hc, HPc] = crbm_inference_response(gpuMode, vis, CRBM, sigma, spacing)
%
if ~exist('sigma','var') || isempty(sigma), 
    sigma = 1;
end

numvis = size(vis, 4);
numhid = size(CRBM.W, 5);

% transfer data to GPU
if gpuMode ~= 0
    CRBM.W = gpuArray(CRBM.W);
    vis = gpuArray(vis);
    hidprobs = zeros(size(vis ,1) - size(CRBM.W, 1) + 1, ...
        size(vis, 2) - size(CRBM.W, 2) + 1, size(vis, 3) - size(CRBM.W, 3) + 1, ...
        numhid, 'single', 'gpuArray');
else
    hidprobs = zeros(size(vis, 1) - size(CRBM.W, 1) + 1, ...
        size(vis, 2) - size(CRBM.W, 2) + 1, size(vis, 3) - size(CRBM.W, 3) + 1, ...
        numhid, 'single');
end

for b = 1:numhid,
    for c = 1:numvis,
        hidprobs(:, :,:,b) = hidprobs(:,:,:,b) + ...
            convn(vis(:,:,:,c), CRBM.W(end:-1:1, end:-1:1, end:-1:1, c, b), 'valid');
    end
    hidprobs(:,:,:,b) = hidprobs(:,:,:,b) + CRBM.hbias(b);
end
clear vis CRBM;

hidprobs = 1 / (sigma^2) .* hidprobs;

% gather data from GPU
if gpuMode ~= 0
    hidprobs = gather(hidprobs);
end

[H, HP, Hc, HPc] = sample_multrand(hidprobs, struct('gpu', gpuMode), spacing);

return;
