%%% hidden unit inference 
%%% of convolutional restricted Boltzmann machine
%%% with probabilistic max-pooling

function [H HP Hc HPc] = crbm_inference_response(vis, CRBM, sigma, spacing)
%
if ~exist('sigma','var') || isempty(sigma), 
    sigma = 1;
end

numvis = size(vis, 4);
numhid = size(CRBM.W, 5);

hidprobs = zeros(size(vis,1)-size(CRBM.W,1)+1, size(vis,2)-size(CRBM.W,2)+1, size(vis,3)-size(CRBM.W,3)+1, numhid);
for b = 1:numhid,
    for c = 1:numvis,
        try
            gpuVis = gpuArray(vis(:,:,:,c));
            gpuW = gpuArray(CRBM.W(end:-1:1, end:-1:1, end:-1:1, c, b));
            hidprobs(:, :,:,b) = hidprobs(:,:,:,b) + gather(convn(gpuVis, gpuW, 'valid'));g
        catch
            hidprobs(:, :,:,b) = hidprobs(:,:,:,b) + convn(vis(:,:,:,c), CRBM.W(end:-1:1, end:-1:1, end:-1:1, c, b), 'valid');
        end
    end
    hidprobs(:,:,:,b) = hidprobs(:,:,:,b) + CRBM.hbias(b);
end

hidprobs = 1/(sigma^2).*hidprobs;
[H HP Hc HPc] = sample_multrand(hidprobs, [], spacing);

return;
