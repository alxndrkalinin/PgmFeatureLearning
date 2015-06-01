function [H HP Hc HPc] = sample_multrand(poshidexp, params, spacing)
if ~exist('spacing','var'),
    spacing = params.spacing;
end

% transfer data to GPU and pre-allocate
if params.gpu ~= 0
    poshidexp = gpuArray(poshidexp);
    poshidprobs_mult = zeros(spacing^3 + 1, size(poshidexp, 1) * ...
        size(poshidexp, 2) * size(poshidexp, 3) * ...
        size(poshidexp, 4) / spacing^3, 'single', 'gpuArray');
else
    poshidprobs_mult = zeros(spacing^3 + 1, size(poshidexp, 1) * ...
    size(poshidexp, 2) * size(poshidexp, 3) * ...
    size(poshidexp, 4) / spacing^3, 'single');
end

poshidprobs_mult(end,:) = 0;

for d = 1:spacing,
    for c = 1:spacing,
        for r = 1:spacing,
            temp = poshidexp(r:spacing:end, c:spacing:end, d:spacing:end, :);
            poshidprobs_mult((d - 1) * 2 * spacing + (c - 1) * spacing + r, :) = temp(:);
        end
    end
end
clear temp;

% substract from max exponent to make values numerically more stable
poshidprobs_mult = bsxfun(@minus, poshidprobs_mult, max(poshidprobs_mult, [], 1));
poshidprobs_mult = exp(poshidprobs_mult);
poshidprobs_mult = poshidprobs_mult';

sumP = sum(poshidprobs_mult, 2);
P = bsxfun(@rdivide, poshidprobs_mult, sumP);
clear poshidprobs_mult sumP;

cumP = cumsum(P, 2);
if params.gpu ~= 0
    unifrnd = rand(size(P,1), 1, 'single', 'gpuArray');
else
    unifrnd = rand(size(P,1), 1, 'single');
end
tmp = bsxfun(@gt, cumP, unifrnd);
clear cumP unifrnd;

Sindx = diff(tmp, 1, 2);
clear tmp;

if params.gpu ~= 0
    S = zeros(size(P), 'single', 'gpuArray');
else
    S = zeros(size(P), 'single');
end
S(:,1) = 1 - sum(Sindx, 2);
S(:,2:end) = Sindx;
clear Sindx;

S = S';
P = P';

% transfer data to GPU and pre-allocate
if params.gpu ~= 0
    H = zeros(size(poshidexp), 'single', 'gpuArray');
    HP = zeros(size(poshidexp), 'single', 'gpuArray');
else
    H = zeros(size(poshidexp), 'single');
    HP = zeros(size(poshidexp), 'single');
end

% convert back to original sized matrix
for d = 1:spacing,
    for c = 1:spacing,
        for r = 1:spacing,
            H(r:spacing:end, c:spacing:end, d:spacing:end, :) = ...
                reshape(S((d - 1) * 2 * spacing + (c - 1) * spacing + r, :), ...
                [size(H, 1) / spacing, size(H, 2) / spacing, size(H, 3) / spacing, size(H, 4)]);
            
            HP(r:spacing:end, c:spacing:end, d:spacing:end, :) = ...
                reshape(P((d - 1) * 2 * spacing + (c - 1) * spacing + r, :), ...
                [size(H, 1) / spacing, size(H, 2) / spacing, size(H, 3) / spacing, size(H, 4)]);
        end
    end
end

% gather data from GPU
if params.gpu ~= 0
    H = gather(H);
    HP = gather(HP);
end

if nargout >2
    Sc = sum(S(1:end-1,:));
    Pc = sum(P(1:end-1,:));
    clear S P;
    
    Hc = reshape(Sc, [size(poshidexp, 1) / spacing, size(poshidexp, 2) / spacing, ...
        size(poshidexp, 3) / spacing, size(poshidexp, 4)]);
    HPc = reshape(Pc, [size(poshidexp, 1) / spacing, size(poshidexp, 2) / spacing, ...
        size(poshidexp, 3) / spacing, size(poshidexp, 4)]); 
    
    % gather data from GPU
    if params.gpu ~= 0
        if nargout >2
            Hc = gather(Hc);
            HPc = gather(HPc);
        end
    end
end

clear poshidexp;

return



