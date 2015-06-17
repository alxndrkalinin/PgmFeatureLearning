function [H HP Hc HPc] = sample_multrand(poshidexp, params, spacing)
if ~exist('spacing','var'),
    spacing = params.spacing;
end
    
poshidprobs_mult = zeros(spacing^3 + 1, size(poshidexp, 1) * ...
    size(poshidexp, 2) * size(poshidexp, 3) * ...
    size(poshidexp, 4) / spacing^3, 'single');

poshidprobs_mult(end,:) = 0;

for d = 1:spacing,
    for c = 1:spacing,
        for r = 1:spacing,
            temp = poshidexp(r:spacing:end, c:spacing:end, d:spacing:end, :);
            poshidprobs_mult((d - 1) * 2 * spacing + (c - 1) * spacing + r, :) = temp(:);
        end
    end
end

poshidexp_size = size(poshidexp);
clear temp poshidexp;

% substract from max exponent to make values numerically more stable
poshidprobs_mult = bsxfun(@minus, poshidprobs_mult, max(poshidprobs_mult, [], 1));
poshidprobs_mult = exp(poshidprobs_mult);
poshidprobs_mult = poshidprobs_mult';

sumP = sum(poshidprobs_mult, 2);
P = bsxfun(@rdivide, poshidprobs_mult, sumP);
clear poshidprobs_mult sumP;

% only P left in memory
cumP = cumsum(P, 2);
unifrnd = rand(size(P,1), 1, 'single');
tmp = bsxfun(@gt, cumP, unifrnd);
clear cumP unifrnd;

Sindx = diff(tmp, 1, 2);
clear tmp;

S = zeros(size(P), 'single');
S(:,1) = 1 - sum(Sindx, 2);
S(:,2:end) = Sindx;
clear Sindx;
S = S';
P = P';

% pre-allocate
H = zeros(poshidexp_size, 'single');
HP = zeros(poshidexp_size, 'single');
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

if nargout > 2
    Sc = sum(S(1:end-1,:));
    Pc = sum(P(1:end-1,:));
    clear S P;
    
    Hc = reshape(Sc, [poshidexp_size(1) / spacing, poshidexp_size(2) / spacing, ...
        poshidexp_size(3) / spacing, poshidexp_size(4)]);
    HPc = reshape(Pc, [poshidexp_size(1) / spacing, poshidexp_size(2) / spacing, ...
        poshidexp_size(3) / spacing, poshidexp_size(4)]);
end

return
