function [H HP Hc HPc] = sample_multrand(poshidexp, params, spacing)
if ~exist('spacing','var'),
    spacing = params.spacing;
end

% poshidexp is 3d array
% poshidprobs_mult = zeros(spacing^2 + 1, size(poshidexp, 1) * size(poshidexp, 2) * size(poshidexp, 3) / spacing^2);
% poshidprobs_mult(end,:) = 0;

poshidprobs_mult = zeros(spacing^3 + 1, size(poshidexp, 1) * size(poshidexp, 2) * size(poshidexp, 3) * size(poshidexp, 4) / spacing^3);
poshidprobs_mult(end,:) = 0;

for d = 1:spacing,
    for c = 1:spacing,
        for r = 1:spacing,
            temp = poshidexp(r:spacing:end, c:spacing:end, d:spacing:end, :);
            poshidprobs_mult((d - 1) * 2 * spacing + (c - 1) * spacing + r, :) = temp(:);
        end
    end
end

% substract from max exponent to make values numerically more stable
poshidprobs_mult = bsxfun(@minus, poshidprobs_mult, max(poshidprobs_mult, [], 1));
poshidprobs_mult = exp(poshidprobs_mult);

[S1 P1] = multrand_col(poshidprobs_mult');
S = S1';
P = P1';
clear S1 P1

% convert back to original sized matrix
H = zeros(size(poshidexp));
HP = zeros(size(poshidexp));
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

if nargout >2
    Sc = sum(S(1:end-1,:));
    Pc = sum(P(1:end-1,:));
    Hc = reshape(Sc, [size(poshidexp, 1) / spacing, size(poshidexp, 2) / spacing, ...
        size(poshidexp, 3) / spacing, size(poshidexp, 4)]);
    HPc = reshape(Pc, [size(poshidexp, 1) / spacing, size(poshidexp, 2) / spacing, ...
        size(poshidexp, 3) / spacing, size(poshidexp, 4)]);
end


return

function [S P] = multrand_col(P)
% P is 2-d matrix: 2nd dimension is # of choices

try
    if params.gpu ~= 0
        reset(params.gpu);
        gpuP = gpuArray(P);
        gpuSumP = sum(gpuP, 2);
        gpuP = bsxfun(@rdivide, gpuP, gpuSumP);

        gpuCumP = cumsum(gpuP, 2);
        gpuUnifrnd = rand([size(P, 1), 1], 'gpuArray');
        gpuTemp = bsxfun(@gt, gpuCumP, gpuUnifrnd);
        gpuSindx = diff(gpuTemp, 1, 2);
        gpuS = zeros(size(gpuP), 'gpuArray');
        gpuS(:,1) = 1 - sum(gpuSindx, 2);
        gpuS(:, 2:end) = gpuSindx;

        S = gather(gpuS);
        P = gather(gpuP);
        reset(params.gpu);
    else
        msg = 'GPU is not available.';
        error(msg);
    end
catch
    sumP = sum(P,2);
    P = bsxfun(@rdivide, P, sumP);

    cumP = cumsum(P,2);
    unifrnd = rand(size(P,1),1);
    temp = bsxfun(@gt,cumP,unifrnd);
    Sindx = diff(temp,1,2);
    S = zeros(size(P));
    S(:,1) = 1-sum(Sindx,2);
    S(:,2:end) = Sindx;
end

return;

