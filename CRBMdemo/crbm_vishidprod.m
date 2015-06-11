%%% compute gradient w.r.t. weight tensor between visible and hidden units

function PAR = crbm_vishidprod(PAR, params, opt)
if ~exist('opt','var'), opt = 'pos'; end

% transfer data to GPU
if params.gpu ~= 0
    PAR.hidprobs = gpuArray(PAR.hidprobs);
    if strcmp(opt, 'pos')
        PAR.vis = gpuArray(PAR.vis);
        if isfield(PAR, 'posprods')
            PAR.posprods = gpuArray(PAR.posprods);
        end
    elseif strcmp(opt, 'neg')
        PAR.negdata = gpuArray(PAR.negdata);
        if isfield(PAR, 'negprods')
            PAR.negprods = gpuArray(PAR.negprods);
        end
    end
end

selidx1 = size(PAR.hidprobs, 1):-1:1;
selidx2 = size(PAR.hidprobs, 2):-1:1;
selidx3 = size(PAR.hidprobs, 3):-1:1;

if strcmp(opt,'pos'),
    
    nX = max(size(PAR.vis, 1) - ...
        size(PAR.hidprobs(selidx1, selidx2, selidx3, :), 1) + 1, 0);
    nY = max(size(PAR.vis, 2) - ...
        size(PAR.hidprobs(selidx1, selidx2, selidx3, :), 2) + 1, 0);
    nZ = max(size(PAR.vis, 3) - ...
        size(PAR.hidprobs(selidx1, selidx2, selidx3, :), 3) + 1, 0);

    PAR.posprods = zeros(nX, nY, nZ, params.numvis, params.numhid);
    
    %%% --- positive phase --- %%%
    for c = 1:params.numvis,
        for b = 1:params.numhid,
            PAR.posprods(:,:,:,c,b) = convn(PAR.vis(:,:,:,c), PAR.hidprobs(selidx1, selidx2, selidx3, b), 'valid');
        end
    end
elseif strcmp(opt,'neg'),
    
    nX = max(size(PAR.negdata, 1) - ...
        size(PAR.hidprobs(selidx1, selidx2, selidx3, :), 1) + 1, 0);
    nY = max(size(PAR.negdata, 2) - ...
        size(PAR.hidprobs(selidx1, selidx2, selidx3, :), 2) + 1, 0);
    nZ = max(size(PAR.negdata, 3) - ...
        size(PAR.hidprobs(selidx1, selidx2, selidx3, :), 3) + 1, 0);

    PAR.negprods = zeros(nX, nY, nZ, params.numvis, params.numhid);
    
    %%% --- negative phase --- %%%
    for c = 1:params.numvis,
        for b = 1:params.numhid,
            PAR.negprods(:,:,:,c,b) = convn(PAR.negdata(:,:,:,c), PAR.hidprobs(selidx1, selidx2, selidx3, b), 'valid');
        end
    end
end

% gather from GPU
if params.gpu ~= 0
    PAR = gather(PAR);
    if strcmp(opt, 'pos')
        PAR.vis = gather(PAR.vis);
        PAR.posprods = gather(PAR.posprods);
    elseif strcmp(opt, 'neg')
        PAR.negdata = gather(PAR.negdata);
        PAR.negprods = gather(PAR.negprods);
    end
end

return