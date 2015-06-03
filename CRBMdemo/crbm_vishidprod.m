%%% compute gradient w.r.t. weight tensor between visible and hidden units

function PAR = crbm_vishidprod(PAR, params, opt)
if ~exist('opt','var'), opt = 'pos'; end

% transfer data to GPU
if params.gpu ~= 0
    PAR.hidprobs = gpuArray(PAR.hidprobs);
    if strcmp(opt, 'pos')
        PAR.vis = gpuArray(PAR.vis);
        PAR.posprods = gpuArray(PAR.posprods);
    elseif strcmp(opt, 'neg')
        PAR.negdata = gpuArray(PAR.negdata);
        PAR.negprods = gpuArray(PAR.negprods);
    end
end

selidx1 = size(PAR.hidprobs, 1):-1:1;
selidx2 = size(PAR.hidprobs, 2):-1:1;
selidx3 = size(PAR.hidprobs, 3):-1:1;

if strcmp(opt,'pos'),
    %%% --- positive phase --- %%%
    for c = 1:params.numvis,
        for b = 1:params.numhid,
            PAR.posprods(:,:,:,c,b) = convn(PAR.vis(:,:,:,c), PAR.hidprobs(selidx1, selidx2, selidx3, b), 'valid');
        end
    end
elseif strcmp(opt,'neg'),
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