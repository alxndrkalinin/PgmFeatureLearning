%%% hidden unit inference 
%%% of convolutional restricted Boltzmann machine
%%% with probabilistic max-pooling

function PAR = crbm_inference(CRBM, PAR, params, opt)
%
if ~exist('opt','var'), opt = 'pos'; end

PAR.hidprobs = CRBM.hbiasmat;

% transfer data to GPU
if params.gpu ~= 0
    PAR.hidprobs = gpuArray(PAR.hidprobs);
    CRBM.Wlr = gpuArray(CRBM.Wlr);
    if strcmp(opt, 'pos')
        PAR.vis = gpuArray(PAR.vis);
    elseif strcmp(opt, 'neg')
        PAR.negdata = gpuArray(PAR.negdata);
    end
end

if strcmp(opt, 'pos'),
    %%% --- positive phase --- %%%
    for c = 1:params.numvis,
        for b = 1:params.numhid,
            PAR.hidprobs(:,:,:,b) = PAR.hidprobs(:,:,:,b) + convn(PAR.vis(:,:,:,c), CRBM.Wlr(:,:,:, b,c), 'valid');
        end
    end
elseif strcmp(opt, 'neg'),
    %%% --- negative phase --- %%%
    for c = 1:params.numvis,
        for b = 1:params.numhid,
            PAR.hidprobs(:,:,:,b) = PAR.hidprobs(:,:,:,b) + convn(PAR.negdata(:,:,:,c), CRBM.Wlr(:,:,:, b,c), 'valid');
        end
    end
end

clear CRBM.Wlr;

PAR.hidprobs = 1 / (params.sigma^2) .* PAR.hidprobs;

% gather data from GPU
if params.gpu ~= 0
    PAR.hidprobs = gather(PAR.hidprobs);
    if strcmp(opt, 'pos')
        PAR.vis = gather(PAR.vis);
    elseif strcmp(opt, 'neg')
        PAR.negdata = gather(PAR.negdata);
    end
end

[PAR.hidstates, PAR.hidprobs] = sample_multrand(PAR.hidprobs, params);

return;
