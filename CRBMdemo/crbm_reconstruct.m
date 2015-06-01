%%% visible unit inference (reconstruction)
%%% of convolutional restricted Boltzmann machine
%%% with probabilistic max-pooling

function PAR = crbm_reconstruct(CRBM, PAR, params, opt)

if ~exist('opt','var'), opt = 'neg'; end

% transfer data to GPU
if params.gpu ~= 0
    CRBM.W = gpuArray(CRBM.W);
    if strcmp(opt, 'recon')
        PAR.reconst = gpuArray(PAR.reconst);
        PAR.hidprobs = gpuArray(PAR.hidprobs);
    elseif strcmp(opt, 'neg')
        PAR.negdata = gpuArray(PAR.negdata);
        PAR.hidstates = gpuArray(PAR.hidstates);
    end
end

if strcmp(opt, 'recon'),
    %%% --- reconstruction --- %%%
    PAR.reconst = CRBM.vbiasmat;
    for b = 1:params.numhid,
        for c = 1:params.numvis,
            PAR.reconst(:,:,:,c) = PAR.reconst(:,:,:,c) + convn(PAR.hidprobs(:,:,:,b), CRBM.W(:,:,:,c,b), 'full');
        end
    end
    
    if strcmp(params.intype, 'binary'),
        PAR.reconst = (1 / params.sigma^2) * PAR.reconst;
        PAR.reconst = sigmoid(PAR.reconst);
    end
    
elseif strcmp(opt,'neg'),
    %%% --- negative phase --- %%%
    PAR.negdata = CRBM.vbiasmat;
    for b = 1:params.numhid,
        for c = 1:params.numvis,
            PAR.negdata(:,:,:,c) = PAR.negdata(:,:,:,c) + convn(PAR.hidstates(:,:,:,b), CRBM.W(:,:,:,c,b), 'full');
        end
    end
    
    if strcmp(params.intype,'binary'),
        PAR.negdata = (1 / params.sigma ^ 2) * PAR.negdata;
        PAR.negdata = sigmoid(PAR.negdata);
    end
end

% gather data from GPU
if params.gpu ~= 0
    CRBM.W = gather(CRBM.W);
    if strcmp(opt, 'pos')
        PAR.reconst = gather(PAR.reconst);
        PAR.hidprobs = gather(PAR.hidprobs);
        PAR.vis = gather(PAR.vis);
    elseif strcmp(opt, 'neg')
        PAR.negdata = gather(PAR.negdata);
        PAR.hidstates = gather(PAR.hidstates);
    end
end

return
