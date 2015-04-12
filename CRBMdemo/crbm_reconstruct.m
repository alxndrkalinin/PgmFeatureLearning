%%% visible unit inference (reconstruction)
%%% of convolutional restricted Boltzmann machine
%%% with probabilistic max-pooling

function PAR = crbm_reconstruct(CRBM, PAR, params, opt)

if ~exist('opt','var'), opt = 'neg'; end

if strcmp(opt,'recon'),
    %%% --- reconstruction --- %%%
    PAR.reconst = CRBM.vbiasmat;
    for b = 1:params.numhid,
        for c = 1:params.numvis,
            try
                hidprobs = gpuArray(PAR.hidprobs(:,:,:,b));
                W = gpuArray(CRBM.W(:,:,:,c,b));
                gpuConv = convn(hidprobs, W, 'full');
                PAR.reconst(:,:,:,c) = PAR.reconst(:,:,:,c) + gather(gpuConv);
            catch
                PAR.reconst(:,:,:,c) = PAR.reconst(:,:,:,c) + convn(PAR.hidprobs(:,:,:,b), CRBM.W(:,:,:,c,b), 'full');
            end
        end
    end
    
    if strcmp(params.intype,'binary'),
        PAR.reconst = (1/params.sigma^2)*PAR.reconst;
        PAR.reconst = sigmoid(PAR.reconst);
    end
elseif strcmp(opt,'neg'),
    %%% --- negative phase --- %%%
    PAR.negdata = CRBM.vbiasmat;
    for b = 1:params.numhid,
        for c = 1:params.numvis,
            try
                hidstates = gpuArray(PAR.hidstates(:,:,:,b));
                W = gpuArray(CRBM.W(:,:,:,c,b));
                gpuConv = convn(hidstates, W, 'full');
                PAR.negdata(:,:,:,c) = PAR.negdata(:,:,:,c) + gather(gpuConv);
            catch
                PAR.negdata(:,:,:,c) = PAR.negdata(:,:,:,c) + convn(PAR.hidstates(:,:,:,b), CRBM.W(:,:,:,c,b), 'full');
            end
        end
    end
    
    if strcmp(params.intype,'binary'),
        PAR.negdata = (1 / params.sigma ^ 2) * PAR.negdata;
        PAR.negdata = sigmoid(PAR.negdata);
    end
end

return
