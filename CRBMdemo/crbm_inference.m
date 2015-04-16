%%% hidden unit inference 
%%% of convolutional restricted Boltzmann machine
%%% with probabilistic max-pooling

function PAR = crbm_inference(CRBM, PAR, params, opt)
%
if ~exist('opt','var'), opt = 'pos'; end

PAR.hidprobs = CRBM.hbiasmat;
if strcmp(opt,'pos'),
    %%% --- positive phase --- %%%
    for c = 1:params.numvis,
        for b = 1:params.numhid,
            try
                if params.gpu ~= 0
                    reset(params.gpu);
                    vis = gpuArray(PAR.vis(:,:,:,c));
                    Wlr = gpuArray(CRBM.Wlr(:,:,:, b,c));
                    gpuConv = convn(vis, Wlr, 'valid');
                    PAR.hidprobs(:,:,:,b) = PAR.hidprobs(:,:,:,b) + gather(gpuConv);
                    reset(params.gpu);
                else
                    msg = 'GPU is not available.';
                    error(msg);
                end
            catch
                PAR.hidprobs(:,:,:,b) = PAR.hidprobs(:,:,:,b) + convn(PAR.vis(:,:,:,c), CRBM.Wlr(:,:,:, b,c), 'valid');
            end
        end
    end
elseif strcmp(opt,'neg'),
    %%% --- negative phase --- %%%
    for c = 1:params.numvis,
        for b = 1:params.numhid,
            try
                if params.gpu ~= 0
                    reset(params.gpu);
                    vis = gpuArray(PAR.negdata(:,:,:,c));
                    Wlr = gpuArray(CRBM.Wlr(:,:,:, b,c));
                    gpuConv = convn(vis, Wlr, 'valid');
                    PAR.hidprobs(:,:,:,b) = PAR.hidprobs(:,:,:,b) + gather(gpuConv);
                    reset(params.gpu);
                else
                    msg = 'GPU is not available.';
                    error(msg);
                end
            catch
                PAR.hidprobs(:,:,:,b) = PAR.hidprobs(:,:,:,b) + convn(PAR.negdata(:,:,:,c), CRBM.Wlr(:,:,:, b,c), 'valid');
            end
        end
    end
end

PAR.hidprobs = 1 / (params.sigma^2) .* PAR.hidprobs;
[PAR.hidstates, PAR.hidprobs] = sample_multrand(PAR.hidprobs, params);

return;
