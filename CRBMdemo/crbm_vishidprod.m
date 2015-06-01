%%% compute gradient w.r.t. weight tensor between visible and hidden units

function PAR = crbm_vishidprod(PAR, params, opt)
if ~exist('opt','var'), opt = 'pos'; end

% transfer data to GPU
if params.gpu ~= 0
    PAR = gpuArray(PAR);
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
end

return