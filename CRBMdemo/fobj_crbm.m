%%% compute gradients using constrastive divergence

function [CRBM, PAR] = fobj_crbm(CRBM, PAR, params, opt)

PAR.ferr = 0;
PAR.sparsity = 0;
PAR.recon_err = zeros(params.numvis, 1, 'single');

for c = 1:params.numvis,
    CRBM.Wlr(:, :, :, :, c) = reshape(CRBM.W(end:-1:1, end:-1:1, end:-1:1, c, :), ...
        [params.ws, params.ws, params.ws, params.numhid]);
end

CRBM.vbiasmat = repmat(permute(CRBM.vbias, [2 3 4 1]), opt.visrow, opt.viscol, opt.visdepth);
CRBM.hbiasmat = repmat(permute(CRBM.hbias, [2 3 4 1]), opt.hidrow, opt.hidcol, opt.hiddepth);

%%% --- positive phase --- %%%
PAR = crbm_inference(CRBM, PAR, params, 'pos');

% ey gradient
PAR = crbm_vishidprod(PAR, params, 'pos');
PAR.poshidact = squeeze(sum(sum(sum(PAR.hidprobs,1),2),3));
PAR.posvisact = squeeze(sum(sum(sum(PAR.vis,1),2),3));

% reconstruction
PAR = crbm_reconstruct(CRBM, PAR, params, 'recon');
%%% ---------------------- %%%


% reconstruction error, sparsity
% imshow3D(PAR.vis);
% figure;
% imshow3D(PAR.reconst);
PAR.reconst_images = [PAR.reconst_images PAR.reconst];
PAR.ferr = sum(sum(sum(sum((PAR.vis - PAR.reconst).*(PAR.vis - PAR.reconst),1),2),3),4)/(opt.vissize*params.numvis);
PAR.sparsity = sum(sum(sum(sum(PAR.hidprobs,1),2),3),4)/(opt.hidsize*params.numhid);
PAR.recon_err = squeeze(sum(sum(sum((PAR.vis - PAR.reconst).*(PAR.vis - PAR.reconst),1),2),3))/opt.vissize;

%%% --- negative phase (CD-K) --- %%%
for kcd = 1:params.kcd,
    % negative data
    PAR = crbm_reconstruct(CRBM, PAR, params, 'neg');
    % hidden unit inference
    PAR = crbm_inference(CRBM, PAR, params, 'neg');
end

PAR.reconst_images_neg = [PAR.reconst_images_neg PAR.negdata];

% ey gradient
PAR = crbm_vishidprod(PAR, params, 'neg');

PAR.neghidact = squeeze(sum(sum(sum(PAR.hidprobs,1),2),3));
PAR.negvisact = squeeze(sum(sum(sum(PAR.negdata,1),2),3));
%%% ----------------------------- %%%

% gradient of log-likelihood
PAR.dW_total = (PAR.posprods-PAR.negprods)/opt.hidsize - params.l2reg*CRBM.W - params.l1reg*((CRBM.W>0)*2-1);
PAR.dh_total = (PAR.poshidact-PAR.neghidact)/opt.hidsize;
PAR.dv_total = (PAR.posvisact-PAR.negvisact)/opt.vissize;

return;
