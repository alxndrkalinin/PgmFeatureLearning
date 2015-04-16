function resp = compute_v1_response_sub(dataname, CRBM, params, spacing, nimg)
%%
if ~exist('spacing','var') || isempty(spacing), 
    spacing = params.spacing; 
end

%%% load images
load data/cells/data_41_cube.mat;

ws_pad = 0;

imsize = 41;
D = 20;

for k = 1:length(data),
    im = data{k};
    
    %%% compute response
    [~, ~, ~, HPc] = crbm_v1_response(params.gpu, im, CRBM, params.sigma, spacing, imsize, D, ws_pad);
    
    BUF = 1;
    cur_resp = HPc;
    cur_resp(1:BUF,:,:,:)=0;
    cur_resp(:, 1:BUF,:,:)=0;
    cur_resp(:, :,1:BUF,:)=0;
    cur_resp(end-BUF+1:end,:,:,:)=0;
    cur_resp(:, end-BUF+1:end,:,:)=0;
    cur_resp(:, :,end-BUF+1:end,:)=0;
    
    %%% store in cell
    resp{k} = cur_resp;
end

fprintf('\n');

return