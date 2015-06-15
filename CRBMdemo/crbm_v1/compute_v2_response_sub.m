function resp = compute_v2_response_sub(CRBM, params, spacing)
%%
if ~exist('spacing','var') || isempty(spacing), 
    spacing = params.spacing; 
end

%%% load prev layer response
load H.mat
%load H_class.mat;
%H = H_class;

resp = cell(length(H));

% for k = 1:length(idx),
for k = 1:length(H)
    im = H{k};
    
    %%% compute response
    [~, ~, ~, HPc] = crbm_v2_response(params.gpu, im, CRBM, params.sigma, spacing);
    
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
