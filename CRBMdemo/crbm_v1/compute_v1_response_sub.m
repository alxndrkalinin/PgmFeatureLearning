function resp = compute_v1_response_sub(dataname, CRBM, params, spacing, nimg)
%%
if ~exist('spacing','var') || isempty(spacing), 
    spacing = params.spacing; 
end

%%% load images
% load data/cells/class_1_59_size100.mat;
% load data/cells/class_2_59_size100.mat;
% data = [class1_data100 class2_data100];

load data/3DHela/Gpp.mat;
load data/3DHela/Tub.mat;
data = [Gpp(1:40) Tub(1:40)];

resp = cell(length(data));

for k = 1:length(data),
    im = data{k};
    
    %%% compute response
    [~, ~, ~, HPc] = crbm_v1_response(params.gpu, im, CRBM, params.sigma, spacing);
    
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