function resp = compute_v1_response_sub(dataname, CRBM, params, spacing, nimg)
%%
if ~exist('spacing','var') || isempty(spacing), 
    spacing = params.spacing; 
end

%%% load images
ws_pad = 0;
% fpath = sprintf('data/%s', dataname);
% flist = dir(sprintf('%s/*.jpg', fpath));
% if spacing == 3,
%     imsize = 180;
% elseif spacing == 2,
%     imsize = 150;
% end
imsize = 41;
D = 20;
% 
% if ~exist('nimg', 'var') || isempty(nimg),
%     nimg = length(flist);
% end
% 
% resp = cell(min(length(flist), nimg),1);
% if length(resp) < length(flist),
%     idx = randsample(1:length(flist), min(length(flist), nimg));
% else
%     idx = 1:length(flist);
% end

load data/cells/data_41_cube.mat;

% for k = 1:length(idx),
for k = 1:length(data41),
%     imidx = idx(k);
%     fprintf('[%d]', imidx);
%     fname = sprintf('%s/%s', fpath, flist(imidx).name);
%     im = imread(fname);

    im = data41{k};
    
    %%% compute response
    [~, ~, ~, HPc] = crbm_v1_response(im, CRBM, params.sigma, spacing, imsize, D, ws_pad);
    
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