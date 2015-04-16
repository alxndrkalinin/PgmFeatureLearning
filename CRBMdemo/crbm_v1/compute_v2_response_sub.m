function resp = compute_v2_response_sub(dataname, CRBM, params, spacing, nimg)
%%
if ~exist('spacing','var') || isempty(spacing), 
    spacing = params.spacing; 
end

%%% load images


%load data/cells/data_41_cube.mat;
load H.mat;
% load data/cells/data_82_cube.mat;

ws_pad = 0;
% fpath = sprintf('data/%s', dataname);
% flist = dir(sprintf('%s/*.jpg', fpath));
% if spacing == 3,
%     imsize = 180;
% elseif spacing == 2,
%     imsize = 150;
% end
imsize = 17;
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
count = 0;
% for k = 1:length(idx),
for k = 1:length(H)
    one_image_4d = H{k};
%     for j = 1:size(H{k},4)

%       im = one_image_4d(:,:,:,j);
    im = one_image_4d;
    
    %%% compute response
    [~, ~, ~, HPc] = crbm_v2_response(im, CRBM, params.sigma, spacing, imsize, D, ws_pad);
    
    BUF = 1;
    cur_resp = HPc;
    cur_resp(1:BUF,:,:,:)=0;
    cur_resp(:, 1:BUF,:,:)=0;
    cur_resp(:, :,1:BUF,:)=0;
    cur_resp(end-BUF+1:end,:,:,:)=0;
    cur_resp(:, end-BUF+1:end,:,:)=0;
    cur_resp(:, :,end-BUF+1:end,:)=0;
    
    %%% store in cell
    resp = cur_resp;
%     resp{count+1} = cur_resp;
%     count =count+1;
%     end
end

fprintf('\n');

return