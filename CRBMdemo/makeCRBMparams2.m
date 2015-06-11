function params = makeCRBMparams2(params)
warning off all;

% reference parameters
ref_params = [];
% filter size
ref_params.ws = 3;
ref_params.batch_ws = [];
% number of filters / hidden units
ref_params.numhid = 100;
ref_params.kcd = 1;
ref_params.l1reg = 0;
ref_params.l2reg = 1e-4;
ref_params.batchsize = [];
ref_params.pbias = 0;
ref_params.plambda = 0.1;
ref_params.epsilon = 0.01;
ref_params.epsdecay = 0.01;
ref_params.verbose = 1;
ref_params.savedir = 'results';
ref_params.fname_save = [];
ref_params.intype = 'real';
ref_params.sptype = 'approx';
ref_params.dataset = '';
ref_params.maxiter = 7;
% target sparsity level
ref_params.eta_sparsity = 0;
ref_params.eta_sigma = 0.05;
ref_params.optmminit = 0;
ref_params.mmiter = 1;
ref_params.optdouble = 1;
ref_params.nlayer = 2;
ref_params.showfig = 0;
ref_params.spacing = 2;
ref_params.eta_sparsity = 0;
% use gpu for convn
ref_params.gpu = 0;

% new parameters
new_params = struct;
if ~exist('params','var') || isempty(params), params = struct; end
names = fieldnames(params);
for i = 1:length(names),
    v = getfield(params,names{i});
    new_params = setfield(new_params,lower(names{i}),v);
end
clear params;
params = new_params; clear new_params;
params = catstruct(params,'sorted');

% insert blank fields
params = catstruct(ref_params,params);
if params.plambda == 0, params.pbias = 0; end
if params.pbias*params.numhid > 6, params.optmminit = 0; end
mkdir(params.savedir);

return;
