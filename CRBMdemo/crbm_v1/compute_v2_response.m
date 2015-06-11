function resp = compute_v2_response(class, CRBM, params, spacing)

if ~exist('spacing','var'),
    spacing = params.spacing;
end

if ~exist('nimg','var'),
    nimg = [];
end

CRBM.W = single(CRBM.W);
CRBM.hbias = single(CRBM.hbias);

switch class
    case 'Faces_easy',
    case 'car_side',
    case 'cells'
    case 'response_layer1'
end

resp = compute_v2_response_sub(CRBM, params, spacing);

return
