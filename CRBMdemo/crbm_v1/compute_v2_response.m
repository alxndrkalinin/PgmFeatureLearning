function resp = compute_v2_response(class, CRBM, params, spacing, nimg)

if ~exist('spacing','var'),
    spacing = params.spacing;
end

if ~exist('nimg','var'),
    nimg = [];
end

CRBM.W = double(CRBM.W);
CRBM.hbias = double(CRBM.hbias);

switch class
    case 'Faces_easy',
    case 'car_side',
    case 'cells'
    case 'response_layer1'
end

resp = compute_v2_response_sub(class, CRBM, params, spacing, nimg);

return
