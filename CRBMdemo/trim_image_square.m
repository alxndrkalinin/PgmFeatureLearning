function imresp = trim_image_square(imdata, ws, batch_ws, spacing)
% trim the image into batch_ws x batch_ws
[rows, cols, depth, ~] = size(imdata);

rowstart = randi(rows - batch_ws + 1);
rowidx = rowstart : rowstart + batch_ws - 1;
colstart = randi(cols - batch_ws + 1);
colidx = colstart : colstart + batch_ws - 1;
depthstart = randi(depth - batch_ws + 1);
depthidx = depthstart : depthstart + batch_ws - 1;

imresp = imdata(rowidx, colidx, depthidx, :);
imresp = trim_image_for_spacing(imresp, ws, spacing);

return;

function im2 = trim_image_for_spacing(im2, ws, spacing)
% % Trim image so that it matches the spacing.
if mod(size(im2, 1) - ws + 1, spacing) ~= 0
    n = mod(size(im2, 1) - ws + 1, spacing);
    im2(1 : floor(n / 2), : , :) = [];
    im2(end - ceil(n / 2) + 1 : end, : ,:) = [];
end
if mod(size(im2, 2) - ws + 1, spacing) ~= 0
    n = mod(size(im2, 2) - ws + 1, spacing);
    im2(:, 1 : floor(n / 2), :) = [];
    im2(:, end - ceil(n / 2) + 1 : end, :) = [];
end
if mod(size(im2, 3) - ws + 1, spacing)~=0
    n = mod(size(im2, 3) - ws + 1, spacing);
    im2(:, :, 1 : floor(n / 2)) = [];
    im2(:, :, end - ceil(n / 2) + 1 : end) = [];
end
return
