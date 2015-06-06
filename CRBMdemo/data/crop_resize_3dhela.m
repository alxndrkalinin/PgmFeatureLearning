%%%% 3D Hela cells pre-processing

function crop_resize_3dhela(path)

    % default size of images in X and Y
    XY_DIM = 1024;
    Z_DIM = 30;
    [X, Y, Z] = meshgrid(1:XY_DIM, 1:XY_DIM, 1:Z_DIM);
    X = uint8(X);
    Y = uint8(Y);
    Z = uint8(Z);

    % open tif images as stack by path
    cellList = dir(strcat(path, '/cell*'));
    nCells = size(cellList);
    data{nCells(1)} = []; % pre-allocate
    
    for cellIdx = 1:nCells(1) % loop through directories with cells

        cellPath = strcat(path, '/cell', num2str(cellIdx));        
        maskFile = dir(strcat(cellPath, '/crop/*.tif'));
        mask = imread(strcat(cellPath, '/crop/', maskFile(1).name));
        
        tiffList = dir(strcat(cellPath, '/prot/*.tif'));
        nTiffImages = size(tiffList);
        fileName = tiffList(1).name(1:strfind(tiffList(1).name, '.z') + 1);
        
        img = zeros(XY_DIM, XY_DIM, nTiffImages(1), 'uint8'); % pre-allocate
        for ii = 1:nTiffImages(1) % loop throught separate tifs
            img(:, :, ii) = imread(strcat(cellPath, ...
                '/prot/', fileName, num2str(ii), '.tif')) .* mask; % mask out
        end
        
        % find centroid of most abundant slice
        level = graythresh(img);
        mask = int8(img > level * 255);
        cZ = uint8(ceil(mean(Z(mask == 1))));
        slice = mask(:, :, cZ);
        stat = regionprops(slice, 'centroid');
        cropdImg = zeros(601, 601, size(img, 3), 'uint8');
        % zero-padding in X and Y to allow valid crop
        img = padarray(img, [300 300]);
        for x = 1: numel(stat)
            if ~isnan(stat(x).Centroid(1))
                for j = 1:size(img, 3) 
                    cropdImg(:, :, j) = imcrop(img(:, :, j), ...
                        [stat(x).Centroid(1) - 300, ...
                        stat(x).Centroid(2) - 300, 600, 600]);
                end
            end
        end
        img = cropdImg; clear cropdImg;
        
        % zero-padding in Z
        imgSize = size(img);
        padSize = imgSize(1) - imgSize(3);
        if mod(padSize, 2) == 0
            img = padarray(img, [0, 0, padSize / 2]);
        else
            img = padarray(img, [0, 0, floor(padSize / 2)], 'pre');
            img = padarray(img, [0, 0, ceil(padSize / 2)], 'post');
        end
        
        img = imresize(img, 0.25);
        
        data(cellIdx) = {img};
    end
    save(strcat(path, '.mat'), 'data'); % save to corresponding file
end