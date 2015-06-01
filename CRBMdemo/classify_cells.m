function classify_cells()

load 'H_84.mat'
load 'H2_84.mat'

hFeat = [];
hFeat2 = [];

for k = 1:length(H)
    hFeat = vertcat(hFeat, reshape(H{k}, [1, (size(H{k},1) .^ 3) *size(H{1},4)]));
end

for k = 1:length(H2)
    hFeat2 = vertcat(hFeat2, reshape(H2{k}, [1, (size(H2{k},1) .^ 3) *size(H2{1},4)]));
end

% load 'hFeat.mat'
% load 'hFeat2.mat'

features = horzcat(hFeat, hFeat2);

labels = vertcat(ones(length(H) / 2,1), zeros(length(H2) / 2,1));

% features = cat(1, codFeat, nonCodFeat);
% labels = cat(1, codLabels, nonCodLabels);

k = 10;

cvFolds = crossvalind('Kfold', labels, k);
cp = classperf(labels);
cp2 = classperf(labels);

options = optimset('Display','iter', 'MaxIter', 100000);

for j = 1:k                                  %# for each fold
    testIdx = (cvFolds == j);                %# get indices of test instances
    trainIdx = ~testIdx;                     %# get indices training instances

    % train an SVM model over training instances
      svmModel = svmtrain(features(trainIdx, :), labels(trainIdx), ...
                   'Autoscale',true, 'Showplot',true, 'Method','SMO', ...
                   'BoxConstraint', 10, 'Kernel_Function','mlp', ...
                   'options', options);

    % test using test instances
     pred = svmclassify(svmModel, features(testIdx, :), 'Showplot',true);

    %# evaluate and update performance object
    cp = classperf(cp, pred, testIdx);
end

%# get accuracy
cp.CorrectRate

%# get confusion matrix
%# columns:actual, rows:predicted, last-row: unclassified instances
cp.CountingMatrix

return