function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.
hvalue = exp(theta * data);
allSum = sum(hvalue);
allSum = repmat(allSum, numClasses, 1);
hvalue = hvalue ./ allSum;

cost = -1 * sum(sum(log(hvalue).*groundTruth)) / numCases + 0.5 * lambda * sum(sum(theta.^2));
thetagrad = -1 *  (groundTruth - hvalue) * data' ./ numCases + lambda * theta;


% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

