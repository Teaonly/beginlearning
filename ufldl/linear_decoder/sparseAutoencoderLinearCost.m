function [cost,grad] = sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, ...
                                                   lambda, sparsityParam, beta, data)
% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Copy sparseAutoencoderCost in sparseAutoencoderCost.m from your
%   earlier exercise onto this file, renaming the function to
%   sparseAutoencoderLinearCost, and changing the autoencoder to use a
%   linear decoder.
% -------------------- YOUR CODE HERE --------------------

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);
m = size(data, 2);

z2 = W1 * data + repmat(b1, [1, m]);
a2 = sigmoid(z2);
z3 = W2 * a2 + repmat(b2, [1, m]);
a3 = z3;

rhohats = mean(a2,2);
rho = sparsityParam;
KLsum = sum(rho * log(rho ./ rhohats) + (1-rho) * log((1-rho) ./ (1-rhohats)));

squares = (a3 - data).^2;
squared_err_J = (1/2) * (1/m) * sum(squares(:));
weight_decay_J = (lambda/2) * (sum(W1(:).^2) + sum(W2(:).^2));
sparsity_J = beta * KLsum;

cost = squared_err_J + weight_decay_J + sparsity_J;

delta3 = -(data - a3);
beta_term = beta * (- rho ./ rhohats + (1-rho) ./ (1-rhohats));
delta2 = ((W2' * delta3) + repmat(beta_term, [1,m]) ) .* a2 .* (1-a2);

W2grad = (1/m) * delta3 * a2' + lambda * W2;
b2grad = (1/m) * sum(delta3, 2);
W1grad = (1/m) * delta2 * data' + lambda * W1;
b1grad = (1/m) * sum(delta2, 2);

%-------------------------------------------------------------------
% Convert weights and bias gradients to a compressed form
% This step will concatenate and flatten all your gradients to a vector
% which can be used in the optimization method.
grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

% Sigmoid function
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
