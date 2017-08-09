function [P loglikelihood] = LearnCPDsGivenGraph(dataset, G, labels)
%
% Inputs:
% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha)
% G: graph parameterization as explained in PA description
% labels: N x 2 true class labels for the examples. labels(i,j)=1 if the 
%         the ith example belongs to class j and 0 elsewhere        
%
% Outputs:
% P: struct array parameters (explained in PA description)
% loglikelihood: log-likelihood of the data (scalar)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset, 1);
numParts = size(dataset, 2);
K = size(labels, 2);

loglikelihood = 0;
P.c = zeros(1,K);
P.clg = repmat(struct('mu_y', [], 'sigma_y', [], 'mu_x', [], 'sigma_x', [], ...
                      'mu_angle', [], 'sigma_angle', [], 'theta', []), 1, numParts);

% estimate parameters
% fill in P.c, MLE for class probabilities
% fill in P.clg for each body part and each class
% choose the right parameterization based on G(i,1)
% compute the likelihood - you may want to use ComputeLogLikelihood.m
% you just implemented.
%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



if (size(G, 3) == 1)
    G1 = zeros(size(G, 1), size(G, 2), K);
    for k = 1:K
        G1(:, :, k) = G;
    end
    G = G1;
end

for k = 1:K
    c = find(labels(:, k) == 1);
    P.c(k) = length(c) / N;
    for j = 1:numParts
        if (G(j, 1, k) == 0)
            [P.clg(j).mu_y(k), P.clg(j).sigma_y(k)] = FitGaussianParameters(dataset(c, j, 1));
            [P.clg(j).mu_x(k), P.clg(j).sigma_x(k)] = FitGaussianParameters(dataset(c, j, 2));
            [P.clg(j).mu_angle(k), P.clg(j).sigma_angle(k)] = FitGaussianParameters(dataset(c, j, 3));
        else
            par = G(j, 2, k);
            U = reshape(dataset(c, par, :), length(c), 3);
            
            [Beta1, P.clg(j).sigma_y(k)] = FitLinearGaussianParameters(dataset(c, j, 1), U);
            P.clg(j).theta(k, 2:4) = Beta1(1:3);
            P.clg(j).theta(k, 1) = Beta1(4);
            
            [Beta2, P.clg(j).sigma_x(k)] = FitLinearGaussianParameters(dataset(c, j, 2), U);
            P.clg(j).theta(k, 6:8) = Beta2(1:3);
            P.clg(j).theta(k, 5) = Beta2(4);
            
            [Beta3, P.clg(j).sigma_angle(k)] = FitLinearGaussianParameters(dataset(c, j, 3), U);
            P.clg(j).theta(k, 10:12) = Beta3(1:3);
            P.clg(j).theta(k, 9) = Beta3(4);
        end
    end
end

loglikelihood = ComputeLogLikelihood(P, G, dataset);


fprintf('log likelihood: %f\n', loglikelihood);


end
