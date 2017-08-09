function loglikelihood = ComputeLogLikelihood(P, G, dataset)
% returns the (natural) log-likelihood of data given the model and graph structure
%
% Inputs:
% P: struct array parameters (explained in PA description)
% G: graph structure and parameterization (explained in PA description)
%
%    NOTICE that G could be either 10x2 (same graph shared by all classes)
%    or 10x2x2 (each class has its own graph). your code should compute
%    the log-likelihood using the right graph.
%
% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha)
% 
% Output:
% loglikelihood: log-likelihood of the data (scalar)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset, 1); % number of examples
K = length(P.c); % number of classes
numParts = size(dataset, 2);

% You should compute the log likelihood of data as in eq. (12) and (13)
% in the PA description
% Hint: Use lognormpdf instead of log(normpdf) to prevent underflow.
%       You may use log(sum(exp(logProb))) to do addition in the original
%       space, sum(Prob).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if (size(G, 3) == 1)
    G1 = zeros(size(G, 1), size(G, 2), K);
    for k = 1:K
        G1(:, :, k) = G;
    end
    G = G1;
end

loglikelihood1 = zeros(1, N);
for i = 1:N
    
    logProb = zeros(1, K);
    for k = 1:K
        
        logPoseProb = zeros(1, numParts);
        for j = 1:numParts
            
            y = dataset(i, j, 1);
            x = dataset(i, j, 2);
            angle = dataset(i, j, 3);
            
            if (G(j, 1, k) == 0)
                mu1 = P.clg(j).mu_y(k);
                mu2 = P.clg(j).mu_x(k);
                mu3 = P.clg(j).mu_angle(k);
                sigma1 = P.clg(j).sigma_y(k);
                sigma2 = P.clg(j).sigma_x(k);
                sigma3 = P.clg(j).sigma_angle(k);
            else
                par = G(j, 2, k);
                yp = dataset(i, par, 1);
                xp = dataset(i, par, 2);
                anglep = dataset(i, par, 3);
                mu1 = [1, yp, xp, anglep] * (P.clg(j).theta(k, 1:4))';
                mu2 = [1, yp, xp, anglep] * (P.clg(j).theta(k, 5:8))';
                mu3 = [1, yp, xp, anglep] * (P.clg(j).theta(k, 9:12))';
                sigma1 = P.clg(j).sigma_y(k);
                sigma2 = P.clg(j).sigma_x(k);
                sigma3 = P.clg(j).sigma_angle(k);
            end
            lognorm1 = lognormpdf(y, mu1, sigma1);
            lognorm2 = lognormpdf(x, mu2, sigma2);
            lognorm3 = lognormpdf(angle, mu3, sigma3);
            logPoseProb(j) = lognorm1 + lognorm2 +lognorm3;
        end
        logProb(k) = log(P.c(k)) + sum(logPoseProb);
        
    end
    loglikelihood1(i) = log(sum(exp(logProb)));
end

loglikelihood = sum(loglikelihood1);

end
                
