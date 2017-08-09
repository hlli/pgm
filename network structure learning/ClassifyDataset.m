function accuracy = ClassifyDataset(dataset, labels, P, G)
% returns the accuracy of the model P and graph G on the dataset 
%
% Inputs:
% dataset: N x 10 x 3, N test instances represented by 10 parts
% labels:  N x 2 true class labels for the instances.
%          labels(i,j)=1 if the ith instance belongs to class j 
% P: struct array model parameters (explained in PA description)
% G: graph structure and parameterization (explained in PA description) 
%
% Outputs:
% accuracy: fraction of correctly classified instances (scalar)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset, 1);
K = size(labels, 2);
numParts = size(dataset, 2);
accuracy = 0.0;

if (size(G, 3) == 1)
    G1 = zeros(size(G, 1), size(G, 2), K);
    for k = 1:K
        G1(:, :, k) = G;
    end
    G = G1;
end

predLabels = zeros(size(labels));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numCorrect = 0;
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
     [maximum, index] = max(logProb);
     predLabels(i, index) = 1;
     k = find(labels(i, :) == 1);
     if (k == index)
         numCorrect = numCorrect + 1;
     end
end

accuracy = numCorrect / N;

fprintf('Accuracy: %.2f\n', accuracy);