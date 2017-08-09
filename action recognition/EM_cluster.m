% File: EM_cluster.m
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [P loglikelihood ClassProb] = EM_cluster(poseData, G, InitialClassProb, maxIter)

% INPUTS
% poseData: N x 10 x 3 matrix, where N is number of poses;
%   poseData(i,:,:) yields the 10x3 matrix for pose i.
% G: graph parameterization as explained in PA8
% InitialClassProb: N x K, initial allocation of the N poses to the K
%   classes. InitialClassProb(i,j) is the probability that example i belongs
%   to class j
% maxIter: max number of iterations to run EM

% OUTPUTS
% P: structure holding the learned parameters as described in the PA
% loglikelihood: #(iterations run) x 1 vector of loglikelihoods stored for
%   each iteration
% ClassProb: N x K, conditional class probability of the N examples to the
%   K classes in the final iteration. ClassProb(i,j) is the probability that
%   example i belongs to class j

% Initialize variables
N = size(poseData, 1);
numParts = size(poseData, 2);
numVars = size(poseData, 3);
K = size(InitialClassProb, 2);

ClassProb = InitialClassProb;

loglikelihood = zeros(maxIter,1);



% EM algorithm
for iter=1:maxIter
  
  % M-STEP to estimate parameters for Gaussians
  %
  % Fill in P.c with the estimates for prior class probabilities
  % Fill in P.clg for each body part and each class
  % Make sure to choose the right parameterization based on G(i,1)
  %
  % Hint: This part should be similar to your work from PA8
  
  P.c = zeros(1,K);
  P.clg = repmat(struct('mu_y', [], 'sigma_y', [], 'mu_x', [], 'sigma_x', [], ...
                      'mu_angle', [], 'sigma_angle', [], 'theta', []), 1, numParts);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  for k = 1:K
      W = ClassProb(:, k);
      P.c(k) = sum(ClassProb(:, k)) / N;
      for j = 1:numParts
          if (G(j, 1) == 0)
              [P.clg(j).mu_y(k), P.clg(j).sigma_y(k)] = FitG(poseData(:, j, 1), W);
              [P.clg(j).mu_x(k), P.clg(j).sigma_x(k)] = FitG(poseData(:, j, 2), W);
              [P.clg(j).mu_angle(k), P.clg(j).sigma_angle(k)] = FitG(poseData(:, j, 3), W);
          else
              pa = G(j, 2);
              U = reshape(poseData(:, pa, :), N, 3);
              
              [Beta1, P.clg(j).sigma_y(k)] = FitLG(poseData(:, j, 1), U, W);
              P.clg(j).theta(k, 1) = Beta1(4);
              P.clg(j).theta(k, 2:4) = Beta1(1:3);
              
              [Beta2, P.clg(j).sigma_x(k)] = FitLG(poseData(:, j, 2), U, W);
              P.clg(j).theta(k, 5) = Beta2(4);
              P.clg(j).theta(k, 6:8) = Beta2(1:3);
              
              [Beta3, P.clg(j).sigma_angle(k)] = FitLG(poseData(:, j, 3), U, W);
              P.clg(j).theta(k, 9) = Beta3(4);
              P.clg(j).theta(k, 10:12) = Beta3(1:3);
          end
      end
  end
              
              
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % E-STEP to re-estimate ClassProb using the new parameters
  %
  % Update ClassProb with the new conditional class probabilities.
  % Recall that ClassProb(i,j) is the probability that example i belongs to
  % class j.
  %
  % You should compute everything in log space, and only convert to
  % probability space at the end.
  %
  % Tip: To make things faster, try to reduce the number of calls to
  % lognormpdf, and inline the function (i.e., copy the lognormpdf code
  % into this file)
  %
  % Hint: You should use the logsumexp() function here to do
  % probability normalization in log space to avoid numerical issues
  
  ClassProb = zeros(N, K);
  ClassProb_temp = zeros(N, K);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  for i = 1:N
      for k = 1:K
          logProb = 0;
          for j = 1:numParts
              y = poseData(i, j, 1);
              x = poseData(i, j, 2);
              angle = poseData(i, j, 3);
              if (G(j, 1) == 0)
                  mu_y = P.clg(j).mu_y(k);
                  mu_x = P.clg(j).mu_x(k);
                  mu_angle = P.clg(j).mu_angle(k);
              else
                  pa = G(j, 2);
                  v = zeros(numVars, 1);
                  for l = 1:numVars
                      v(l) = poseData(i, pa, l);
                  end
                  mu_y = P.clg(j).theta(k, 1:4) * [1; v];
                  mu_x = P.clg(j).theta(k, 5:8) * [1; v];
                  mu_angle = P.clg(j).theta(k, 9:12) * [1; v];
              end
              sigma_y = P.clg(j).sigma_y(k);
              sigma_x = P.clg(j).sigma_x(k);
              sigma_angle = P.clg(j).sigma_angle(k);
              
              logProb = logProb + lognormpdf(y, mu_y, sigma_y) + ...
                  lognormpdf(x, mu_x, sigma_x) + ...
                  lognormpdf(angle, mu_angle, sigma_angle);
          end
          ClassProb_temp(i, k) = exp(log(P.c(k)) + logProb);
      end
      ClassProb(i, :) = ClassProb_temp(i, :) ./ sum(ClassProb_temp(i, :));
  end
  
              
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % Compute log likelihood of dataset for this iteration
  % Hint: You should use the logsumexp() function here
  loglikelihood(iter) = 0;
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  loglikelihood(iter) = sum(log(sum(ClassProb_temp, 2)));
 
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % Print out loglikelihood
  disp(sprintf('EM iteration %d: log likelihood: %f', ...
    iter, loglikelihood(iter)));
  if exist('OCTAVE_VERSION')
    fflush(stdout);
  end
  
  % Check for overfitting: when loglikelihood decreases
  if iter > 1
    if loglikelihood(iter) < loglikelihood(iter-1)
      break;
    end
  end
  
end

% Remove iterations if we exited early
loglikelihood = loglikelihood(1:iter);

end
