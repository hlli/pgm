%CLIQUETREECALIBRATE Performs sum-product or max-product algorithm for 
%clique tree calibration.

%   P = CLIQUETREECALIBRATE(P, isMax) calibrates a given clique tree, P 
%   according to the value of isMax flag. If isMax is 1, it uses max-sum
%   message passing, otherwise uses sum-product. This function 
%   returns the clique tree where the .val for each clique in .cliqueList
%   is set to the final calibrated potentials.
%
% Copyright (C) Daphne Koller, Stanford University, 2012

function P = CliqueTreeCalibrate(P, isMax)


% Number of cliques in the tree.
N = length(P.cliqueList);

% Setting up the messages that will be passed.
% MESSAGES(i,j) represents the message going from clique i to clique j. 
MESSAGES = repmat(struct('var', [], 'card', [], 'val', []), N, N);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% We have split the coding part for this function in two chunks with
% specific comments. This will make implementation much easier.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% YOUR CODE HERE
% While there are ready cliques to pass messages between, keep passing
% messages. Use GetNextCliques to find cliques to pass messages between.
% Once you have clique i that is ready to send message to clique
% j, compute the message and put it in MESSAGES(i,j).
% Remember that you only need an upward pass and a downward pass.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%
% Now the clique tree has been calibrated. 
% Compute the final potentials for the cliques and place them in P.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 if (isMax == 1)
     for n = 1:N
         P.cliqueList(n).val = log(P.cliqueList(n).val);
     end
 end

 [i, j] = GetNextCliques(P, MESSAGES);
 while all([i, j])    
     neighbors = find(P.edges(:, i));
     neighborsNotJ = setdiff(neighbors, j);    
     MESSAGES(i, j) = P.cliqueList(i);
     if ~isempty(neighborsNotJ)
         for k = 1:length(neighborsNotJ)
             if (isMax == 1)
                 MESSAGES(i, j) = FactorSum(MESSAGES(i, j), MESSAGES(neighborsNotJ(k), i));
             else
                 MESSAGES(i, j) = FactorProduct(MESSAGES(i, j), MESSAGES(neighborsNotJ(k), i));
             end
         end
     end
         
     if (isMax == 1)
         d = setdiff(P.cliqueList(i).var, P.cliqueList(j).var);
         MESSAGES(i, j) = FactorMaxMarginalization(MESSAGES(i, j), d);
     else
         s = intersect(P.cliqueList(i).var, P.cliqueList(j).var);
         MESSAGES(i, j) = ComputeMarginal(s, MESSAGES(i, j), []);
         MESSAGES(i, j).val = MESSAGES(i, j).val ./ sum(MESSAGES(i, j).val); 
     end 
     [i, j] = GetNextCliques(P, MESSAGES);
 end
     
 for  l = 1:N
     neighbors = find(P.edges(:, l));
     for h = 1:length(neighbors)
         if (isMax == 1)
             P.cliqueList(l) = FactorSum(P.cliqueList(l), MESSAGES(neighbors(h), l));
         else
             P.cliqueList(l) = FactorProduct(P.cliqueList(l), MESSAGES(neighbors(h), l));
         end
     end
 end
 
 end
