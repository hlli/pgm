%COMPUTEEXACTMARGINALSBP Runs exact inference and returns the marginals
%over all the variables (if isMax == 0) or the max-marginals (if isMax == 1). 
%
%   M = COMPUTEEXACTMARGINALSBP(F, E, isMax) takes a list of factors F,
%   evidence E, and a flag isMax, runs exact inference and returns the
%   final marginals for the variables in the network. If isMax is 1, then
%   it runs exact MAP inference, otherwise exact inference (sum-prod).
%   It returns an array of size equal to the number of variables in the 
%   network where M(i) represents the ith variable and M(i).val represents 
%   the marginals of the ith variable. 
%
% Copyright (C) Daphne Koller, Stanford University, 2012


function M = ComputeExactMarginalsBP(F, E, isMax)

% initialization
% you should set it to the correct value in your code

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%
% Implement Exact and MAP Inference.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

P = CreateCliqueTree(F, E);
    
P = CliqueTreeCalibrate(P, isMax);
N = length(P.cliqueList);
M = repmat(struct('var', [], 'card', [], 'val', []), N, 1);
    
variables = P.cliqueList(1).var;
for i = 2:N
    variables = union(variables, P.cliqueList(i).var);
end
    
for j = 1:length(variables)
    for k = 1:N
        if ismember(variables(j), P.cliqueList(k).var)
            out = setdiff(P.cliqueList(k).var, variables(j));
            if (isMax == 1)
                M(j) = FactorMaxMarginalization(P.cliqueList(k), out);
            else
                M(j) = FactorMarginalization(P.cliqueList(k), out);
                M(j).val = M(j).val ./ sum(M(j).val);                    
                continue
            end
        end
    end
end
