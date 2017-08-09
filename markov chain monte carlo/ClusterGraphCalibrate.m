% CLUSTERGRAPHCALIBRATE Loopy belief propagation for cluster graph calibration.
%   P = CLUSTERGRAPHCALIBRATE(P, useSmart) calibrates a given cluster graph, G,
%   and set of of factors, F. The function returns the final potentials for
%   each cluster. 
%   The cluster graph data structure has the following fields:
%   - .clusterList: a list of the cluster beliefs in this graph. These entries
%                   have the following subfields:
%     - .var:  indices of variables in the specified cluster
%     - .card: cardinality of variables in the specified cluster
%     - .val:  the cluster's beliefs about these variables
%   - .edges: A cluster adjacency matrix where edges(i,j)=1 implies clusters i
%             and j share an edge.
%  
%   UseSmart is an indicator variable that tells us whether to use the Naive or Smart
%   implementation of GetNextClusters for our message ordering
%
%   See also FACTORPRODUCT, FACTORMARGINALIZATION
%
% Copyright (C) Daphne Koller, Stanford University, 2012

function [P MESSAGES] = ClusterGraphCalibrate(P,useSmartMP)

if(~exist('useSmartMP','var'))
  useSmartMP = 0;
end

N = length(P.clusterList);

MESSAGES = repmat(struct('var', [], 'card', [], 'val', []), N, N);
[edgeFromIndx, edgeToIndx] = find(P.edges ~= 0);

for m = 1:length(edgeFromIndx),
    i = edgeFromIndx(m);
    j = edgeToIndx(m);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % YOUR CODE HERE
    %
    %
    %
    % Set the initial message values
    % MESSAGES(i,j) should be set to the initial value for the
    % message from cluster i to cluster j
    %
    % The matlab/octave functions 'intersect' and 'find' may
    % be useful here (for making your code faster)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    [v, ia, ib] = intersect(P.clusterList(i).var, P.clusterList(j).var);
    
    MESSAGES(i, j).var = v;
    MESSAGES(i, j).card = P.clusterList(i).card(ia);
    MESSAGES(i, j).val = ones(1, MESSAGES(i, j).card);
   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end;



% perform loopy belief propagation
tic;
iteration = 0;

lastMESSAGES = MESSAGES;

deltaInit = 0;

delta1 = [];
delta2 = [];
delta3 = [];


while (1),
    iteration = iteration + 1;
    [i, j] = GetNextClusters(P, MESSAGES,lastMESSAGES, iteration, useSmartMP); 
    prevMessage = MESSAGES(i,j);
    

    delta1 = [delta1, 0];
    delta2 = [delta2, 0];
    delta3 = [delta3, 0];

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % YOUR CODE HERE
    % We have already selected a message to pass, \delta_ij.
    % Compute the message from clique i to clique j and put it
    % in MESSAGES(i,j)
    % Finally, normalize the message to prevent overflow
    %
    % The function 'setdiff' may be useful to help you
    % obtain some speedup in this function
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
     neighbors = find(P.edges(:, i));
     neighborsNotJ = setdiff(neighbors, j);    
     incoming = struct('var', [], 'card', [], 'val', []);
     for k = 1:length(neighborsNotJ)
         incoming = FactorProduct(incoming, MESSAGES(neighborsNotJ(k), i));
     end
     
     message = FactorProduct(incoming, P.clusterList(i));
         
     s = setdiff(P.clusterList(i).var, P.clusterList(j).var);
     MESSAGES(i, j) = FactorMarginalization(message, s);
     MESSAGES(i, j).val = MESSAGES(i, j).val ./ sum(MESSAGES(i, j).val);
     
     
     if (i == 19 && j == 3)
         delta1(iteration) = MessageDelta(MESSAGES(i, j), prevMessage);
     else
         if (iteration == 1)
             delta1(iteration) = 1;
         else
             delta1(iteration) = delta1(iteration - 1);
         end
     end
     
     
     if (i == 15 && j == 40)
         delta2(iteration) = MessageDelta(MESSAGES(i, j), prevMessage);
     else
         if (iteration == 1)
             delta2(iteration) = 1;
         else
             delta2(iteration) = delta2(iteration - 1);
         end
     end
     
     if (i == 17 && j == 2)
         delta3(iteration) = MessageDelta(MESSAGES(i, j), prevMessage);
     else
         if (iteration == 1)
             delta3(iteration) = 1;
         else
             delta3(iteration) = delta3(iteration - 1);
         end
     end     
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if(useSmartMP==1)
      lastMESSAGES(i,j)=prevMessage;
    end
    
    % Check for convergence every m iterations
    if mod(iteration, length(edgeFromIndx)) == 0
        if (CheckConvergence(MESSAGES, lastMESSAGES))
            break;
        end
        disp(['LBP Messages Passed: ', int2str(iteration), '...']);
        if(useSmartMP~=1)
          lastMESSAGES=MESSAGES;
        end
    end
    
    
end;
toc;
disp(['Total number of messages passed: ', num2str(iteration)]);


% Compute final potentials and place them in P
for m = 1:length(edgeFromIndx),
    j = edgeFromIndx(m);
    i = edgeToIndx(m);
    P.clusterList(i) = FactorProduct(P.clusterList(i), MESSAGES(j, i));
end


% Get the max difference between the marginal entries of 2 messages -------
function delta = MessageDelta(Mes1, Mes2)
delta = max(abs(Mes1.val - Mes2.val));
return;


