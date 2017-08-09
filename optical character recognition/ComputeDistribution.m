function D = ComputeDistribution(vars, cliqueList)
    N = length(vars);
    D = repmat(struct('var', [], 'card', [], 'val', []), 1, N);
    n = length(cliqueList);
    
    %for k = 1:n 
        %cliqueList(k).val = cliqueList(k).val / sum(cliqueList(k).val);
    %end
    
    for i = 1:N
        for j = 1:n
            if all(ismember(vars{i}, cliqueList(j).var))
                diff = setdiff(cliqueList(j).var, vars{i});
                D(i) = FactorMarginalization(cliqueList(j), diff);
                D(i).val = D(i).val / sum(D(i).val);
                break
            end
        end
    end
    
end