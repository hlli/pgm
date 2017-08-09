function [F, vars] = ConvertFeaturesToFactorList(features, numHiddenStates, theta)
    n = length(features);
    vars = {};
    vars{1} = features(1).var;
    len = length(vars);
    for i = 2:n
        Exist = 0;
        for j = 1:len
            if (features(i).var == vars{j})
                Exist = 1;
                break
            end
        end
        if (Exist == 0)
            len = len + 1;
            vars{len} = features(i).var;
        end
    end
    
    N = length(vars);
    F = repmat(struct('var', [], 'card', [], 'val', []), 1, N);
    
    for i = 1:N
        F(i).var = vars{i};
        F(i).card = numHiddenStates * ones(size(vars{i}));
        F(i).val = zeros(1, prod(F(i).card));
        for j = 1:n
            if all(features(j).var == F(i).var)
                idx = AssignmentToIndex(features(j).assignment, F(i).card);
                F(i).val(idx) = F(i).val(idx) + theta(features(j).paramIdx);
            end
        end
        F(i).val = exp(F(i).val);
    end
    
end

            


