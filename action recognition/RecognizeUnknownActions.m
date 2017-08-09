% You should put all your code for recognizing unknown actions in this file.
% Describe the method you used in YourMethod.txt.
% Don't forget to call SavePrediction() at the end with your predicted labels to save them for submission, then submit using submit.m



function RecognizeUnknownActions(datasetTrain, datasetTest, G, maxIter, kMeansIter)


Q = length(datasetTrain);

nRows = [];
dataPoses = [];
for q = 1:Q
    dataPoses = [dataPoses; reshape(datasetTrain(q).poseData, size(datasetTrain(q).poseData, 1), ...
        (size(datasetTrain(q).poseData, 2) * size(datasetTrain(q).poseData, 3)))];
    nRows = [nRows, size(datasetTrain(q).poseData, 1)];
end

dataPoses = (dataPoses - (mean(dataPoses, 2) * ones(1, size(dataPoses, 2)))) ./ (std(dataPoses, 0, 2) * ones(1, size(dataPoses, 2)));
nRowsCum = cumsum(nRows);


JCollection = [];
classCollection = [];
for iter = 1:kMeansIter
    initialRows = [];
    for q = 1:Q
        permu = randperm(nRows(q));
        initialRows = [initialRows, permu(1)];
    end
    
    for q = 2:Q;
        initialRows(q) = initialRows(q) + nRowsCum(q - 1);
    end
    
    mu_s = dataPoses(initialRows, :);
    
    J = 0;
    diff = 1;
    while (diff > 1e-6)
        class = zeros(1, size(dataPoses, 1));
        for i = 1:size(dataPoses, 1)
            D = zeros(1, Q);
            for q = 1:Q
                for j = 1:size(dataPoses, 2)
                    D(q) = D(q) + (dataPoses(i, j) - mu_s(q, j))^2;
                end
            end
            [minVal, minIndex] = min(D);
            class(i) = minIndex;
        end

        for q = 1:Q
            newRowsIndex = find(class == q);
            mu_s(q, :) = mean(dataPoses(newRowsIndex, :));
        end
    
        JPrev = J;
        J = 0;
        for i = 1:size(dataPoses, 1)
            J = J + (1 / size(dataPoses, 2)) * (dataPoses(i, :) - mu_s(class(i), :)) * (dataPoses(i, :) - mu_s(class(i), :))';
        end
    
        diff = abs(J - JPrev);
    end
    
    JCollection = [JCollection, J];
    classCollection = [classCollection; class];
end

[smlst, smlstidx] = min(JCollection);
cls = classCollection(smlstidx, :);
finalMu = zeros(Q, size(dataPoses, 2));
for q = 1:Q
    finalRowIndex = find(cls == q);
    finalMu(q, :) = mean(dataPoses(finalRowIndex, :));
    datasetTrain(q).InitialClassProbUpdated = datasetTrain(q).InitialClassProb;
end


for q = 1:Q
    if (q == 1)
        Sigma{q} = cov(dataPoses(1:nRows(q), :));
        Sigma{q} = Sigma{q} + 0.0000001 * eye(size(Sigma{q}, 1));
        dataP{q} = dataPoses(1:nRows(q), :);
    else
        Sigma{q} = cov(dataPoses((nRowsCum(q - 1) + 1):nRowsCum(q), :));
        Sigma{q} = Sigma{q} + 0.0000001 * eye(size(Sigma{q}, 1));
        dataP{q} = dataPoses((nRowsCum(q - 1) + 1):nRowsCum(q), :);
    end
end

for q = 1:Q
    for r = 1:Q
        datasetTrain(q).InitialClassProbUpdated(:, r) = mvnpdf(dataP{q}, finalMu(r, :), Sigma{r});
    end
    datasetTrain(q).InitialClassProbUpdated = ...
        datasetTrain(q).InitialClassProbUpdated ./ (sum(datasetTrain(q).InitialClassProbUpdated, 2) * ones(1, Q));
end


datasetTrain(q).InitialPairProbUpdated = zeros(size(datasetTrain(q).InitialPairProb));
for q = 1:Q
    L = length(datasetTrain(q).actionData);
    for g = 1:L
        lengthSing = length(datasetTrain(q).actionData(g).marg_ind);
        for e = 1:(lengthSing-1)
            t1 = datasetTrain(q).actionData(g).marg_ind(e);
            t2 = datasetTrain(q).actionData(g).pair_ind(e);
            assign = IndexToAssignment(1:(Q^2), [Q, Q]);
            for as = 1:size(assign, 1)
                datasetTrain(q).InitialPairProbUpdated(t2, as) = ...
                    datasetTrain(q).InitialClassProbUpdated(t1, assign(as, 1)) * datasetTrain(q).InitialClassProbUpdated(t1 + 1, assign(as, 2));
            end
        end
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            

numParts = size(datasetTrain(1).poseData, 2);
for i = 1:length(datasetTrain)
    [output(i).P output(i).loglikelihood output(i).ClassProb output(i).PairProb] =  ...
        EM_HMM(datasetTrain(i).actionData, datasetTrain(i).poseData, G, ...
        datasetTrain(i).InitialClassProbUpdated, datasetTrain(i).InitialPairProbUpdated, ...
        maxIter);
end



accuracy = 0;
predicted_labels = [];

for a = 1:length(datasetTest.actionData)
    rowNumber = datasetTest.actionData(a).marg_ind;
    N = length(datasetTest.actionData(a).marg_ind);
    loglikelihood_vec = zeros(1, length(output));
    poseData = datasetTest.poseData(rowNumber, :, :);
    for b = 1:length(output)
        K = size(output(b).ClassProb, 2);
        P = output(b).P;
        logEmissionProb = zeros(N,K);
        for i = 1:N
            for k = 1:K
                for j = 1:numParts
                    y = poseData(i, j, 1);
                    x = poseData(i, j, 2);
                    angle = poseData(i, j, 3);
                    sigma_y = P.clg(j).sigma_y(k);
                    sigma_x = P.clg(j).sigma_x(k);
                    sigma_angle = P.clg(j).sigma_angle(k);
                    if (G(j, 1) == 0)
                        mu_y = P.clg(j).mu_y(k);
                        mu_x = P.clg(j).mu_x(k);
                        mu_angle = P.clg(j).mu_angle(k);
                    else
                        pa = G(j, 2);
                        y_pa = poseData(i, pa, 1);
                        x_pa = poseData(i, pa, 2);
                        angle_pa = poseData(i, pa, 3);
                        v = [1, y_pa, x_pa, angle_pa];
                        mu_y = P.clg(j).theta(k, 1:4) * v';
                        mu_x = P.clg(j).theta(k, 5:8) * v';
                        mu_angle = P.clg(j).theta(k, 9:12) * v';
                    end
                    logEmissionProb(i, k) = logEmissionProb(i, k) + ...
                        lognormpdf(y, mu_y, sigma_y) + ...
                        lognormpdf(x, mu_x, sigma_x) + ...
                        lognormpdf(angle, mu_angle, sigma_angle);
                end
            end
        end
        
        rows = datasetTest.actionData(a).marg_ind;
        numSingletons = length(rows);
        pairRows = datasetTest.actionData(a).pair_ind;
        numPairs = length(pairRows);
        numFactors = 1 + numSingletons + numPairs;
        F = repmat(struct('var', [], 'card', [], 'val', []), 1, numFactors);
      
        F(1).var = [1];
        F(1).card = length(P.c);
        F(1).val = log(P.c);
      
        for m = 1:numSingletons
            F(m + 1).var = [ m ];
            F(m + 1).card = size(logEmissionProb, 2);
            F(m + 1).val = logEmissionProb(m, :);
        end
      
        for h = 1:numPairs
            F(h + numSingletons + 1).var = [h, h + 1];
            F(h + numSingletons + 1).card = size(P.transMatrix);
            assignment = IndexToAssignment(1:prod(F(h + numSingletons + 1).card), F(h + numSingletons + 1).card);
            for s = 1:size(assignment, 1)
                F(h + numSingletons + 1).val(s) = log(P.transMatrix(assignment(s, 1), assignment(s, 2)));
            end
        end
      
        [M, PCalibrated] = ComputeExactMarginalsHMM(F);
        loglikelihood_vec(b) = loglikelihood_vec(b) + logsumexp(PCalibrated.cliqueList(1).val);
    end
    [C, maxIdx] = max(loglikelihood_vec);
    predicted_labels = [predicted_labels; maxIdx];
end

        
   
SavePredictions(predicted_labels);

end
