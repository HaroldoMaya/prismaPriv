function [B SSE FIT]=logisticFit(x,y,alpha,base,kindFit)
%
% Logistic curve fitting for power curve estimation with coefficients
% computed via ordinary differential evolution (DE) method.
%
% For k = 3 (default) we will have 3 variables (B1,B2,B3) and the function 
% will be defined by:
%  y(x) = B1/{1 + exp[-B2*(x - B3)]}
%
% For k = 4 we will have 4 variables (B1,B2,B3,B4) and the function will be
% defined by:
%  y(x) = B4 + (B1 - B4)/{1 + exp[-B2*(x - B3)]}
%
% For k = 5 we will have 5 variables (B1,B2,B3,B4,B5) and the function 
% will be defined by:
%  y(x) = B4 + (B1 - B4)/{{1 + exp[-B2*(x - B3)]}^B5}     with B5 > 0
%
% And for k = 6 we will have 6 variables (B1,B2,B3,B4,B5,B6) and the function 
% will be defined by:
%  y(x) = B4 + (B1 - B4)/{{B6 + exp[-B2*(x - B3)]}^B5}     with B5 > 0
% INPUTS
% ======
%
%  x: vector with input observations (regressors).
%  y: vector with output observations (same dimension as x).
%  k: Kind of logistic expression (3, 4, 5 or 6 parameters). 3 as default.
%  noNegativePot: Penalty for negative power. True as default.
%
% OUTPUTS
% =======
%
%  w: estimated coefficients (via ordinary least-squares)
%  yhat: predicted output values
%  errors: prediction errors (residuals)
%  SSE: sum-of-squared errors
%  FIT: regression evaluation criterion
%
%  Author: Haroldo C. Maya
%  Date: August 25th, 2019
%

x=x(:); y=y(:);  % input,output data always as column vectors
N=length(x);  % number of input-output pairs (xi,yi)

%
%% ATTENTION!! Values for change are here.
gen = 10000;       % Number of generations
C1 = 0.7;           % Initial mutation factor [0 ~~ oo]
C2 = 0.6;           % Final mutation factor [0 ~~ oo]
FacCross = 0.9;     % Crossing Factor
stopCrit = 400;     % If the sse stays stable for "stopCrit" generations, stop it.
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialization of variables
dim = sum(alpha);            % Vector dimension.
index = find(alpha);
size_pop = 5*dim;  % Population size (It is recommended a minimum of ten times the number of dimensions)

% Initialize population vectors
V_pop = rand(size_pop, dim)-0.5;	% Initialization of the initial population as a random matrix.

% Initialize DE vectors
mutVec = zeros(size_pop, 2);	% Mutation vector
croVec = zeros(size_pop, dim);	% Crossing vector
ssePop = zeros(size_pop, 1); 	% SSE of the population
sseOff = zeros(size_pop, 1);  	% SSE of offspring
fitPop = zeros(size_pop, 1);   	% Fitness of the population
fitOff = zeros(size_pop, 1);   	% Fitness of offspring

sseBest = zeros(gen + 1, 1);    % Best SSE of generation.
fitBest = zeros(gen + 1, 1);    % Best fitness of generation.


f = @(v,B) B(4) + (B(1)-B(4))./((B(6) + exp(-B(3)*(v - B(2)))).^B(5));

%%%%%%%%%%%%%%%%%%

%% Initial analysis
for i = 1 : size_pop
    B = base;
    B(index) = B(index) + V_pop(i, :);
    y_hat = f(x,B);
    if (B(5) <= 0)
       y_hat = inf;
    end
    [fitPop(i), ssePop(i)] = fitValue(y, y_hat, dim, B, kindFit, 0, max(y));
    
        %% This part is generated to save time in the first iteration of Differential Evolution
    mutVec(i, :) = randperm(size_pop, 2);	% Selection of the indices of two distinct and random elements used to generate the donor vectors.
    croVec(i, ceil(dim*rand)) = 1;       	% Selection of one of the variables of each element of the trial vector to be obligatorily part of the crossover.
end

if strcmp(kindFit, 'r2')||strcmp(kindFit, 'r2a')
    [fitBest(1), iBest] = max(fitPop); 	% Best fitness of the initial population
else
    [fitBest(1), iBest] = min(fitPop); 	% Best fitness of the initial population
end
sseBest(1) = ssePop(iBest);         % SSE of the best fitness of the initial population
vBest = V_pop(iBest, :);            % Best vector of the initial population

%% Differential evolution
for t = 2 : gen

    C = C2 + (C1 - C2)*(gen - t)/gen; % Dynamic mutation factor.

    %% Generating offspring
    % Mutation: generate donor vectors ui = V_best + C*(Va - Vb)
    u = repmat(vBest, size_pop, 1) + C*(V_pop(mutVec(1:size_pop, 1), :) - V_pop(mutVec(1:size_pop, 2), :));
    
    % Generate the set of progeny vectors
    m_V_don = ((rand(size_pop, dim) <= FacCross) + croVec) ~= 0;    % Mask for elements of the trial vector that will participate in the crossing.
    m_V_pop = 1 - m_V_don;                                          % Mask for elements of the current population vector that will participate in the crossing.
    V_off = V_pop.*m_V_pop + u.*m_V_don;                          % Offspring
    croVec = zeros(size_pop, dim);     
    


    %% To analyze offspring
    for i = 1 : size_pop
        B = base;
        B(index) = B(index) + V_off(i, :);
        y_hat = f(x,B);
        if (B(5) <= 0)
           y_hat = inf;
        end
        %% Calculate Attributes
        [fitOff(i), sseOff(i)] = fitValue(y, y_hat, dim, B, kindFit, 0, max(y));

        %% This part is generated to save time in the first iteration of Differential Evolution
        mutVec(i, :) = randperm(size_pop, 2);	% Selection of the indices of two distinct and random elements used to generate the donor vectors.
        croVec(i, ceil(dim*rand)) = 1;       	% Selection of one of the variables of each element of the trial vector to be obligatorily part of the crossover.
    end  
    
    %% Greedy selection
    idx = fitOff < fitPop;          % Mask of offspring elements better than current generation
    V_pop(idx, :) = V_off(idx, :);	% Update population to the next generation.
    fitPop(idx) = fitOff(idx);    	% Update fitness to the next generation.
    ssePop(idx) = sseOff(idx);    	% Update SSE to the next generation.
    
    
    %% Extraction and presentation of data
    if strcmp(kindFit, 'r2')||strcmp(kindFit, 'r2a')
        [fitBest(t), iBest] = max(fitPop); 	% Best fitness.
    else
        [fitBest(t), iBest] = min(fitPop); 	% Best fitness.
    end
    sseBest(t) = ssePop(iBest);         % SSE of the best fitness.
    vBest = V_pop(iBest, :);            % Best vector.

    
    
    %% stopping criterion
    if ((t > stopCrit)&&(round(fitBest(t)*10000) == round(fitBest(t - stopCrit)*10000)))
        break
    end
end

B = base;
B(index) = B(index) + vBest;

if B(2) > 0
    sinal2 = '-';
else
    sinal2 = '+';
end
if B(3) > 0
    sinal1 = '-';
else
    sinal1 = '';
end
if B(4) >= 0
    sinal3 = '-';
else
    sinal3 = '+';
end

st = sprintf('%.4g + (%.4g %c %.4g)/{{%.4g + exp[%c%.4g*(x %c %.4g)]}^%.4g}', B(4), B(1), sinal3, abs(B(4)), B(6), sinal1, abs(B(3)), sinal2, abs(B(2)),B(5));

fprintf(' ### %d Parameters:  %s ###\n', dim, num2str(alpha));
fprintf(' Best fitness(%s): %1.4f\n', kindFit, fitBest(t));
fprintf(' Best SSE: %1.4f\n', sseBest(t));
fprintf(' %s\n\n',st)

SSE = sseBest(t);
FIT = fitBest(t);

end






function [fit, sse] = fitValue(y_data, y_pred, NParam, Param, kindFit, limInf, limSup)
    N = numel(y_data);
    err = y_data - y_pred;
    sse = err'*err;    % Sum-of-squared errors.
    switch(kindFit)
        case 'sse' % Sum-of-squared errors.
            fit = sse; 
        case 'nmse' % Normalized Mean Square Error.
            fit = sse/((y_data - mean(y_data))'*(y_data - mean(y_data))) + 0.00001*Param*Param'; % 
        case 'aic' % Akaike information criterion (AIC).
            fit = N*log(sse) + 2*(NParam + 1); 
        case 'bic' % Bayesian information criterion (BIC).
            fit = N*log(sse) + (NParam + 1)*log(N); 
        case 'fpe' % Final prediction error criterion (FPE).
            fit = N*log(sse) + popN*log((N + NParam)/(N-NParam)); 
        case 'r2' % R-Squared (R2).
            fit = 1 - sse/(sum((mean(y_data)-y_data).^2)); 
        case 'r2a' % R-Squared adjusted (R2A).
            fit = 1 - ((N-1)/(N - (NParam+1)))*sse/(sum((mean(y_data)-y_data).^2)); 
        otherwise
            fit = sse; % Sum-of-squared errors.
    end
    if (limInf>-Inf)
        if (sum(y_pred < limInf) > 0)
            fit = 2*fit;  % If there is power below limInf, penalize it.
        end
    end
    
    if (limSup<Inf)
        if (sum(y_pred > limSup) > 0)
            fit = 2*fit;  % If there is power below limInf, penalize it.
        end
    end
end
