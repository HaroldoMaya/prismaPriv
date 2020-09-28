function [vBest y_hat err SSE FIT]=logisticfit(x,y,k,noNegativePot,kindFit)
%
% Logistic curve fitting for power curve estimation with coefficients
% computed via ordinary differential evolution (DE) method.
%
% For k = 3 (default) we will have 3 variables (B1,B2,B3) and the function 
% will be defined by:
%  y(x) = B1/{1 + (x/B2)^(-B3)}
%
% For k = 4 we will have 4 variables (B1,B2,B3,B4) and the function will be
% defined by:
%  y(x) = B4 + (B1 - B4)/{1 + (x/B2)^(-B3)}
%
% For k = 5 we will have 5 variables (B1,B2,B3,B4,B5) and the function 
% will be defined by:
%  y(x) = B4 + (B1 - B4)/{{1 + (x/B2)^(-B3)}^B5}     with B5 > 0
%
% And for k = 6 we will have 6 variables (B1,B2,B3,B4,B5,B6) and the function 
% will be defined by:
%  y(x) = B4 + (B1 - B4)/{{B6 + (x/B2)^(-B3)}^B5}     with B5 > 0
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
if  nargin < 2      
    error('LOGISTICFIT requires at least three input arguments.');     
end 
if nargin < 3
    k = 3;
end
if (k < 3)||(k > 6)
    error('k most be between 3 and 6.');     
end 
if nargin < 4
    noNegativePot = true;
end
if nargin < 5
    kindFit = 'bic';
end

x=x(:); y=y(:);  % input,output data always as column vectors
N=length(x);  % number of input-output pairs (xi,yi)


%% ATTENTION!! Values for change are here.
gen = 10000;       % Number of generations
B1 = 0.8;           % Initial mutation factor [0 ~~ oo]
B2 = B1;            % Final mutation factor [0 ~~ oo]
FacCross = 0.8;     % Crossing Factor
stopCrit = 200;     % If the sse stays stable for "stopCrit" generations, stop it.
%


%% Initialization of variables
dim = k;            % Vector dimension.
size_pop = 10*dim;  % Population size (It is recommended a minimum of ten times the number of dimensions)

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

if k == 3
    f = @(v,B) ...
        B(1)./(1 + (v/B(2)).^(-B(3)));
elseif k == 4
    f = @(v,B) ...
        B(4) + (B(1)-B(4))./(1 + (v/B(2)).^(-B(3)));
elseif k == 5
    f = @(v,B) ...
        B(4) + (B(1)-B(4))./((1 + (v/B(2)).^(-B(3))).^B(5));
elseif k == 6
    f = @(v,B)...
        B(4) + (B(1)-B(4))./((B(6) + (v/B(2)).^(-B(3))).^B(5));
end
%%%%%%%%%%%%%%%%%%

%% Initial analysis
for i = 1 : size_pop
    B = V_pop(i, :);
    y_hat = f(x,B);
    
    %% Attribute calculation
    [fitPop(i), ssePop(i)] = fitValue(y, y_hat, k, kindFit,noNegativePot);
    
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
        
    B = B2 + (B1 - B2)*(gen - t)/gen; % Dynamic mutation factor.

    %% Generating offspring
    % Mutation: generate donor vectors ui = V_best + B*(Va - Vb)
    u = repmat(vBest, size_pop, 1) + B*(V_pop(mutVec(1:size_pop, 1), :) - V_pop(mutVec(1:size_pop, 2), :));
    
    % Generate the set of progeny vectors
    m_V_don = ((rand(size_pop, dim) <= FacCross) + croVec) ~= 0;    % Mask for elements of the trial vector that will participate in the crossing.
    m_V_pop = 1 - m_V_don;                                          % Mask for elements of the current population vector that will participate in the crossing.
    V_off = V_pop.*m_V_pop + u.*m_V_don;                          % Offspring
    croVec = zeros(size_pop, dim);                                  % Clear crossing vector

    %% To analyze offspring
    for i = 1 : size_pop
        B = V_off(i, :);
        y_hat = f(x,B);
        if (B(3) <= 0)
           y_hat = inf;
        end
        if ((k > 4)&&(B(5) <= 0))
           y_hat = inf;
        end
        
        %% Calculate Attributes
        [fitOff(i), sseOff(i)] = fitValue(y, y_hat, k, kindFit, noNegativePot);

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
    
    B = vBest;
    if B(2) > 0
        sinal1 = '';
    else
        sinal1 = '-';
    end
    if B(3) > 0
        sinal2 = '-';
    else
        sinal2 = '+';
    end
    if(k > 3)
        if B(4) > 0
            sinal3 = '-';
        else
            sinal3 = '+';
        end
    end
    
    y_hat = f(x,B);
    if k == 3
        st = sprintf('%.4g/{1 + (%cv/%.4g)^(%c%.4g)}', B(1), sinal1, abs(B(2)), sinal2, abs(B(3)));
    elseif k == 4
        st = sprintf('%.4g + (%.4g %c %.4g)/{1 + (%cv/%.4g)^(%c%.4g)}', B(4), B(1), sinal3, B(4), sinal1, abs(B(2)), sinal2, abs(B(3)));
    elseif k == 5
        st = sprintf('%.4g + (%.4g %c %.4g)/{{1 + (%cv/%.4g)^(%c%.4g)}^%.4g}', B(4), B(1), sinal3, B(4), sinal1, abs(B(2)), sinal2, abs(B(3)),B(5));
    elseif k == 6
        st = sprintf('%.4g + (%.4g %c %.4g)/{{%.4g + (%cv/%.4g)^(%c%.4g)}^%.4g}', B(4), B(1), sinal3, B(4), B(6), sinal1, abs(B(2)), sinal2, abs(B(3)),B(5));
    end
    
    clc
    fprintf('\n\n ### Generation %d ###\n\n', t);
    fprintf(' Best fitness(%s): %1.4f\n', kindFit, fitBest(t));
    fprintf(' Best SSE: %1.4f\n', sseBest(t));
    fprintf(' %s\n',st)
    
    
    %% stopping criterion
    if ((t > stopCrit)&&(round(fitBest(t)*10000) == round(fitBest(t - stopCrit)*10000)))
        break
    end
end

err = y - y_hat;  % errors
SSE = ssePop(end);
FIT = fitPop(end);

%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% STEP 3: Plot power curve, histogram and boxplot of the errors  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Power curve plotting
xmin=min(x); xmax=max(x);
incr=0.1;  % increments for curve plotting purposes
xgrid=xmin:incr:xmax;  % x interval for curve plotting
xgrid=xgrid(:);
ygrid = f(xgrid,vBest);

figure;
plot(x,y,'ro',xgrid,ygrid,'b-'); grid
xlabel('wind speed [m/s]')
ylabel('generated power [KWatts]')

% Error histogram
figure;
%histfit(errors);
hist(err, 20)
xlabel('prediction errors')
title('Histogram - prediction errors (residuals)')
% Boxplot of prediction errors
figure;
boxplot(err);
title('Boxplot - prediction errors (residuals)')
end










function [fit, sse] = fitValue(y_data, y_pred, NParam, kindFit, noNegativePot)
    N = numel(y_data);
    err = y_data - y_pred;
    sse = err'*err;    % Sum-of-squared errors.
    switch(kindFit)
        case 'sse'
            fit = sse; % Sum-of-squared errors.
        case 'aic'
            fit = N*log(sse) + 2*(NParam + 1); % Akaike information criterion (AIC).
        case 'bic'
            fit = N*log(sse) + (NParam + 1)*log(N); % Bayesian information criterion (BIC).
        case 'fpe'
            fit = N*log(sse) + popN*log((N + NParam)/(N-NParam)); % Final prediction error criterion (FPE).
        case 'r2'
            fit = 1 - sse/(sum((mean(y_data)-y_data).^2)); % R-Squared (R2).
        case 'r2a'
            fit = 1 - ((N-1)/(N - (NParam+1)))*sse/(sum((mean(y_data)-y_data).^2)); % R-Squared adjusted (R2A).
        otherwise
            fit = sse; % Sum-of-squared errors.
    end
    if ((noNegativePot)&&(sum(y_pred < 0) > 0))
        fit = 10*fit;  % If there is power below 0, penalize it.
    end
end
