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
gen = 10000;	% Number of generations
stopCrit = 400;     % If the sse stays stable for "stopCrit" generations, stop it.
w_ini = 0.9;      % Inicial weight. [ (c1+c2)/2 - 1 < w < 1  ]
w_fin = 0.9;    % Final weight. [ (c1+c2)/2 - 1 < w < 1  ]
cog = 0.30;      % Cognitive factor.
soc = 0.70;      % Social factor.
spdMax = 0.05;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialization of variables
dim = sum(alpha);            % Vector dimension.
index = find(alpha);
size_pop = 10*dim;  % Population size (It is recommended a minimum of ten times the number of dimensions)

% Initialize population vectors
position = rand(size_pop, dim)-0.5;	% Initialization of the initial population as a random matrix.



% Initialize PSO vectors
speed = zeros(size_pop, dim);             % Inicialização do vetor velocidade (inicialmente em repouso).
inertia = linspace(w_ini, w_fin, gen);   % Inicializar a componente inercial da partícula.

sse = zeros(size_pop, 1); 	% SSE of the population.
fit = zeros(size_pop, 1);   % Fitness of the population.

bestSSE = zeros(gen + 1, 1);	% Best SSE of generation.
bestFit = zeros(gen + 1, 1);    % Best fitness of generation.


f = @(v,B) B(4) + (B(1)-B(4))./((B(6) + exp(-B(3)*(v - B(2)))).^B(5));

%%%%%%%%%%%%%%%%%%

%% Initial analysis
for i = 1 : size_pop
    B = base;
    B(index) = B(index) + position(i, :);
    y_hat = f(x,B);
    if (B(5) <= 0)
       y_hat = inf;
    end
    [fit(i), sse(i)] = fitValue(y, y_hat, dim, B, kindFit, 0, max(y));
end
if strcmp(kindFit, 'r2')||strcmp(kindFit, 'r2a')
    [bestFit(1), iBest] = max(fit); 	% Best fitness of the initial population
else
    [bestFit(1), iBest] = min(fit); 	% Best fitness of the initial population
end

bestIndPos = position;
bestIndFit = fit;
bestIndSSE = sse;
bestGlobalPos = position(iBest, :);            % Best vector of the initial population
bestSSE(1) = sse(iBest);         % SSE of the best fitness of the initial population

%% Differential evolution
for t = 2 : gen
    %%%%%%%%%%%%%%%% Atualização da velocidade %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    spdInertial = inertia(t).*speed;                                                      % Memória da velocidade anterior.
    spdCog = cog*rand(size_pop, dim).*(bestIndPos - position);                                   % Componente cognitiva da velocidade.
    spdSocial = soc*rand(size_pop, dim).*(repmat(bestGlobalPos, size_pop, 1) - position);   % Componente social da velocidade.
    speed = spdInertial + spdCog + spdSocial;                                    % Atualização da velocidade.
    speed(speed > spdMax) = spdMax;                                             % Ajustar a velocidade em caso da velocidade ultrapassar a velocidade máxima absoluta.
    speed(speed < -spdMax) = -spdMax;                                           % Ajustar a velocidade em caso da velocidade ultrapassar a velocidade máxima absoluta.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    position = position + speed;   
    
    for i = 1 : size_pop
        B = base;
        B(index) = B(index) + position(i, :);
        y_hat = f(x,B);
        if (B(5) <= 0)
           y_hat = inf;
        end
        %% Performance calculation 
        [fit(i), sse(i)] = fitValue(y, y_hat, dim, B, kindFit, 0, max(y));
    end
    
    if strcmp(kindFit, 'r2')||strcmp(kindFit, 'r2a')
        aux = fit > bestIndFit;
        bestIndPos(aux, :) = position(aux, :);
        bestIndFit(aux) = fit(aux);
        bestIndSSE(aux) = sse(aux);
        [bestFit(t), iBest] = max(bestIndFit); % Best fitness of the initial population
    else
        aux = fit < bestIndFit;
        bestIndPos(aux, :) = position(aux, :);
        bestIndFit(aux) = fit(aux);
        bestIndSSE(aux) = sse(aux);
        [bestFit(t), iBest] = min(bestIndFit); % Best fitness of the initial population
    end
    bestGlobalPos = position(iBest, :);	% Best vector of the initial population
    bestSSE(t) = bestIndSSE(iBest);              	% SSE of the best fitness of the initial population

    

    
    
    %% stopping criterion
    if ((t > stopCrit)&&(round(bestFit(t)*10000) == round(bestFit(t - stopCrit)*10000)))
        break
    end
end

B = base;
B(index) = B(index) + bestGlobalPos;

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
fprintf(' Best fitness(%s): %1.4f\n', kindFit, bestFit(t));
fprintf(' Best SSE: %1.4f\n', bestSSE(t));
fprintf(' %s\n\n',st)

SSE = bestSSE(t);
FIT = bestFit(t);

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
