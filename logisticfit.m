function [vBest y_hat err SSE FIT]=logisticfit(x,y,k,noNegativePot,kindFit)
%
% Logistic curve fitting for power curve estimation with coefficients
% computed via ordinary differential evolution (DE) method.
%
% For k = 3 (default) we will have 3 variables (L, p, m) and the function 
% will be defined by:
%  y(x) = L/{1 + exp[-p*(x - m)]}
%
% For k = 4 we will have 4 variables (a, m, n, t) and the function will be
% defined by:
%  y(x) = a*{[1 + m*exp(-x/t)]/[1 + n*exp(-x/t)]}
%
% For k = 5 we will have 5 variables (a, b, c, d, g) and the function 
% will be defined by:
%  y(x) = d + (a - d)/{[1 + (x/c)^b]^g}     with c,g > 0
%
% And for k = 6 we will have 6 variables (a, b, c, d, g) and the function 
% will be defined by:
%  y(x) = d + (a - d)/{[1 + (x/c)^b]^g}     with c,g > 0
% INPUTS
% ======
%
%  x: vector with input observations (regressors).
%  y: vector with output observations (same dimension as x).
%  k: Kind of logistic expression (3, 4 or 5 parameters). 3 as default.
%  noNegativePot: Penalty for negative power. True as default.
%
% OUTPUTS
% =======
%
%  w: estimated coefficients (via ordinary least-squares)
%  yhat: predicted output values
%  errors: prediction errors (residuals)
%  SSE: sum-of-squared errors
%  R2: coefficient of determination (R2=1 - SSE/var(y))
%
%  Author: Haroldo C. Maya
%  Date: January 8th, 2017
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
gen = 100000;       % Number of generations
B1 = 0.9;           % Initial mutation factor [0 ~~ oo]
B2 = B1;            % Final mutation factor [0 ~~ oo]
FacCross = 0.8;     % Crossing Factor
stopCrit = 200;     % If the sse stays stable for "stopCrit" generations, stop it.
%


%% Initialization of variables
dim = k;            % Vector dimension.
size_pop = 30*dim;  % Population size (It is recommended a minimum of ten times the number of dimensions)

% Initialize population vectors
V_pop = rand(size_pop, dim)-0.5;	% Initialization of the initial population as a random matrix.
if k == 5
    V_pop(:,3) = abs(V_pop(:,3)) + eps;
    V_pop(:,5) = abs(V_pop(:,5)) + eps;
end

V_off = zeros(size_pop, dim);
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
%     f = @(L,p,m) L./(1 + exp(-p*(x - m)));
%     f = @(v,L,p,m) L./(1 + exp(4*p*(m-v)/L));
    f = @(v,L,p,m) m + L*v./sqrt(p-v.*v);
elseif k == 4
    f = @(v,a,m,n,tau) a*((1 + m*exp(-v/tau))./(1 + n*exp(-v/tau)));
elseif k == 5
    f = @(v,a,b,c,d,g) d + (a - d)./((1+(v./c).^b).^g);
elseif k == 6
    f = @(v,lAsymp,hAsymp,gRt,clstAsymp,v0,adj)...
        lAsymp + (hAsymp - lAsymp)./((adj + exp(-gRt*(v-v0))).^(1/clstAsymp));
end
%%%%%%%%%%%%%%%%%%

%% Initial analysis
for i = 1 : size_pop
    if k == 3
        L = V_pop(i, 1);
        p = V_pop(i, 2);
        m = V_pop(i, 3);
        y_hat = f(x,L,p,m);
    elseif k == 4
        a = V_pop(i, 1);
        m = V_pop(i, 2);
        n = V_pop(i, 3);
        tau = V_pop(i, 4);
        y_hat = f(x,a,m,n,tau);
    elseif k == 5
        a = V_pop(i, 1);
        b = V_pop(i, 2);
        c = V_pop(i, 3);
        d = V_pop(i, 4);
        g = V_pop(i, 5);
        if ((c <= 0)||(g <= 0))
            y_hat = inf;
        else
            y_hat = f(x,a,b,c,d,g);
        end
    elseif k == 6
        a = V_pop(i, 1);
        b = V_pop(i, 2);
        c = V_pop(i, 3);
        d = V_pop(i, 4);
        e = V_pop(i, 5);
        g = V_pop(i, 6);
        y_hat = f(x,a,b,c,d,e,g);
    end
    %% Attribute calculation
    err = y - y_hat;        % Errors
    ssePop(i) = err'*err;   % Sum-of-squared errors
    
    switch(kindFit)
        case 'sse'
            fitPop(i) = ssePop(i); % Sum-of-squared errors.
        case 'aic'
            fitPop(i) = N*log(ssePop(i)) + 2*(k+1); % Akaike information criterion (AIC).
        case 'bic'
            fitPop(i) = N*log(ssePop(i)) + (k+1)*log(N); % Bayesian information criterion (BIC).
        case 'fpe'
            fitPop(i) = N*log(ssePop(i)) + N*log((N+k)/(N-k)); % Final prediction error criterion (FPE).
        case 'r2'
            fitPop(i) = 1 - ssePop(i)/(sum((mean(y)-y).^2)); % R-Squared (R2).
        case 'r2a'
            fitPop(i) = 1 - ((N-1)/(N - (k+1)))*ssePop(i)/(sum((mean(y)-y).^2)); % R-Squared adjusted (R2A).
        otherwise
            fitPop(i) = N*log(ssePop(i)/N) + (k+1)*log(N); % Bayesian information criterion (BIC).
    end
    
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
        if k == 3
            L = V_off(i, 1);
            p = V_off(i, 2);
            m = V_off(i, 3);
            y_hat = f(x,L,p,m);
        elseif k == 4
            a = V_off(i, 1);
            m = V_off(i, 2);
            n = V_off(i, 3);
            tau = V_off(i, 4);
            y_hat = f(x,a,m,n,tau);
        elseif k == 5
            a = V_off(i, 1);
            b = V_off(i, 2);
            c = V_off(i, 3);
            d = V_off(i, 4);
            g = V_off(i, 5);
            if ((c <= 0)||(g <= 0))
                y_hat = inf;
            else
                y_hat = f(x,a,b,c,d,g);
            end
        elseif k == 6
            a = V_off(i, 1);
            b = V_off(i, 2);
            c = V_off(i, 3);
            d = V_off(i, 4);
            e = V_off(i, 5);
            g = V_off(i, 6);
            y_hat = f(x,a,b,c,d,e,g);
        end
        %% Calculate Attributes
        err = y - y_hat;            % Erros.
        sseOff(i) = sum(err.^2);    % Sum-of-squared errors.
        switch(kindFit)
            case 'sse'
                fitOff(i) = sseOff(i); % Sum-of-squared errors.
            case 'aic'
                fitOff(i) = N*log(sseOff(i)) + 2*(k+1); % Akaike information criterion (AIC).
            case 'bic'
                fitOff(i) = N*log(sseOff(i)) + (k+1)*log(N); % Bayesian information criterion (BIC).
            case 'fpe'
                fitOff(i) = N*log(sseOff(i)) + N*log((N+k)/(N-k)); % Final prediction error criterion (FPE).
            case 'r2'
                fitOff(i) = 1 - sseOff(i)/(sum((mean(y)-y).^2)); % R-Squared (R2).
            case 'r2a'
                fitOff(i) = 1 - ((N-1)/(N - (k+1)))*sseOff(i)/(sum((mean(y)-y).^2)); % R-Squared adjusted (R2A).
            otherwise
                fitOff(i) = N*log(sseOff(i)) + (k+1)*log(N); % Bayesian information criterion (BIC).
        end
        
        if ((noNegativePot)&&(sum(y_hat < 0) > 0))
            fitOff(i) = 10*fitOff(i);  % If there is power below 0, penalize it.
        end
    
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
    
    if k == 3
        L = vBest(1);
        p = vBest(2);
        m = vBest(3);
        if p >= 0
            sinal1 = '-';
        else
            sinal1 = '';
        end
        if m >= 0
            sinal2 = '-';
        else
            sinal2 = '+';
        end
        y_hat = f(x,L,p,m);
        st = sprintf('%.4f/{1 + exp[%c%.4f*(x %c %.4f)]}', L, sinal1, abs(k), sinal2, abs(m));
    elseif k == 4
        a = vBest(1);
        m = vBest(2);
        n = vBest(3);
        tau = vBest(4);
        if m >= 0
            sinal1 = '+';
        else
            sinal1 = '-';
        end
        if tau >= 0
            sinal2 = '-';
        else
            sinal2 = '';
        end
        if n >= 0
            sinal3 = '+';
        else
            sinal3 = '-';
        end
        y_hat = f(x,a,m,n,tau);
        st = sprintf('%.4f{[1 %c %.4fexp(%cx/%.4f)]/[1 %c %.4f*exp(%cx/%.4f)]}', a, sinal1, abs(m), sinal2, abs(tau), sinal3, abs(n), sinal2, abs(tau));
    elseif k == 5
        a = vBest(1);
        b = vBest(2);
        c = vBest(3);
        d = vBest(4);
        g = vBest(5);
        if d >= 0
            sinal1 = '';
            sinal3 = '-';
        else
            sinal1 = '-';
            sinal3 = '+';
        end
        if a >= 0
            sinal2 = '';
        else
            sinal2 = '-';
        end
        if b >= 0
            sinal4 = '';
        else
            sinal4 = '-';
        end
        y_hat = f(x,a,b,c,d,g);
        st = sprintf('%c%.4f + (%c%.4f %c %.4f)/[(1 + (x/%.4f)^{(%c %.4f)})^{%.4f}]', sinal1, abs(d), sinal2, abs(a), sinal3, abs(d), c, sinal4, abs(b), g);
    elseif k == 6
        a = vBest(1);
        b = vBest(2);
        c = vBest(3);
        d = vBest(4);
        e = vBest(5);
        g = vBest(6);
        if a >= 0
            sinal1 = '';
            sinal3 = '-';
        else
            sinal1 = '-';
            sinal3 = '+';
        end
        if b >= 0
            sinal2 = '';
        else
            sinal2 = '-';
        end
        if c >= 0
            sinal4 = '';
        else
            sinal4 = '-';
        end
        y_hat = f(x,a,b,c,d,e,g);
        st = sprintf('%.4f + (%c%.4f %c %.4f)/[(%.4f + (x/%.4f)^{(%c %.4f)})^{%.4f}]', sinal1, abs(a), sinal2, abs(b), sinal3, abs(a), g, sinal4, abs(b), g);
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

if k == 3
    ygrid = f(xgrid,L,p,m);
elseif k == 4
    ygrid = f(xgrid,a,m,n,tau);
elseif k == 5
    ygrid = f(xgrid,a,b,c,d,g);
end

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

