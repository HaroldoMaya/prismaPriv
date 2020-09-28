function [vBest y_hat err SSE FIT]=logisticfit(x,y,k,noNegativePot,kindFit)
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
NIter = 100000;       % Number of iterations
Interval = 1;
%


%% Initialization of variables
if k == 3
    f = @(v,B) ...
        B(1)./(1 + exp(-B(2)*(v - B(3))));
elseif k == 4
    f = @(v,B) ...
        B(4) + (B(1)-B(4))./(1 + exp(-B(2)*(v - B(3))));
elseif k == 5
    f = @(v,B) ...
        B(4) + (B(1)-B(4))./((1 + exp(-B(2)*(v - B(3)))).^B(5));
elseif k == 6
    f = @(v,B)...
        B(4) + (B(1)-B(4))./((B(6) + exp(-B(2)*(v - B(3)))).^B(5));
end

s = rand(k, 1); % Initial values.
%% Attribute calculation
y_hat = f(x,s);
[fit, sse] = fitValue(y, y_hat, k, kindFit,noNegativePot);
%%%%%%%%%%%%%%%

%% Simulated annealing
for i = 2 : NIter
        
    T = NIter/i; % Tempareture
    
    s_n = s + Interval*(rand(k, 1) - 0.5); % Escolha de um vizinho aleatório no intevalo Interval.
    y_hat_n = f(x,s_n);
    
    if ((k > 4)&&(s(5) <= 0)) % Penalidades
       y_hat_n = inf;
    end
    %% Attribute calculation
    [fit_n, sse_n] = fitValue(y, y_hat_n, k, kindFit,noNegativePot);
    
    %% 
    
    %% Extraction and presentation of data
    if strcmp(kindFit, 'r2')||strcmp(kindFit, 'r2a')
        varEnergy = fit - fit_n; 	% Best fitness.
    else
        varEnergy = fit_n - fit;	% Best fitness.
    end
    
    
    varEnergy
    P = exp(-varEnergy/T)
%     pause
        
    if (P >= rand)
        s = s_n;
        fit = fit_n;
        sse = sse_n;
    end
    
    fitBest(i) = fit;
    
    B = s;
    if B(2) > 0
        sinal1 = '-';
    else
        sinal1 = '';
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
        st = sprintf('%.4g/{1 + exp[%c%.4g*(x %c %.4g)]}', B(1), sinal1, abs(B(2)), sinal2, abs(B(3)));
    elseif k == 4
        st = sprintf('%.4g + (%.4g %c %.4g)/{1 + exp[%c%.4g*(x %c %.4g)]}', B(4), B(1), sinal3, B(4), sinal1, abs(B(2)), sinal2, abs(B(3)));
    elseif k == 5
        st = sprintf('%.4g + (%.4g %c %.4g)/{{1 + exp[%c%.4g*(x %c %.4g)]}^%.4g}', B(4), B(1), sinal3, B(4), sinal1, abs(B(2)), sinal2, abs(B(3)),B(5));
    elseif k == 6
        st = sprintf('%.4g + (%.4g %c %.4g)/{{%.4g + exp[%c%.4g*(x %c %.4g)]}^%.4g}', B(4), B(1), sinal3, B(4), B(6), sinal1, abs(B(2)), sinal2, abs(B(3)),B(5));
    end
    
    clc
    fprintf('\n\n ### Generation %d ###\n\n', i);
    fprintf(' Best fitness(%s): %1.4f\n', kindFit, fit);
    fprintf(' Best SSE: %1.4f\n', sse);
    fprintf(' %s\n',st)
%     P
%     s'
%     pause
    
end

err = y - y_hat;  % errors
SSE = sse;
FIT = fit;
vBest = s;
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
