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
NNew = 100;            % Number of iterations
stopCrit = 200;       % If the sse stays stable for "stopCrit" generations, stop it.
Interval = 2;
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

  N = k;               % number of objective variables/problem dimension
  xmean = rand(N,1);    % objective variables initial point
  sigma = 0.01;          % coordinate wise standard deviation (step size)
  stopfitness = 1e-10;  % stop if fitness < stopfitness (minimization)
  stopeval = 1e3*N^2;   % stop after stopeval number of function evaluations
  
  % Strategy parameter setting: Selection  
  lambda = 4+floor(3*log(N));  % population size, offspring number
  mu = lambda/2;               % number of parents/points for recombination
  weights = log(mu+1/2)-log(1:mu)'; % muXone array for weighted recombination
  mu = floor(mu);        
  weights = weights/sum(weights);     % normalize recombination weights array
  mueff=sum(weights)^2/sum(weights.^2); % variance-effectiveness of sum w_i x_i

  % Strategy parameter setting: Adaptation
  cc = (4+mueff/N) / (N+4 + 2*mueff/N);  % time constant for cumulation for C
  cs = (mueff+2) / (N+mueff+5);  % t-const for cumulation for sigma control
  c1 = 2 / ((N+1.3)^2+mueff);    % learning rate for rank-one update of C
  cmu = min(1-c1, 2 * (mueff-2+1/mueff) / ((N+2)^2+mueff));  % and for rank-mu update
  damps = 1 + 2*max(0, sqrt((mueff-1)/(N+1))-1) + cs; % damping for sigma 
                                                      % usually close to 1
  % Initialize dynamic (internal) strategy parameters and constants
  pc = zeros(N,1); ps = zeros(N,1);   % evolution paths for C and sigma
  B = eye(N,N);                       % B defines the coordinate system
  D = ones(N,1);                      % diagonal D defines the scaling
  C = B * diag(D.^2) * B';            % covariance matrix C
  invsqrtC = B * diag(D.^-1) * B';    % C^-1/2 
  eigeneval = 0;                      % track update of B and D
  chiN=N^0.5*(1-1/(4*N)+1/(21*N^2));  % expectation of 
                                      %   ||N(0,I)|| == norm(randn(N,1)) 
  % -------------------- Generation Loop --------------------------------
  counteval = 0;  % the next 40 lines contain the 20 lines of interesting code 
  while counteval < stopeval
    
      % Generate and evaluate lambda offspring
      for j=1:lambda,
          arx(:,j) = xmean + sigma * B * (D .* randn(N,1)); % m + sig * Normal(0,C) 
          arx(:,j)
          pause
          y_hat = f(x,arx(:,j));
          [x_new_fit(j), x_new_sse(j)] = fitValue(y, y_hat, k, kindFit,noNegativePot);
          counteval = counteval+1;
      end
    
      % Sort by fitness and compute weighted mean into xmean
      [x_new_fit, arindex] = sort(x_new_fit); % minimization
      xold = xmean;
      xmean = arx(:,arindex(1:mu))*weights;   % recombination, new mean value
    
      % Cumulation: Update evolution paths
      ps = (1-cs)*ps ... 
            + sqrt(cs*(2-cs)*mueff) * invsqrtC * (xmean-xold) / sigma; 
      hsig = norm(ps)/sqrt(1-(1-cs)^(2*counteval/lambda))/chiN < 1.4 + 2/(N+1);
      pc = (1-cc)*pc ...
            + hsig * sqrt(cc*(2-cc)*mueff) * (xmean-xold) / sigma;

      % Adapt covariance matrix C
      artmp = (1/sigma) * (arx(:,arindex(1:mu))-repmat(xold,1,mu));
      C = (1-c1-cmu) * C ...                  % regard old matrix  
           + c1 * (pc*pc' ...                 % plus rank one update
                   + (1-hsig) * cc*(2-cc) * C) ... % minor correction if hsig==0
           + cmu * artmp * diag(weights) * artmp'; % plus rank mu update

      % Adapt step size sigma
      sigma = sigma * exp((cs/damps)*(norm(ps)/chiN - 1)); 
    
      % Decomposition of C into B*diag(D.^2)*B' (diagonalization)
      if counteval - eigeneval > lambda/(c1+cmu)/N/10  % to achieve O(N^2)
          eigeneval = counteval;
          C = triu(C) + triu(C,1)'; % enforce symmetry
          [B,D] = eig(C);           % eigen decomposition, B==normalized eigenvectors
          D = sqrt(diag(D));        % D is a vector of standard deviations now
          invsqrtC = B * diag(D.^-1) * B';
      end
    
    param = arx(:, arindex(1));
    if param(2) > 0
        sinal1 = '-';
    else
        sinal1 = '';
    end
    if param(3) > 0
        sinal2 = '-';
    else
        sinal2 = '+';
    end
    if(k > 3)
        if param(4) > 0
            sinal3 = '-';
        else
            sinal3 = '+';
        end
    end
    
    y_hat = f(x,param);
    if k == 3
        st = sprintf('%.4g/{1 + exp[%c%.4g*(x %c %.4g)]}', param(1), sinal1, abs(param(2)), sinal2, abs(param(3)));
    elseif k == 4
        st = sprintf('%.4g + (%.4g %c %.4g)/{1 + exp[%c%.4g*(x %c %.4g)]}', param(4), param(1), sinal3, param(4), sinal1, abs(param(2)), sinal2, abs(param(3)));
    elseif k == 5
        st = sprintf('%.4g + (%.4g %c %.4g)/{{1 + exp[%c%.4g*(x %c %.4g)]}^%.4g}', param(4), param(1), sinal3, param(4), sinal1, abs(param(2)), sinal2, abs(param(3)),param(5));
    elseif k == 6
        st = sprintf('%.4g + (%.4g %c %.4g)/{{%.4g + exp[%c%.4g*(x %c %.4g)]}^%.4g}', param(4), param(1), sinal3, param(4), param(6), sinal1, abs(param(2)), sinal2, abs(param(3)),param(5));
    end
    
    clc
    fprintf('\n\n ### Generation %d ###\n\n', counteval);
    fprintf(' paramest fitness(%s): %1.4f\n', kindFit, x_new_fit(1));
    fprintf(' paramest SSE: %1.4f\n', x_new_sse(1));
    fprintf(' %s\n',st)
%     P
%     s'
%     pause
      
      % Break, if fitness is good enough or condition exceeds 1e14, better termination methods are advisable 
      if x_new_fit(1) <= stopfitness || max(D) > 1e7 * min(D)
          break;
      end
      
  end % while, end generation loop

% ---------------------------------------------------------------  


err = y - y_hat;  % errors
SSE = x_new_sse(1);
FIT = x_new_fit(1);
vBest = arx(:, arindex(1));;
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
