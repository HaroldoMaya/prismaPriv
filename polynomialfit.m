function [w yhat errors SSE R2]=polynomialfit(x,y,k)
%
% polynomial curve fitting for power curve estimation
% with coefficients computed via ordinary least-squares (OLS) method.
%
%  y(x)=w0+w1*x+w2*x^2+w3*x^3+...+wp*x^k
%
% INPUTS
% ======
%
%  x: vector with input observations (regressors)
%  y: vector with output observations (same dimension as x)
%  k: order of the polynomial
%
% OUTPUTS
% =======
%
%  w: estimated coefficients (via standard OLS)
%  yhat: predicted output values
%  errors: prediction errors (residuals)
%  SSE: sum-of-squared errors
%  R2: coefficient of determination (R2=1 - SSE/var(y))
%
%  Author: Guilherme A. Barreto
%  Date: January 25th, 2018
%
close all;

x=x(:); y=y(:);  % input,output data always as column vectors

N=length(x);  % number of input-output pairs (xi,yi)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% STEP 1: Create Vandermode Matrix (X) and output vector (y)  %%%%
%%%%         and estimate the coefficients via OLS method        %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STEP 1.1 - Generate vandermonde matrix
expon=cumsum(ones(1,k));
for n=1:N,
    X(n,:)=[1 x(n).^expon];
end
    
% STEP 1.2 -Parameter estimation (OLS method)
%w=inv(X'*X)*X'*y;   % Textbook equations (not recommended)
%w=X\y;              % Estimation using QR decomposition
w=pinv(X)*y;        % Estimation using SVD technique

%% OBS: Steps 1.1 and 1.2 can be fully replaced by the command 
%% polyfit available in Matlab and Octave
%w=polyfit(x,y,k);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% STEP 2: Compute predictions, errors and figures of merit   %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
yhat=X*w;  % predictions (in-sample)
errors=y-yhat;  % errors
SSE=sum(errors.^2);  % sum-of-squared errors
R2=1-SSE/(N*var(y,1));  % Coefficient of determination (the closer to 1, the better)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% STEP 3: Plot power curve, histogram and boxplot of the errors  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Power curve plotting
xmin=min(x); xmax=max(x);
incr=0.1;  % increments for curve plotting purposes
xgrid=xmin:incr:xmax;  % x interval for curve plotting
xgrid=xgrid(:);

for n=1:length(xgrid),
    Xgrid(n,:)=[1 xgrid(n).^expon];
end

ygrid=Xgrid*w;  % compute outputs for each value in xgrid

figure;
plot(x,y,'ro',xgrid,ygrid,'b-'); grid
xlabel('wind speed [m/s]')
ylabel('generated power [KWatts]')

% Error histogram
figure;
%histfit(errors);
hist(errors,20)
xlabel('prediction errors')
title('Histogram - prediction errors (residuals)')

% Boxplot of prediction errors
figure;
boxplot(errors);
title('Boxplot - prediction errors (residuals)')


	
