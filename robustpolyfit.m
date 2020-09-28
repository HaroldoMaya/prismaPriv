function [w yhat errors SSE R2]=robustpolyfit(x,y,k)
%
% Robust polynomial curve fitting for power curve estimation
% with coefficients computed via M-estimation method.
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
%  w: estimated coefficients (via ordinary least-squares)
%  yhat: predicted output values
%  errors: prediction errors (residuals)
%  SSE: sum-of-squared errors
%  R2: coefficient of determination (R2=1 - SSE/var(y))
%
%  Author: Guilherme A. Barreto
%  Date: November 18, 2016
%
close all;

x=x(:); y=y(:);  % input,output data always as column vectors

N=length(x);  % number of input-output pairs (xi,yi)

% Generate vandermonde matrix
expon=cumsum(ones(1,k));
for n=1:N,
    Xa(n,:)=[1 x(n).^expon];
end
    
% Parameter estimation 
w=robustfit(Xa(:,2:end),y);

% Compute predictions, errors and figures of merit
yhat=Xa*w;  % predictions (in-sample)
errors=y-yhat;  % errors
SSE=sum(errors.^2);  %sum-of-squared errors
R2=1-SSE/(N*var(y,1));  % Coefficient of determination (the closer to 1, the better)

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
histfit(errors);
xlabel('prediction errors')
title('Histogram - prediction errors (residuals)')

% Boxplot of prediction errors
figure;
boxplot(errors);
title('Boxplot - prediction errors (residuals)')


	
