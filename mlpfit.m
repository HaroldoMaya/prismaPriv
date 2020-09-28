function [w M yhat errors SSE R2]=mlpfit(x,y,Q)
%
% neural network curve fitting for power curve estimation
% using the Multilayer perceptron (MLP) neural network model.
%
% INPUTS
% ======
%
%  x: vector with input observations (regressors)
%  y: vector with output observations (same dimension as x)
%  Q: number of hidden neurons
%
% OUTPUTS
% =======
%
%  M: Input-to-hidden-layer weight matrix (randomly built)
%  w: estimated output weight vector (via standard OLS)
%  yhat: predicted output values
%  errors: prediction errors (residuals)
%  SSE: sum-of-squared errors
%  R2: coefficient of determination (R2 = 1 - SSE/(N*var(y)))
%
%  Author: Guilherme A. Barreto
%  Date: January 25th, 2018
%
close all;

x=x(:); y=y(:);  % input,output data always as column vectors

N=length(x);  % number of input-output pairs (xi,yi)

M=rand(Q,2);  % Initialize the input-to-hidden-layer weight matrix
W=rand(Q+1,1); % Initialize the hidden-to-output-layer weight matrix


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Compute predictions, errors and figures of merit   %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

yhat=Z'*w;  % predictions (in-sample)
errors=y-yhat;  % errors
SSE=sum(errors.^2);  % sum-of-squared errors
R2=1-SSE/(N*var(y,1));  % Coefficient of determination (the closer to 1, the better)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Plot power curve, histogram and boxplot of the errors  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Power curve plotting
xmin=min(x); xmax=max(x);
incr=0.1;  % increments for curve plotting purposes
xgrid=xmin:incr:xmax;  % x interval for curve plotting
xgrid=xgrid(:);

Xgrid=[ones(1,length(xgrid)); xgrid'];  % Add bias (row of 1's)
Ugrid=M*Xgrid;  % Compute activations (net values)
aux=exp(-Ugrid);  % Auxiliary term
Zgrid=1./(1+aux);  % Apply logistic function 
ygrid=Zgrid'*w; % Linear function (output layer)

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



