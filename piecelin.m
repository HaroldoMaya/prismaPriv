%function [w yhat errors SSE R2]=piecelin(x,y,k)
%
% local non-overlapping linear fit for power curve estimation
% with coefficients computed via ordinary least-squares (OLS) method.
%
%  y_i(x)=w0_i+w1_i*x, where i denotes the i-th interval
%
% INPUTS
% ======
%
%  x: vector with input observations (regressors)
%  y: vector with output observations (same dimension as x)
%  k: number of non-overlapping regions to segment the wind speed range
%
% OUTPUTS
% =======
%
%  w: estimated coefficients via OLS (2 x k matrix: 1st row w0_i, 2nd row w1_i )
%  yhat: predicted output values
%  errors: prediction errors (residuals)
%  SSE: sum-of-squared errors
%  R2: coefficient of determination (R2=1 - SSE/var(y))
%
%  Author: Guilherme A. Barreto
%  Date: January 11, 2018
%
close all;

clear; clc;

X=load('turbine1.dat');
x=X(:,1);
y=X(:,2);

k=3;

x=x(:); y=y(:);  % input,output data always as column vectors

N=length(x);  % number of input-output pairs (xi,yi)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% STEP 1: Determine the boundaries of the non-overlapping intervals %%%
%%%%         and estimate the coefficients via OLS method              %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x_min=min(x); x_max=max(x);
incr=(x_max-x_min)/k;  % Length of each sub-interval
INT=x_min:incr:x_max;

w=[];
yhat=[];
for i=1:k,
   I=find(x>INT(i) & x<=INT(i+1)); % Find speed values within i-th region
   Xi=x(I); Yi=y(I); % Get (speed, power) values within i-th region
   XXi=[ones(length(Xi),1) Xi];  % Add columns of 1's (bias term)
   Bi=pinv(XXi)*Yi; % Compute parameters for line of the i-th region
   yhat(I) = XXi*Bi;  % Compute the corresponding preditions 
   w=[w Bi];  % Store parameters of the lines for the k regions
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% STEP 2: Compute errors and figures of merit   %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for n=1:N,
  % Find the region to which x(n) belongs to
  for j=1:k,
    if( (x(n) > INT(j)) & (x(n) <= INT(j+1)) ),
         i=j;
    end
  end
  Bi=w(:,i);  % Get the corresponding parameters of the line
  X=[1; x(n)];  % Mount the input vector
  yhat(n)=Bi'*X; % Compute the prediction for x(n)
end

yhat=yhat(:);
errors=y-yhat;  % errors
SSE=sum(errors.^2);  % sum-of-squared errors
R2=1-SSE/(N*var(y,1));  % Coefficient of determination (the closer to 1, the better)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% STEP 3: Plot power curve, histogram and boxplot of the errors  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Power curve plotting
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

	
