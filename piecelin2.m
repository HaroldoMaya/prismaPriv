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
x_min=min(x);
x_max=max(x);
incr=(x_max-x_min)/k;  % Length of each sub-interval
INT=x_min:incr:x_max;

w=[];
for i=2:k+1,
   Regions{i-1}=[INT(i-1) INT(i)]; % Find speed values within i-th region
end

