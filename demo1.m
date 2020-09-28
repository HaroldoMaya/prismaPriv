%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% DEMO 1: Estimation of the power curve using polinomial regression  %%%%%%%%
%%%%                                                                   %%%%%%%%
%%%% Author: Guilherme A. Barreto                                      %%%%%%%%
%%%% Date: January 8th, 2018                                           %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; close all

X=load('turbine1.dat');
x=X(:,1);  % speed samples
y=X(:,2);  % power samples

k=5;  % Chosen polynomial order

[w yhat errors SSE R2]=polynomialfit(x,y,k)
