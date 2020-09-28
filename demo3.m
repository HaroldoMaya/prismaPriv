%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% DEMO 3: Estimation of the power curve using the Extreme Learning  %%%%%%%%
%%%%        Machine (ELM) neural network model                         %%%%%%%%
%%%%                                                                   %%%%%%%%
%%%% Author: Guilherme A. Barreto                                      %%%%%%%%
%%%% Date: January 25th, 2018                                          %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; close all

X=load('turbine2.dat');
x=X(:,1);  % speed samples
y=X(:,2);  % power samples

Q=15;  % Number of hidden neurons

[W M yhat errors SSE R2]=elmfit(x,y,Q);
