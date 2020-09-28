%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% DEMO 3: Estimation of the power curve using logistic expression.   %%%%%%%%
%%%% Author: Haroldo C. Maya                                           %%%%%%%%
%%%% Date: January 9th, 2018                                           %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; clc; close all
more off;
tic
X=load('data\JASAturbine1.dat');
x=X(:,1);  % speed samples
y=X(:,2);  % power samples

for i = 1:1
k=6;                    % Chosen parameters of logistic expression: [3, 4, 5 or 6].
kindFit = 'aic';       % Criterion to measure fitness: [aic, bic, fpe, r2 or r2a]
penBelow = 0;           % penalty for values below. Default = -inf.
penAbove = max(y);         % penalty for values above. Default = inf.
forceIniParam = true;   


[vBest yhat errors SSE FIT] = logisticfit1DE(x,y,k,kindFit,forceIniParam,penBelow,penAbove);

RMSD(i) = sqrt(SSE/numel(x));
end
toc
% mean(RMSD)
% min(RMSD)
% max(RMSD)
% var(RMSD)
