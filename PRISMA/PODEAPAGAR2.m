clear all; close all;
% inicio = datestr(now)
%% Aquisição e pré-processamento de dados.
dados = load('data\turbine2.dat');  % Carregar dados.

x = dados(:, 1);	% Valores das ordenadas dos dados.
y = dados(:, 2);	% Valores das abscissas dos dados.


f = @(v,B) B(4) + (B(1)-B(4))./((1 + exp(-B(3)*(v - B(2)))).^B(5));

base(1) = max(y);
interv = 0.25;
p = [];
p = [p; min(x), mean(y(x <= interv))]; 
for i=(min(x)+interv):interv:floor(max(x))
    p = [p; i, mean(y((x >= p(end,1)) & (x <= p(end,1)+2*interv)))];
end
[~,indx] = min(abs(p(:,2) - max(y)/2));
base(2) = p(indx,1);
base(3) = 4/max(y)*((p(indx+1,2) - p(indx-1,2))/(p(indx+1,1) - p(indx-1,1)));
base(4) = min(y);
base(5) = 1;
% base(6) = 1;

str = '@(B,x) B(4) + (B(1)-B(4))./((1 + exp(-B(3)*(x - B(2)))).^B(5))'

options = optimoptions('lsqcurvefit','MaxIter', 10000, 'MaxFunEvals', 10000, 'Display','off');
pFit = lsqcurvefit(str2func(str) , base, x, y, [], [], options)

y_pred = f(x,pFit);
N = numel(y);
err = y - y_pred;
NParam = 5;
sse = err'*err    % Sum-of-squared errors.
bic = N*log(sse) + (NParam + 1)*log(N)

xmin=min(x); xmax=max(x);
incr=0.1;  % increments for curve plotting purposes
xgrid=xmin:incr:xmax;  % x interval for curve plotting
xgrid=xgrid(:);
ygrid = f(xgrid,pFit);


figure;
plot(x, y, '*', 'color', [0.5 0.5 0.5], 'MarkerSize',2)
hold on
plot(xgrid, ygrid, 'k-', 'linewidth', 3);
xlabel('wind speed [m/s]')
ylabel('generated power [KWatts]')
