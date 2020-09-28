clear all; clc; close all
more off;
tic
X=load('data\JASAturbine1.dat');
x=X(:,1);  % speed samples
y=X(:,2);  % power samples


% f = @(B,x) B(1)*x.^4 + B(2)*x.^3 + B(3)*x.^2 + B(4)*x + B(5);
f = @(B,x) B(4) + (B(1)-B(4))./((B(6) + exp(-B(3)*(x - B(2)))).^B(5));

B0 = randn(1,6);

options = optimoptions('lsqcurvefit','MaxIter', 10000, 'MaxFunEvals', 10000, 'Display','off');

Bpol = polyfit(x,y,5)
Blsqcur = lsqcurvefit(f, B0, x, y, [], [], options)
% Bnlinf = nlinfit(x,y,f,B0)


xx = linspace(min(x),max(x), 200);
y_pred1 = polyval(Bpol, xx);
y_pred2 = polyval(Blsqcur, xx);
% y_pred3 = polyval(Bnlinf, xx);

figure;
plot(x, y, '*', 'color', [0.5 0.5 0.5], 'MarkerSize',2)
hold on
plot(xx, y_pred1, '-b', 'linewidth', 4);
plot(xx, y_pred2, '-r', 'linewidth', 3);
% plot(xx, y_pred3, '-g', 'linewidth', 2);
legend('DATA', 'POLYFIT', 'LSQCURVEFIT', 'NLINFIT')
xlabel('wind speed [m/s]')
ylabel('generated power [KWatts]')