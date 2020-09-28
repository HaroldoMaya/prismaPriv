clear all; close all;
% inicio = datestr(now)
%% Aquisição e pré-processamento de dados.
dados = load('data\turbine1.dat');  % Carregar dados.

x = dados(:, 1);	% Valores das ordenadas dos dados.
y = dados(:, 2);	% Valores das abscissas dos dados.


f = @(v,B) B(4) + (B(1)-B(4))./((B(6) + exp(-B(3)*(v - B(2)))).^B(5));

% base(1) = max(y);
% interv = 0.25;
% p = [];
% p = [p; min(x), mean(y(x <= interv))]; 
% for i=(min(x)+interv):interv:floor(max(x))
%     p = [p; i, mean(y((x >= p(end,1)) & (x <= p(end,1)+2*interv)))];
% end
% [~,indx] = min(abs(p(:,2) - max(y)/2));
% base(2) = p(indx,1);
% base(3) = 4/max(y)*1;
% base(4) = min(y);
% base(5) = 1;
% base(6) = 1;

base(1) = 2;
base(2) = 0;
base(3) = 0.5*4/(2*base(1));
base(4) = 0;
base(5) = 1
base(6) = 1

    figure;
    hold on
for i=1:1
    % xmin=min(x); xmax=max(x);
    xmin=base(2)-5; xmax=base(2)+5;
    incr=0.1;  % increments for curve plotting purposes
    xgrid=xmin:0.1:xmax;  % x interval for curve plotting
    xgrid=xgrid(:);
%     base(6) = i
    ygrid = f(xgrid,base);


    % plot(x, y, '*', 'color', [0.5 0.5 0.5], 'MarkerSize',2)
    % hold on
    plot(xgrid, ygrid, 'k-', 'linewidth', 3);
%     xlim([-0.5 0.5])
%     ylim([-0.5 0.5])

end
    xlabel('wind speed [m/s]')
    ylabel('generated power [KWatts]')
