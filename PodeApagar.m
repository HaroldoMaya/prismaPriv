clear all; close all
more off;

X=load('data\JASAturbine2.dat');
% X=load('data\JASAturbine5.dat');
x=X(:,1);  % speed samples
y=X(:,2);  % power samples

[x, i] = sort(x); % Ordena os dados de x em ordem crescente(?)
y = y(i);         % Ordena os dados de y em fun��o do x

%% Par�metros
% B(1) � o valor m�ximo de y.
B(1) = max(y); 

% B(2) � a velocidade do vento no ponto de inflex�o.
% Para chegar nesse valor tivemos que achar o valor de x associado ao y que
% mais se aproxima do valor (max(y) - min(y))/2.
% Foi testado fazer (max(x) - min(x))/2 e m�dia de x e n�o surtiu bons 
% resultados mantendo os outros par�metros fixos como definido. 
[~, iIp] = min(abs(y - (max(y) - min(y))/2)); 
B(2) = x(iIp); 

% B(3) � o slope of curve. 
% Ela � aproximadamente igual a 4s/Pr. Onde Pr � a pot�ncia nominal e s �
% a inclina��o no ponto de inflex�o. Para chegar a tal valor, calculamos a 
% m�dia dos valores de y em uma faixa de tamanho ls, a uma dist�ncia +ds e
% a uma dist�ncia -ds e calculamos a inclina��o entre esses dois pontos.
ds = 0.5; ls = 0.1;
B(3) = 4/B(1)*(...
      mean(y((x>=(B(2) + (ds - ls/2))) & (x<=(B(2) + (ds + ls/2))))) ...
    - mean(y((x<=(B(2) - (ds - ls/2))) & (x>=(B(2) - (ds + ls/2))))));
% B(4) � o valor m�nimo de y.
% Foi testado 0 e n�o surtiu bons resultados mantendo os outros par�metros 
% fixos como definido. 
B(4) = min(y);

% B(5) e B(6) tem valores pr�ximos a 1.
B(5) = 1; B(6) = 1;
%%
f = @(v,B)B(4) + (B(1)-B(4))./((B(6) + exp(-B(3)*(v - B(2)))).^B(5));
    
y_hat = f(x,B);

err = y - y_hat;
sse = err'*err
    
figure;
plot(x, y, '*', 'color', [0.5 0.5 0.5], 'MarkerSize',2)
hold on
plot(x, y_hat, 'k-', 'linewidth', 3);
xlabel('wind speed [m/s]')
ylabel('generated power [KWatts]')