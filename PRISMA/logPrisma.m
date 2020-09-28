    %% Polinomial com o grau como vari�vel a ser otimizada.
clear all; close all; clc;
inicio = datestr(now)
%% Aquisi��o e pr�-processamento de dados.
dados = load('data\JASAturbine6.dat');  % Carregar dados.

x = dados(:, 1);	% Valores das ordenadas dos dados.
y = dados(:, 2);	% Valores das abscissas dos dados.

kindFit = 'bic';

%% Hiper par�metros
termos = 3;

% AG
tam_pop = 8;           % Tamanho da populacao
tam_cro = termos;       % Tamanho do cromossomo (no. de genes)
pc = 0.8;               % Probabilidade de cruzamento
pm = 0.1;              % Probabilidade de mutacao
geracoes = 1;          % Numero de geracoes
elitismo = 1;           % Quantidade de elites passadas para pr�xima gera��o.
%


%% Vari�veis
f = @(v,B)...
    B(4) + (B(1)-B(4))./((B(6) + exp(-B(3)*(v - B(2)))).^B(5));

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
base(4) = 0;
base(5) = 1;
base(6) = 1;

% AG
pop = GerarPop(tam_pop, tam_cro);
pop(1,:) = [0 0 0];
pop(2,:) = [0 0 1];
pop(3,:) = [0 1 0];
pop(4,:) = [0 1 1];
pop(5,:) = [1 0 0];
pop(6,:) = [1 0 1];
pop(7,:) = [1 1 0];
pop(8,:) = [1 1 1];
aptBest = zeros(geracoes+1, 1);     % Inicializar com 0s o vetor de Melhor Aptid�o.
sseBest = zeros(geracoes+1, 1);     % Inicializar com 0s o vetor de Melhor Aptid�o.
bBest = zeros(geracoes+1, 6);     % Inicializar com 0s o vetor de Melhor Aptid�o.
%

    

%% Initial analysis
[aptPop, ssePop, bPop] = logAptidao2(pop, y, x, base, kindFit);     % Calcular a aptid�o da popula��o atual e:
[aptBest(1), iAptBest] = min(aptPop);
sseBest(1) = ssePop(iAptBest);
bBest(1,:) = bPop(iAptBest,:);

%% Apresenta��o de dados
B = bBest(1,:);

if B(2) > 0
    sinal2 = '-';
else
    sinal2 = '+';
end
if B(3) > 0
    sinal1 = '-';
else
    sinal1 = '';
end
if B(4) >= 0
    sinal3 = '-';
else
    sinal3 = '+';
end

st = sprintf('%.4g + (%.4g %c %.4g)/{{%.4g + exp[%c%.4g*(x %c %.4g)]}^%.4g}', B(4), B(1), sinal3, abs(B(4)), B(6), sinal1, abs(B(3)), sinal2, abs(B(2)),B(5));

fprintf(' ### Generation %d ###\n\n', 0);
fprintf(' Alpha: %s\n', num2str(pop(iAptBest,:)));
fprintf(' Best fitness(%s): %1.4f\n', kindFit, aptBest(1));
fprintf(' Best SSE: %1.4f\n', sseBest(1));
fprintf(' %s\n\n\n\n\n',st)
%

%% Algoritimo Gen�tico
for t = 2:geracoes+1
    
        %% Sele��o
    S = Selecao(aptPop);                     % Selecionar pais com base na aptid�o atual.
    filhos = Cruzamento(pop, S, pc);    % Gera��o de filhos na etapa de cruzamento.
    filhos = Mutacao(filhos, pm);       % Muta��o de filhos.
    [aptSpa, sseSpa, bSpa] = logAptidao2(filhos, y, x, base, kindFit);   % Calculando a aptid�o para cada o caso.
    
    
    % Bloco do Elitismo:
    % a) Caso elitismo = 0, bloco ser� ignorado em tempo de execu��o;
    % b) Caso contr�rio, ir� ser selecionado a quantidade igual ao valor da
    %    vari�vel elitismo da popula��o atual para compor a pr�xima
    %    gera��o, substituindo os filhos gerados com menos aptid�o.
    if (elitismo)
        il = elitismo;
        [~, iPop] = sort(aptPop);
        [~, iFilho] = sort(aptSpa, 'descend');
        while(il)
            filhos(iFilho(il),:) = pop(iPop(il),:);
            bSpa(iFilho(il),:) = bPop(iPop(il),:);
            aptSpa(iFilho(il)) = aptPop(iPop(il));
            sseSpa(iFilho(il)) = ssePop(iPop(il));
            il = il - 1;
        end
    end
    %%%%%%%%%% Fim do Bloco do Elitismo %%%%%%%%%%
    pop = filhos;       % Atualiza��o da nova popula��o
    aptPop = aptSpa;       % Atualiza��o da nova popula��o
    ssePop = sseSpa;       % Atualiza��o da nova popula��o
    bPop = bSpa;       % Atualiza��o da nova popula��o
    [aptBest(t), iAptBest] = min(aptPop);
    sseBest(t) = ssePop(iAptBest);
    bBest(t,:) = bPop(iAptBest,:);

    %% Apresenta��o de dados
    B = bBest(t,:);

    if B(2) > 0
        sinal2 = '-';
    else
        sinal2 = '+';
    end
    if B(3) > 0
        sinal1 = '-';
    else
        sinal1 = '';
    end
    if B(4) >= 0
        sinal3 = '-';
    else
        sinal3 = '+';
    end

    st = sprintf('%.4g + (%.4g %c %.4g)/{{%.4g + exp[%c%.4g*(x %c %.4g)]}^%.4g}', B(4), B(1), sinal3, abs(B(4)), B(6), sinal1, abs(B(3)), sinal2, abs(B(2)),B(5));

    fprintf(' ### Generation %d ###\n\n', t-1);
    fprintf(' Alpha: %s\n', num2str(pop(iAptBest,:)));
    fprintf(' Best fitness(%s): %1.4f\n', kindFit, aptBest(t));
    fprintf(' Best SSE: %1.4f\n', sseBest(t));
    fprintf(' %s\n\n\n\n\n',st)
    %
end

% aux = [erro];
% save('erroProposto.dat', 'aux', '-ascii');

% Power curve plotting
xmin=min(x); xmax=max(x);
incr=0.1;  % increments for curve plotting purposes
xgrid=xmin:incr:xmax;  % x interval for curve plotting
xgrid=xgrid(:);
ygrid = f(xgrid,B);

figure;
plot(x, y, '*', 'color', [0.5 0.5 0.5], 'MarkerSize',2)
hold on
plot(xgrid, ygrid, 'k-', 'linewidth', 3);
xlabel('wind speed [m/s]')
ylabel('generated power [KWatts]')

beep
fprintf(' %s - %s \n',inicio,datestr(now))
