    %% Polinomial com o grau como variável a ser otimizada.
clear all; close all; clc;
inicio = datestr(now)
%% Aquisição e pré-processamento de dados.
dados = load('data\JASAturbine6.dat');  % Carregar dados.

x = dados(:, 1);	% Valores das ordenadas dos dados.
y = dados(:, 2);	% Valores das abscissas dos dados.

kindFit = 'bic';

%% Hiper parâmetros
termos = 3;

% AG
tam_pop = 8;           % Tamanho da populacao
tam_cro = termos;       % Tamanho do cromossomo (no. de genes)
pc = 0.8;               % Probabilidade de cruzamento
pm = 0.1;              % Probabilidade de mutacao
geracoes = 1;          % Numero de geracoes
elitismo = 1;           % Quantidade de elites passadas para próxima geração.
%


%% Variáveis
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
aptBest = zeros(geracoes+1, 1);     % Inicializar com 0s o vetor de Melhor Aptidão.
sseBest = zeros(geracoes+1, 1);     % Inicializar com 0s o vetor de Melhor Aptidão.
bBest = zeros(geracoes+1, 6);     % Inicializar com 0s o vetor de Melhor Aptidão.
%

    

%% Initial analysis
[aptPop, ssePop, bPop] = logAptidao2(pop, y, x, base, kindFit);     % Calcular a aptidão da população atual e:
[aptBest(1), iAptBest] = min(aptPop);
sseBest(1) = ssePop(iAptBest);
bBest(1,:) = bPop(iAptBest,:);

%% Apresentação de dados
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

%% Algoritimo Genético
for t = 2:geracoes+1
    
        %% Seleção
    S = Selecao(aptPop);                     % Selecionar pais com base na aptidão atual.
    filhos = Cruzamento(pop, S, pc);    % Geração de filhos na etapa de cruzamento.
    filhos = Mutacao(filhos, pm);       % Mutação de filhos.
    [aptSpa, sseSpa, bSpa] = logAptidao2(filhos, y, x, base, kindFit);   % Calculando a aptidão para cada o caso.
    
    
    % Bloco do Elitismo:
    % a) Caso elitismo = 0, bloco será ignorado em tempo de execução;
    % b) Caso contrário, irá ser selecionado a quantidade igual ao valor da
    %    variável elitismo da população atual para compor a próxima
    %    geração, substituindo os filhos gerados com menos aptidão.
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
    pop = filhos;       % Atualização da nova população
    aptPop = aptSpa;       % Atualização da nova população
    ssePop = sseSpa;       % Atualização da nova população
    bPop = bSpa;       % Atualização da nova população
    [aptBest(t), iAptBest] = min(aptPop);
    sseBest(t) = ssePop(iAptBest);
    bBest(t,:) = bPop(iAptBest,:);

    %% Apresentação de dados
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
