    %% Polinomial com o grau como vari�vel a ser otimizada.
    tic
clear all; close all;
warning('off','all')
% warning ('on','all')

%% Aquisi��o e pr�-processamento de dados.
dados = load('data\turbine2.dat');  % Carregar dados.

x_dados = dados(:, 1);	% Valores das ordenadas dos dados.
y_dados = dados(:, 2);	% Valores das abscissas dos dados.
% y_dados = 100*y_dados/max(y_dados);
% Par�metros
vCutIn =  min(x_dados);
vCutOut = max(x_dados);
tol = inf;
interv = floor((max(x_dados) - min(x_dados))*10); %sotavento = 3198

% Corte V_in V_out
% x = (x_dados - min(x_dados))/(max(x_dados) - min(x_dados));
% y = (y_dados - min(y_dados))/(max(y_dados) - min(y_dados));
x = x_dados;
y = y_dados;

fprintf('Elementos: \t %d\n',numel(x_dados));
fprintf('V_min: \t\t %.2f\n', min(x_dados));
fprintf('V_cut_in: \t %.2f\n', vCutIn);
fprintf('V_min: \t\t %.2f\n', max(x_dados));
fprintf('V_cut_out: \t %.2f\n\n', vCutOut);

% iInlier = find((x_dados >= vCutIn)&(x_dados <= vCutOut));
% iOutlier = find((x_dados < vCutIn)|(x_dados > vCutOut));
% % iOutlierAux = [];
% % dv = linspace(min(x_dados), max(x_dados), interv);
% % 
% % disp('      �rea da gaussiana')
% % disp('tol = 0.5:    38,30%')
% % disp('tol = 1.0:    68,26%')
% % disp('tol = 1.5:    86,64%')
% % disp('tol = 2.0:    95,44%')
% % disp('tol = 2.5:    98,76%')
% % disp('tol = 3.0:    99,74%')
% % 
% % 
% % for i = 1: length(dv)-1
% %     %% indice da faixa
% %     index = find((x >= dv(i))&(x < dv(i+1)));
% %     if i == length(dv)-1
% %         index = [index; find(x == dv(i+1))];
% %     end
% %     %%
% %     n = numel(index);
% %     if n ~= 0           % S� entra na rotina se tiver pelo menos 1 elemento 
% %         p = y(index);   % Pot�ncias para o intervalo de velocidade atual
% %         mu = sum(p)/n;  % M�dia
% %         sig = sqrt(sum(abs(p - mu).^2)/n); %Desvio padr�o
% %         z = (p - mu)/(sig + eps);
% %         indexIn = find(abs(z) <= tol);
% %         indexOut = find(abs(z) > tol);
% %         iInlier = [iInlier; index(indexIn)];
% %         iOutlierAux = [iOutlierAux; index(indexOut)];
% %     end
% % end
% % disp('      �rea na pr�tica')
% % fprintf('tol = %.1f:    %.2f %%\n\n', tol, 100*numel(iInlier)/(numel(iOutlierAux)+numel(iInlier)));
% % 
% % iOutlier = [iOutlier; iOutlierAux];
% xOut = x(iOutlier);
% yOut = y(iOutlier);
% 
% x = x(iInlier);
% y = y(iInlier);
% 
%% Ancoragem
quantAncoragem = 0; %round(length(x)*0.0001);
ancoragemX = max(x)*ones(quantAncoragem, 1);
ancoragemY = max(y)*ones(quantAncoragem, 1);

x = [x; ancoragemX];
y = [y; ancoragemY];
% 
% % figure('Name', 'Pre-proc', 'NumberTitle', 'off', 'Units', ...
% % 'normalized', 'Outerposition', [0 0.05 1 0.95]);
% % plot(x, y, '.', xOut, yOut, 'r.');
% % hold on
% % plot([vCutIn vCutIn], get(gca,'ylim'), 'k-')
% % plot([vCutOut vCutOut], get(gca,'ylim'), 'k-')
% % hold off
% % ratio = 100*(numel(x)/numel(x_dados));
% % title(sprintf('inliers/total = %.2f%%', ratio), 'fontsize', 15)
% % legend('Inliers', 'Outliers')
% % pause
% %





%% Hiper par�metros
% Polin�mio
MAXgrau = 9;	% Grau m�ximo do polin�mio.
CoefInd = true; % Existencia do coeficiente independente.
termos = MAXgrau + logical(CoefInd);
robusta = false;

% AG
tam_pop = 20;         % Tamanho da populacao
tam_cro = termos;       % Tamanho do cromossomo (no. de genes)
pc = 0.8;               % Probabilidade de cruzamento
pm = 0.05;               % Probabilidade de mutacao
geracoes = 20;          % Numero de geracoes
elitismo = 1;           % Quantidade de elites passadas para pr�xima gera��o.
%





%% Vari�veis
% Polin�mio
X = ones(numel(x), termos);
if ~CoefInd
    X(:,1) = x;
end
for i = 2:termos
     X(:, i) = x.*X(:,i-1*CoefInd);
end


xx = linspace(min(x), max(x))';
XX = ones(length(xx), termos);
if ~CoefInd
    XX(:,1) = xx;
end
for i = 2:termos
     XX(:, i) = xx.*XX(:,i-1*CoefInd);
end

% AG
pop = GerarPop(tam_pop, tam_cro);
AptMelhor = zeros(geracoes + 1, 1);     % Inicializar com 0s o vetor de Melhor Aptid�o.
%



%% Algoritimo Gen�tico
for t = 1:geracoes
%% Aptid�o
    [Apt, idBest, bBest, aptBest] = Aptidao(pop, y, X, robusta);     % Calcular a aptid�o da popula��o atual e:
    AptMelhor(t) = aptBest;
    %
    
%% Apresenta��o de dados
    m = find(pop(idBest, :));   % Mascara dos par�metros utilizados 
    
    
    Xt = X(:, m);               % Matr�z do melhor polin�mio
%             M�todo alternativo: Zerar os par�metros n�o utilizados.
%     Xt = X;
%     Xt(:, m) = 0;
        
        
    ypred = Xt*bBest;
    erro = y - ypred;
    SEQ = erro'*erro;
    
    fprintf('\n\n ### Gera��o %d ###\n\n', t);
    fprintf(' Melhor apt: %.1f\n', aptBest);
    fprintf(' SEQ: %.1f\n', SEQ);
    fprintf(' Termos: %d\n', numel(m));
    st = [];
    for k = 1:numel(m)
        if bBest(k) >= 0
            if k == 1
                sinal = '';
            else
                sinal = '+';
            end 
        else
            sinal = '-';
        end
        st = strcat(st,sprintf(' %c %fx^%d ', sinal,abs(bBest(k)), m(k)-CoefInd));
    end
    fprintf('%s\n',st)
    
    
    %% Sele��o
    S = Selecao(Apt);                     % Selecionar pais com base na aptid�o atual.

    filhos = Cruzamento(pop, S, pc);    % Gera��o de filhos na etapa de cruzamento.
    filhos = Mutacao(filhos, pm);       % Muta��o de filhos.
    
    % Bloco do Elitismo:
    % a) Caso elitismo = 0, bloco ser� ignorado em tempo de execu��o;
    % b) Caso contr�rio, ir� ser selecionado a quantidade igual ao valor da
    %    vari�vel elitismo da popula��o atual para compor a pr�xima
    %    gera��o, substituindo os filhos gerados com menos aptid�o.
    if (elitismo)
        il = elitismo;
        fitFilho = Aptidao(filhos, y, X, robusta);  % Calculando a aptid�o para cada o caso.
        [~, iPop] = sort(Apt);
        [~, iFilho] = sort(fitFilho, 'descend');
        while(il)
            filhos(iFilho(il),:) = pop(iPop(il),:);
            il = il - 1;
        end
    end
    %%%%%%%%%% Fim do Bloco do Elitismo %%%%%%%%%%

    pop = filhos;       % Atualiza��o da nova popula��o
end
[Apt, idBest, bBest, aptBest] = Aptidao(pop, y, X, robusta);	% Calcular a aptid�o da popula��o atual e:
AptMelhor(t+1) = aptBest;
%

[~, ~, ~, aptBest] = Aptidao(pop(idBest, :), y(1:end-quantAncoragem), X(1:end-quantAncoragem, :), robusta);

%% Apresenta��o de resultados
m = find(pop(idBest, :));


Xt = X(:, m);
%             M�todo alternativo: Zerar os par�metros n�o utilizados.
% Xt = X;
% Xt(:, m) = 0;


ypred = Xt*bBest;
erro = y - ypred;
SEQ = sum(erro.^2);


aux = [erro];
save('erroProposto.dat', 'aux', '-ascii');

% Na tela
fprintf('\n\n\n ### Gera��o %d ###\n\n', t+1);
fprintf(' Melhor apt: %.1f\n', aptBest);
fprintf(' SEQ: %.1f\n', SEQ);
fprintf(' AIC: %.1f\n', AIC(numel(m), numel(x), SEQ));
fprintf(' BIC: %.1f\n', BIC(numel(m), numel(x), SEQ));
fprintf(' Termos: %d\n', numel(m));
st = [];
for k = 1:numel(m)
    if bBest(k) >= 0
        if k == 1
            sinal = '';
        else
            sinal = '+';
        end 
    else
        sinal = '-';
    end
    st = strcat(st,sprintf(' %c %1.10fv^%d ', sinal,abs(bBest(k)), m(k)-CoefInd));
end
fprintf('%s\n',st)

% Gr�ficos
XX = XX(:, m);
y_pred = XX*bBest;

% filename = 'Z:\Mestrado\Disserta��o\scripts\AG\exportExcel.xlsx';
% xlswrite(filename,y_pred', '2', 'f1')
% figure;
% plot(x, y, '*', xx, y_pred);

% figure('Name', 'Teste', 'NumberTitle', 'off', 'Units', ...
%     'normalized', 'Outerposition', [0 0.05 1 0.95]);
% plot(x, y, 'b.', xOut, yOut, 'r.', xx, y_pred, 'k-', xx, y_pred-std(erro), 'k-',xx, y_pred+std(erro), 'k-', 'linewidth', 2);
figure;
plot(x, y, '*', 'color', [0.5 0.5 0.5], 'MarkerSize',2)
hold on
plot(xx, y_pred, 'k-', 'linewidth', 3);
% y_predAux = y_pred-1*std(erro);
% y_predAux(find(y_predAux < min(y)))=min(y);
% plot(xx, y_predAux, 'k--', 'linewidth', 2);
% y_predAux = y_pred+1*std(erro);
% y_predAux(find(y_predAux > max(y))) = max(y);
% plot(xx, y_predAux, 'k--', 'linewidth', 2);
% title(sprintf('APT = %1.6e', aptBest), 'fontsize', 15);
% legend({'Dados coletados', 'Curva de regress�o', 'Limite \sigma'}, 'fontsize', 10)
xlabel('Velocidade do Vento [m/s]', 'fontsize', 12);
ylabel('Pot�ncia de sa�da normalizada [%]', 'fontsize', 12);
% axis([min(x)-0.2 19 min(y)-5 max(y)+5])
% annotation('textbox', [0.15 0.8 0.1 0.1], 'String',strcat('p(v)=',st),'backgroundcolor', 'w', 'FitBoxToText','on');

% offset = 0;% round(geracoes/2);
% figure('Name', 'Teste', 'NumberTitle', 'off', 'Units', ...
%     'normalized', 'Outerposition', [0 0.05 1 0.95]);
% plot([offset:length(AptMelhor)-1], AptMelhor(offset+1:end), 'linewidth', 2);
% title('Evolu��o temporal da aptid�o');
% xlabel('Gera��es');
% ylabel('SEQ');


warning ('on','all');
beep
toc