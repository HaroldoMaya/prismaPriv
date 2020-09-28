function [aptPop, ssePop, BPop] = logAptidao(pop, y, x, base, kindFit)
%% Função de aptidão
% Variáveis de entrada
%   - alphas: Matriz com os alphas. Cada linha representa um arranjo de
%   coeficientes que serão ativados a depender dos valores das colunas, que
%   podem assumir os valores 0 ou 1. Se a coluna de indice i estiver em 1,
%   o coeficiente i será utilizado na regressão. Caso seja 0, não será.
%   - p: Valor da potência mensurada.
%   - V: Matriz de Vandermonde da velocidade mensurada.
%   - flagRobusta: Indicativo se a regressão será robusta (IWLS)

    %% Inicialização das variáveis de saída.
    nPop = size(pop,1);
    aptPop = zeros(nPop, 1); % Aptidão da população.
    ssePop = zeros(nPop, 1); % Aptidão da população.
    BPop = zeros(nPop, 6); % Aptidão da população.
    
    
    %% Inicio da análise dos conjuntos de alphas.
    for i = 1:nPop % Fazer para cada linha de alpha.
        fprintf(' %d',i)
        cPop = pop(i, :);
        indices = find(cPop); % Indice dos coef. utilizados.
        nParam = length(indices); % Quantidade de coef. utilizados.
        if nParam < 1 % Caso não tenha coeficientes, penalize.
            FIT = inf;
            SSE = inf;
            B = zeros(1,6);
        else % Caso tenha, seguir com o procedimento normal.
            %% Resolver B.
%             [B SSE FIT] = logisticFitDE(x,y,pop(i, :),base,kindFit);
            
            ind = 1;
            if cPop(1)
                b1 = sprintf('B(%d)',ind);
                ind = ind + 1;
            else
                b1 = num2str(base(1));
            end
            if cPop(2)
                b2 = sprintf('B(%d)',ind);
                ind = ind + 1;
            else
                b2 = num2str(base(2));
            end
            if cPop(3)
                b3 = sprintf('B(%d)',ind);
                ind = ind + 1;
            else
                b3 = num2str(base(3));
            end
            if cPop(4)
                b4 = sprintf('B(%d)',ind);
                ind = ind + 1;
            else
                b4 = num2str(base(4));
            end
            if cPop(5)
                b5 = sprintf('B(%d)',ind);
                ind = ind + 1;
            else
                b5 = num2str(base(5));
            end
            if cPop(6)
                b6 = sprintf('B(%d)',ind);
            else
                b6 = num2str(base(6));
            end
            fStr = sprintf('@(B,x) %s + (%s-%s)./((%s + exp(-%s*(x - %s))).^%s)',b4,b1,b4,b6,b3,b2,b5);
            f = str2func(fStr);
            
            options = optimoptions('lsqcurvefit','MaxIter', 10000, 'MaxFunEvals', 10000, 'Display','off');
            pFit = lsqcurvefit(f , base(indices), x, y, [], [], options);
            B = base;
            B(indices) = pFit;
            
            y_hat = f(pFit,x);
            [ FIT, SSE ]= fitValue(y, y_hat, nParam, pFit, kindFit, 0, max(y));
        end
        aptPop(i) = FIT; % Inserir valor de aptidão do conjunto alpha i.
        ssePop(i) = SSE;
        BPop(i,:) = B;
        
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

        fprintf(' ### %d Parameters:  %s ###\n', nParam, num2str(cPop));
        fprintf(' Best fitness(%s): %1.4f\n', kindFit, FIT);
        fprintf(' Best SSE: %1.4f\n', SSE);
        fprintf(' %s\n\n',st)
    end
    
    
    function [fit, sse] = fitValue(y_data, y_pred, NParam, Param, kindFit, limInf, limSup)
    N = numel(y_data);
    err = y_data - y_pred;
    sse = err'*err;    % Sum-of-squared errors.
    switch(kindFit)
        case 'sse' % Sum-of-squared errors.
            fit = sse; 
        case 'nmse' % Normalized Mean Square Error.
            fit = sse/((y_data - mean(y_data))'*(y_data - mean(y_data))) + 0.00001*Param*Param'; % 
        case 'aic' % Akaike information criterion (AIC).
            fit = N*log(sse) + 2*(NParam + 1); 
        case 'bic' % Bayesian information criterion (BIC).
            fit = N*log(sse) + (NParam + 1)*log(N); 
        case 'fpe' % Final prediction error criterion (FPE).
            fit = N*log(sse) + popN*log((N + NParam)/(N-NParam)); 
        case 'r2' % R-Squared (R2).
            fit = 1 - sse/(sum((mean(y_data)-y_data).^2)); 
        case 'r2a' % R-Squared adjusted (R2A).
            fit = 1 - ((N-1)/(N - (NParam+1)))*sse/(sum((mean(y_data)-y_data).^2)); 
        otherwise
            fit = sse; % Sum-of-squared errors.
    end
    if (limInf>-Inf)
        if (sum(y_pred < limInf) > 0)
            fit = 10*fit;  % If there is power below limInf, penalize it.
        end
    end
    
    if (limSup<Inf)
        if (sum(y_pred > limSup) > 0)
            fit = 10*fit;  % If there is power below limInf, penalize it.
        end
    end
end
end
