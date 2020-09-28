function [res, iBest, bBest, aptBest] = Aptidao(alphas, p, V, flagRobusta)
%% Fun��o de aptid�o
% Vari�veis de entrada
%   - alphas: Matriz com os alphas. Cada linha representa um arranjo de
%   coeficientes que ser�o ativados a depender dos valores das colunas, que
%   podem assumir os valores 0 ou 1. Se a coluna de indice i estiver em 1,
%   o coeficiente i ser� utilizado na regress�o. Caso seja 0, n�o ser�.
%   - p: Valor da pot�ncia mensurada.
%   - V: Matriz de Vandermonde da velocidade mensurada.
%   - flagRobusta: Indicativo se a regress�o ser� robusta (IWLS)

    %% Inicializa��o das vari�veis de sa�da.
    res = zeros(length(alphas), 1); % Aptid�o da popula��o.
    iBest = 0; % Indice da melhor aptid�o.
    bBest = []; % Coeficientes da melhor aptid�o.
    aptBest = inf; % Melhor (menor) aptid�o da popula��o.
    
    %% Inicio da an�lise dos conjuntos de alphas.
    for i = 1:size(alphas,1) % Fazer para cada linha de alpha.
        indices = find(alphas(i, :)); % Indice dos coef. utilizados.
        nParam = length(indices); % Quantidade de coef. utilizados.
        if nParam < 1 % Caso n�o tenha coeficientes, penalize.
            apt = inf;
        else % Caso tenha, seguir com o procedimento normal.
            Vt = V(:, indices); % Redefina a matriz V em fun��o dos par�metros utilizados.
%             M�todo alternativo: Zerar os par�metros n�o utilizados.
%             Vt = zeros(size(V));
%             Vt(:, indices) = V(:, indices);
        
            %% Resolver B.
            if flagRobusta % Caso robusta, resolver B por IWLS.
                B = robustfit(Vt,p,'bisquare', [],'off');
            else % Caso contr�rio, resolver B pelo m�todo das derivadas utilizando decomposi��o QR.
                if exist('B', 'var') == 1
                    clear B;
                end
                
                [Q,R] = qr(Vt,0); % Decomposi��o matricial QR.
                B = R \ (Q'*p);
                
                
%                 [Q, R, perm] = qr(Vt,0);
%                 tol = abs(R(1)) * max(size(Vt)) * eps(class(R));
%                 xrank = sum(abs(diag(R)) > tol);
%                 if xrank == size(Vt,2)
%                     B(perm, :) = R \ (Q'*p);
%                 else
%                     B(perm, :) = [R(1:xrank,1:xrank) \ (Q(:,1:xrank)'*p); zeros(size(Vt,2)-xrank,1)];
%                 end
            end
            
            %% C�lculo de aptid�o.
            p_pred = Vt*B; % Pot�ncia predita.
            erro = p - p_pred;
            SSE = erro'*erro; % Soma dos erros quadr�ticos.
                apt = SSE;
%                 apt = AIC(nParam, length(p), SSE);
%                 apt = BIC(nParam, length(p), SSE);
    %             apt = FPE(nParam, length(p), SSE);
    %             apt = 1 - R2A(nParam, length(p), SSE, sum((mean(p) - p).^2));
        end
        res(i) = apt; % Inserir valor de aptid�o do conjunto alpha i.
        if apt < aptBest % Se for o melhor conjunto, atualizar valores de sa�da.
            iBest = i;
            bBest = B;
            aptBest = apt;
        end
    end
end
