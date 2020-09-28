function [res, iBest, bBest, aptBest] = Aptidao(alphas, p, V, flagRobusta)
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
    res = zeros(length(alphas), 1); % Aptidão da população.
    iBest = 0; % Indice da melhor aptidão.
    bBest = []; % Coeficientes da melhor aptidão.
    aptBest = inf; % Melhor (menor) aptidão da população.
    
    %% Inicio da análise dos conjuntos de alphas.
    for i = 1:size(alphas,1) % Fazer para cada linha de alpha.
        indices = find(alphas(i, :)); % Indice dos coef. utilizados.
        nParam = length(indices); % Quantidade de coef. utilizados.
        if nParam < 1 % Caso não tenha coeficientes, penalize.
            apt = inf;
        else % Caso tenha, seguir com o procedimento normal.
            Vt = V(:, indices); % Redefina a matriz V em função dos parâmetros utilizados.
%             Método alternativo: Zerar os parâmetros não utilizados.
%             Vt = zeros(size(V));
%             Vt(:, indices) = V(:, indices);
        
            %% Resolver B.
            if flagRobusta % Caso robusta, resolver B por IWLS.
                B = robustfit(Vt,p,'bisquare', [],'off');
            else % Caso contrário, resolver B pelo método das derivadas utilizando decomposição QR.
                if exist('B', 'var') == 1
                    clear B;
                end
                
                [Q,R] = qr(Vt,0); % Decomposição matricial QR.
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
            
            %% Cálculo de aptidão.
            p_pred = Vt*B; % Potência predita.
            erro = p - p_pred;
            SSE = erro'*erro; % Soma dos erros quadráticos.
                apt = SSE;
%                 apt = AIC(nParam, length(p), SSE);
%                 apt = BIC(nParam, length(p), SSE);
    %             apt = FPE(nParam, length(p), SSE);
    %             apt = 1 - R2A(nParam, length(p), SSE, sum((mean(p) - p).^2));
        end
        res(i) = apt; % Inserir valor de aptidão do conjunto alpha i.
        if apt < aptBest % Se for o melhor conjunto, atualizar valores de saída.
            iBest = i;
            bBest = B;
            aptBest = apt;
        end
    end
end
