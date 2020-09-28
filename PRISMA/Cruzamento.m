% Cruzamento
function res = Cruzamento(populacao, indice_selecionados, chance_de_cruzamento)
    tam_pop = size(populacao, 1);
    P = [];
    for i = 1 : round(tam_pop/2)
        if(chance_de_cruzamento >= rand)    % Teste para verificar se h� cruzamento. Caso n�o passe, os filhos ser�o os pr�prios pais.
            % Determina filhos
            div = floor((size(populacao, 2) - 1)*rand) + 1;     % Seleciona um divisor rand�mico para os cromossomos.
            F1 = [populacao(indice_selecionados(i,1),1:div) populacao(indice_selecionados(i,2),div+1:end)]; % Armazene o primeiro filho.
            if(mod(tam_pop, 2) == 0 || i ~= round(tam_pop/2))   % Caso a popula��o seja de tamanho impar e essa seja a ultima itera��o de cruzamento, n�o ter� segundo filho. Caso contr�rio, armazene o segundo filho.
                F2 = [populacao(indice_selecionados(i,2),1:div) populacao(indice_selecionados(i,1),div+1:end)];
            end
        else
            F1 = populacao(indice_selecionados(i,1),:);
            if(mod(tam_pop, 2) == 0 || i ~= round(tam_pop/2))
                F2 = populacao(indice_selecionados(i,2),:);
            end
        end
        if(mod(tam_pop, 2) == 0 || i ~= round(tam_pop/2))   % Caso o tamanho da popula��o seja impar e seja a ultima itera��o, descarte o segundo filho.
            P = [P;F1;F2];
        else
            P = [P;F1];
        end
    end
    res = P;    % Retorna prole.

end