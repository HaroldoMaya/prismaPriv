% Mutação
function res = Mutacao(populacao, chance_de_mutacao)
    tam_pop = size(populacao, 1);
    tam_cro = size(populacao, 2);
    u = rand(tam_pop, tam_cro) <= chance_de_mutacao;    % Sortear chances de mutação
    res = abs(populacao - u);                           % Fazer mutação.
end