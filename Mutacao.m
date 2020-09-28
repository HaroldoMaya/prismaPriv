% Muta��o
function res = Mutacao(populacao, chance_de_mutacao)
    tam_pop = size(populacao, 1);
    tam_cro = size(populacao, 2);
    u = rand(tam_pop, tam_cro) <= chance_de_mutacao;    % Sortear chances de muta��o
    res = abs(populacao - u);                           % Fazer muta��o.
end