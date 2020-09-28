% Seleção para cruzamento
function res = Selecao(fit)
tam_pop = size(fit, 1);    % Tamanho da população.
S = zeros(round(tam_pop/2), 2); % Inicializa o vetor de retorno com os indices de pais selecionados.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Torneio %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
for i = 1 : round(tam_pop/2)

    I = randperm(tam_pop, 2);                  % Permutação randômica de 1 até tam_pop.
    if(fit(I(1)) <= fit(I(2)))    % Competição entre o primeiro e o segundo elementos para decidir quem será o pai.
        pai = I(1);
    else
        pai = I(2);
    end

    while(I(1) == pai || I(2) == pai)       % Selecione 2 candidatos diferentes do pai para pleitear ser a mãe.
        I = randperm(tam_pop);
    end

    if(fit(I(1)) <= fit(I(2)))    % Competição entre o primeiro e o segundo elementos para decidir quem será a mãe.
        mae = I(1);
    else
        mae = I(2);
    end
    S(i, :) = [pai mae];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

res = S;    % Retorna selecionados.

end