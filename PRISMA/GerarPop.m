% Gerar População
function res = GerarPop(Qnt_Individuos, Tam_Cromossomo)
    res = round(rand(Qnt_Individuos, Tam_Cromossomo));
end