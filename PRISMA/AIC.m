function res = AIC(k, N, SSE)
    res = N*log(SSE) + 2*(k+1);
end

