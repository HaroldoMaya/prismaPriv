function res = BIC(k, N, SSE)
    res = N*log(SSE) + log(N)*(k+1);
end