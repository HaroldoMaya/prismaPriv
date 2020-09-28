% % ds = 1; ls = 0.4;
% ds = 0.5; ls = 0.1;
% 
% 4/max(y)*(...
%       mean(y((x>=((max(x) - min(x))/2 + (ds - ls/2))) & (x<=((max(x) - min(x))/2 + (ds + ls/2))))) ...
%     - mean(y((x<=((max(x) - min(x))/2 - (ds - ls/2))) & (x>=((max(x) - min(x))/2 - (ds + ls/2))))))



X=load('data\JASAturbine5.dat');
x=X(:,1);  % speed samples
y=X(:,2);  % power samples
% 
% minx = min(x);
% miny = min(y);
minx = min(x);
miny = min(y);
interv = 0.25;
p = [];
p = [p; minx, mean(y(x <= interv))]; 
for i=(minx+interv):interv:floor(max(x))
    p = [p; i, mean(y((x >= p(end,1)) & (x <= p(end,1)+2*interv)))];
end
[~,indx] = min(abs(p(:,2) - max(y)/2));
vip = p(indx,1)
Beta = 4/max(y)*((p(indx+1,2) - p(indx-1,2))/(p(indx+1,1) - p(indx-1,1)))

figure;
plot(x, y, '*', 'color', [0.5 0.5 0.5], 'MarkerSize',2)
hold on
plot(p(:,1),p(:,2), '*', 'color', 'r', 'MarkerSize',4)
xlabel('wind speed [m/s]')
ylabel('generated power [KWatts]')







Vip = [];
Beta = [];
for interv = 0.1:0.01:1;
    interv
    p = [];
    p = [p; minx, mean(y(x <= interv))]; 
    for i=(minx+interv):interv:floor(max(x))
        p = [p; i, mean(y((x >= p(end,1)) & (x <= p(end,1)+2*interv)))];
    end
    [~,indx] = min(abs(p(:,2) - max(y)/2));
    Vip = [Vip; p(indx,1)];
    Beta = [Beta; 4/max(y)*((p(indx+1,2) - p(indx-1,2))/(p(indx+1,1) - p(indx-1,1)))];
end
figure;
plot(0.1:0.01:1,Beta)
xlabel('Intervalor de regição para cada ponto')
ylabel('Inclinação da reta estimada ')

figure;
plot(0.1:0.01:1,Vip)
xlabel('wind speed [m/s]')
ylabel('generated power [KWatts]')