
X=load('turbine1.dat');
x=X(:,1);  % speed samples
y=X(:,2);  % power samples

xmin=min(x); xmax=max(x);
incr=0.1;  % increments for curve plotting purposes
xgrid=xmin:incr:xmax;  % x interval for curve plotting
xgrid=xgrid(:);


H = 536;
B = 0.7301;
V0 = 10.34;
L = -17.85;
G = 0.4548;
E = 1;
 
% H =  -1.2980;
% B = 0.7301;
% V0 = 3.6465 ;
% L = -17.8489;
% G = 0.4548;
% E = -0.0068;

ygrid = L + (H-L)./((E + exp(-B*(xgrid - V0))).^G);

figure;
plot(x,y,'ro',xgrid,ygrid,'b-'); grid
xlabel('wind speed [m/s]')
ylabel('generated power [KWatts]')