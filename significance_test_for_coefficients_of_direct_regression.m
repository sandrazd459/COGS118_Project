%the significance test for coefficients of direct regression
clc,clear
load data.txt
[n,m] = size(data)
x=data(:,[1:m-1]);
y=data(:,m);
[d,dt,e,et,stats]=regress(y,[ones(n,1),x]);
d,stats
q=sum(e.^2)
ybar=mean(y)
y0=d(1)+x * d(2:end);
u=sum((y0-ybar).^2);
F=u/(m-1)/(q/(n-m))
fw1=finv(0.025,m-1,n-m)
fw2=finv(0.975,m-1,n-m)
c=diag(inv([ones(n,1),x]'*[ones(n,1),x]))
t=d./sqrt(c)/sqrt(q/(n-m))
tfw=tinv(0.975,n-m)
rmse1=sqrt(sum((y0-y).^2)/(n-m))
