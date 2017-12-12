clc,clear
load data.txt
[n,m] = size(data)
x=data(:,[4,5,6,8,11,13]);
y=data(:,m);
[d,dt,e,et,stats]=regress(y,[ones(n,1),x]);
rcoplot(e,et);
