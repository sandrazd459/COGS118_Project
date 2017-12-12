clc,clear
load data.txt
m=size(data,1)
x0=data(:,[1:13]);
y=data(:,14);
stepwise(x0,y)

