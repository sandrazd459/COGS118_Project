%do regression again using variables kept in stepwise regression
clc,clear
load data.txt

x=data(:,1:13);
t=data(:,14);
Q = size(x, 1);
Q1 = floor(Q * 0.90);
Q2 = Q - Q1;
num = 500;
mse = zeros(1, num);
 for j = 1:num
 ind = randperm(Q)
ind2 = ind(Q1 + (1:Q2));
x2 = x(ind2,[4,5,6,8,11,13]);
t2 = t(ind2,:);
    
ind1 = ind(1:Q1)
x1 = x(ind1,[4,5,6,8,11,13]);
t1 = t(ind1,:);

[d,dt,e,et,stats]=regress(t1,[ones(Q1,1),x1]);
d,stats

y=d(1)+x2 * d(2:end);

mse(j) = sum((y-t2).^2)/Q2;
 end
 mse
mean(mse)
