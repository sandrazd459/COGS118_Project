%do regression wiping out the bad datas
clc,clear
load data.txt

x=data(1:506,1:13);
t=data(1:506,14);
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
ind1(ind1==142|ind1==162|ind1==167)=[];
ind1(ind1==187|ind1==196|ind1==204)=[];
ind1(ind1==205|ind1==215|ind1==226)=[];
ind1(ind1==229|ind1==234|ind1==254)=[];
ind1(ind1==268|ind1==365|ind1==366)=[];
ind1(ind1==368|ind1==369|ind1==370)=[];
ind1(ind1==371|ind1==372|ind1==373)=[];
ind1(ind1==374|ind1==375|ind1==376)=[];
ind1(ind1==381|ind1==413|ind1==506)=[];
Q3 = length(ind1);
x1 = x(ind1,[4,5,6,8,11,13]);
t1 = t(ind1,:);

[d,dt,e,et,stats]=regress(t1,[ones(Q3,1),x1]);
d,stats

y=d(1)+x2 * d(2:end);

mse(j) = sum((y-t2).^2)/Q2;
 end
 mse
 mean(mse)
