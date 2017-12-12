clc,clear
load data.txt
x0=data(:,1:13);
y0=data(:,14);
Q = size(x0, 1);
Q1 = floor(Q * 0.90);
Q2 = Q - Q1;
num = 500;
mse = zeros(1, num);

for j = 1:num
    
ind = randperm(Q)
ind2 = ind(Q1 + (1:Q2));
x2 = x0(ind2,:);
y2 = y0(ind2,:);
    
ind1 = ind(1:Q1)
x1 = x0(ind1,:);
y1 = y0(ind1,:);

r=corrcoef(x0)
xd=zscore(x1);
yd=zscore(y1);
[vec1,lamda,rate]=pcacov(r)
f=repmat(sign(sum(vec1)),size(vec1,1),1);
vec2=vec1.*f
contr=cumsum(rate)/sum(rate)
df=xd*vec2;
num = 5;
hg21=df(:,[1:num])\yd                     %主成分变量的回归系数
hg22=vec2(:,1:num)*hg21
hg23=[mean(y0)-std(y0)*mean(x0)./std(x0)*hg22,std(y0)*hg22'./std(x0)]
y=hg23(1)+x2 * hg23(2:end)';

mse(j) = sum((y-y2).^2)/Q2;


end
mse
 mean(mse)
