clc,clear
load data.txt
x=data(:,1:13);
y=data(:,14);
Q = size(x, 1);
Q1 = floor(Q * 0.90);
Q2 = Q - Q1;
num = 500;
mse = zeros(1, num);

for j = 1:num
    
ind = randperm(Q)
ind2 = ind(Q1 + (1:Q2));
x2 = x(ind2,:);
y2 = y(ind2,:);    
ind1 = ind(1:Q1)
x1 = x(ind1,:);
y1 = y(ind1,:);
m = mean(x1);
s = std(x1);
x1=zscore(x1);
r=corrcoef(x1)
[vec1,val,con1]=pcacov(r)
f1=repmat(sign(sum(vec1)),size(vec1,1),1);
vec2=vec1.*f1;
f2=repmat(sqrt(val)',size(vec2,1),1);
a=vec2.*f2
num=5                         
am=a(:,[1:num]);
[bm,t]=rotatefactors(am,'method','varimax')
gongtongdu=sum(bm.^2,2)
bt=[bm,a(:,[num+1:end])];
con2=sum(bt.^2)
rate=con2(1:num)/sum(con2)
coef=inv(r)*bm
score=x1*coef;
t1=score(:,1);
t2=score(:,2);
t3=score(:,3);
t4=score(:,4);
t5=score(:,5);
x0=[ones(Q1,1),t1,t2,t3,t4,t5];
[b,bint,r,rint,stats]=regress(y1,x0)
b,bint,stats
x2=x2-repmat(m,Q2,1);
x2=x2./repmat(s,Q2,1);
score=x2*coef;
y0=b(1)+score * b(2:end);

mse(j) = sum((y0-y2).^2)/Q2;

end
mse
 mean(mse)
