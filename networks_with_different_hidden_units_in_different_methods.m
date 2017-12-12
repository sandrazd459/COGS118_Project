clc,clear
load data.txt

x=data(1:506,1:13)';
t=data(1:506,14)';
Q = size(x, 2);
Q1 = floor(Q * 0.90);
Q2 = Q - Q1;
  numNN = 50;
NN = cell(14, numNN);
perfs = zeros(14, numNN);
for j = 2:15

net=fitnet(j,'trainbr');            
net.trainParam.epochs=3000;                                              
                                                                
for i = 1:numNN                                                              
 ind = randperm(Q);
ind1 = ind(1:Q1);
ind2 = ind(Q1 + (1:Q2));
x1 = x(:, ind1);
t1 = t(:, ind1);
x2 = x(:, ind2);
t2 = t(:, ind2);
[inputn,inputps]=mapminmax(x1);
[outputn,outputps]=mapminmax(t1);
inputn_test=mapminmax('apply',x2,inputps);

NN{j-1,i}=train(net,inputn,outputn);               
bn=NN{j-1,i}(inputn_test);
BPoutput=mapminmax('reverse',bn,outputps);
 perfs(j-1,i) = mse(net, t2, BPoutput);
  
end
perfs

end
