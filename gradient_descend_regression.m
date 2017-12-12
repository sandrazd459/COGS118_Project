clear ; close all; clc;
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

X = [ones(Q1, 1), x1]; % Add a column of ones to x
theta = [37;3;-19;4;-1;-1;0]; % initialize fitting parameters

iterations =100000;
alpha = 0.001;

fprintf('\n开始梯度下降迭代 ...\n')
% run gradient descent
theta = gradientDescent(X, t1, theta, alpha, iterations);

% print theta to screen
fprintf('Theta found by gradient descent:\n');
fprintf('%f\n', theta);

y=theta(1)+x2 * theta(2:end);

mse(j) = sum((y-t2).^2)/Q2;
 end
 mse
 mean(mse)
