function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    hypothesis = X * theta -  y;
    n = length(theta);
    for j = 1:n
        theta(j) = theta(j) - alpha/m * (hypothesis' * X(:,j));
    end

    % ============================================================
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    fprintf('the theta is \n');
     fprintf('%f\n', theta);
    fprintf('the cost is %f\n', J_history(iter));
      if(J_history(iter) < 24)
      break;

end

end