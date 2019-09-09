function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

temp_theta = zeros(size(theta));
temp_theta = [0; theta(2:,1)]

myones = ones(m,1);
T = y'*log(sigmoid(X*theta)) + (myones-y)'*log(myones - sigmoid(X*theta));
J = -1 * T /m + lambda* (temp_theta' *temp_theta)/(2*m);


grad = (X' * (sigmoid(X*theta) - y))/m + lambda * temp_theta / m;

% =============================================================

end
