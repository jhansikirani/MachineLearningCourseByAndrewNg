function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(m,1) X];
Z2 = X * Theta1';
A2 = sigmoid(Z2);
A2 = [ones(rows(A2),1) A2];
Z3 = A2 * Theta2';
A3 = sigmoid(Z3);

vectorizedY = eye(num_labels);
vectorizedY = vectorizedY(y,:); 
myOnes2 = ones(rows(vectorizedY), columns(vectorizedY));


T1 = vectorizedY .* log(A3);
T2 = (myOnes2  - vectorizedY) .* log(myOnes2 - A3);

Theta1Sq = Theta1.^2;
SumTheta1Sq = sum(Theta1Sq);
Theta2Sq = Theta2.^2;
SumTheta2Sq = sum(Theta2Sq);

T =  T1 + T2;
J = -1 * sum(T(:)) /m + lambda * (sum(SumTheta1Sq) + sum(SumTheta2Sq))/(2*m) ;
%Subtract the bias parameters from the cost function. We dont want to regularize the bias parameters
J = J - lambda * (SumTheta1Sq(1) + SumTheta2Sq(1))/(2*m);

for i = 1:m
  A1I = X(i,:);
  Z2I = A1I * Theta1';
  A2I = sigmoid(Z2I);
  A2I = [ones(rows(A2I),1) A2I];
  Z3I = A2I * Theta2';
  A3I = sigmoid(Z3I);
  delta3 = A3I - vectorizedY(i,:);
  delta2 = delta3 * Theta2  .* A2I .* (ones(rows(A2I)) - A2I);
  delta2 = delta2(2:end);
  Theta1_grad = Theta1_grad + delta2' * A1I;
  Theta2_grad = Theta2_grad + delta3' * A2I;
endfor


RegularizationFactorTheta1 = Theta1;
RegularizationFactorTheta1 = RegularizationFactorTheta1(:,2:end);
RegularizationFactorTheta1 = [zeros(rows(RegularizationFactorTheta1),1) RegularizationFactorTheta1];

RegularizationFactorTheta2 = Theta2;
RegularizationFactorTheta2 = RegularizationFactorTheta2(:,2:end);
RegularizationFactorTheta2 = [zeros(rows(RegularizationFactorTheta2),1) RegularizationFactorTheta2];

Theta1_grad = Theta1_grad/m + lambda * RegularizationFactorTheta1 / m;
Theta2_grad = Theta2_grad/m + lambda * RegularizationFactorTheta2 / m;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
