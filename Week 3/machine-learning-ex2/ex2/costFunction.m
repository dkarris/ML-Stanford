function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

% DK Cost function Solution1: vectorized
J = -sum(y .* log(sigmoid(X*theta))+(1-y) .* log(1-sigmoid(X*theta)))/m;
% DK End solution 1


% DK Cost function Solution2: elememt - wise
#{
J_temp = 0;
for i=1:m
    J_temp = J_temp + (y(i)*log(sigmoid(theta'*X(i,:)'))+(1-y(i))*log(1-sigmoid(theta'*X(i,:)')));
end
J = -J_temp/m;
#}
% DK End Cost function solution 2

% DK Gradient descent element wise maximum details from formuals. Most inefficient - non vectorized

#{
for n=1:length(theta)
    subgrad = 0;
    for i=1:m
     h0x = sigmoid(theta'*X(i,:)');
     bias = h0x - y(i);
     subgrad = subgrad + bias*X(i,n);
    end
    grad(n) = subgrad/m;
end
#}

% DK end Gradient descent element wise Most inefficient - non unvectorized

% DK Gradient descent - partially vectorized 

#{
printf('running partially vectorized\n')
for n=1:length(theta)
    grad(n) = sum((sigmoid(X*theta)-y) .* (X(:,n)))/m;
end
#}

% END DK Gradient descent - partially vectorized

% DK - gradient descent fully vectorized


grad = sum((sigmoid(X*theta)-y) .* X)/m;

% DK - END gradient descent fully vectorized
% =============================================================
end
