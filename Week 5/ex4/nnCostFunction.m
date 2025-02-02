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


% DK converting y vector 1..10 into a matrix ([0..0;1..0;0 1 ..] etc)

%y_matrix = zeros(size(y),10); 

y_matrix = eye(num_labels)(y,:);

% DK now each row of y_vectorized is smth like [0 0 0 0 1 0 0 0 ]  


% DK forward propagation

X = [ones(m,1) X]; % now we have 5000X401 matrix with first column of ones

% hidden layer #1

Z_2 = Theta1*X'; %required later for backpropagation
LAYER_2 = sigmoid(Z_2); % Theta1 25x401  * X' 401*5000 = 25*5000; 25 units in rows and 5000 training sets

% add one row with ones to add unit 0

LAYER_2 = [ones(1,size(X,1)); LAYER_2];

LAYER_3 = sigmoid(Theta2*LAYER_2); % Theta2 10*26 * LAYER_2 26*5000 = 10*5000. Rows - 10 units for 5000 training sets

LAYER_3 = LAYER_3; 

% LAYER_3 - contains predictions. Now need to calculate cost function

% DK - cost function - for loop


% unregularized vectorized
temp_cost =  -(trace(log(LAYER_3)*y_matrix) + trace(log(1-LAYER_3)*(1-y_matrix)))/m;

% regularied part

reg_part = (sum(Theta1(:,2:end)(:) .^ 2) + sum(Theta2(:,2:end)(:) .^ 2))*lambda/(2*m); 

J = temp_cost + reg_part;
% end regularized

% DK ----------------------

% DK  Start Backpropagation algorithm with vectorization

delta_3 = LAYER_3' - y_matrix;
D2 = delta_3' *  (LAYER_2)';

delta_2 = (delta_3 * Theta2(:,2:end)) .* sigmoidGradient(Z_2');
D1 = delta_2'*X;


%{

for t=1:m
    
	a_1 = X(t,:)';
    z_2 = Theta1*a_1;
	a_2 = sigmoid(z_2);
    a_2 = [1;a_2];
    a_3 = sigmoid(Theta2*a_2); %forward propagation complete
    
	y_binary = (1:num_labels)' == y(t); % convert y(t) into binary vector smth like [0;0;0;1;0;0;0...])
	
	% Layer 2
	delta_3_old = a_3-y_binary;
	D2 = D2 + delta_3*(a_2)';
	
	
	% Layer 1
	delta_2 = (Theta2(:,2:end))'*delta_3 .*  sigmoidGradient(z_2);
	D1 = D1 + delta_2*(a_1)';
	
	% delta 1 and D1
	delta_1 = (Theta1(:,2:end)'*delta_2) .* sigmoidGradient(a_1(2:end));
	
end

%}






% -------------------------------------------------------------


% Regularization for gradient

Theta1_grad = D1/m + [zeros(size(D1,1),1) Theta1(:,2:end)] * lambda/m;
Theta2_grad = D2/m + [zeros(size(D2,1),1) Theta2(:,2:end)] * lambda/m;




% =========================================================================





% Unroll gradientsgrad
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
