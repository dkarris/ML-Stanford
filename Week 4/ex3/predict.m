function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

X = [ones(m,1) X];

% layer 2 matrix should be 25unitsx5000 data inputs
LAYER_2 = sigmoid(Theta1*X');

% add one row with ones to add unit 0
LAYER_2 = [ones(1,size(X,1)); LAYER_2];

%calc LAYER 3: 10x5000 matrix - 10 unit x 5000 dataset and then transpose to number of rows 
% being number of inputs and columsn being OnevsAll trained units in row
LAYER_3 = (sigmoid(Theta2*LAYER_2))';

% get indexes of max in each row - prediction for OneVsAll
[prob p] = max(LAYER_3, [], 2);

% =========================================================================