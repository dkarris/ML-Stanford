function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
%C = 1;
%sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


C_values = [0.01 0.03 0.1 03 1 3 10 30]';
sigma_values = C_values;
pred_error_matrix = [];
k = 1;
for i=1:length(C_values)
    for j=1:length(sigma_values)
        C_value = C_values(i);sigma_value = sigma_values(j);
        fprintf(['running training cycle:%f'],k);
        fprintf(['training with C and sigma parameters:\n%f,%f'], C_value,sigma_value);
        %model = svmTrain(X,y,C_value, @(x1, x2) gaussianKernel(X(:,1), X(:,2), sigma_value));
        model = svmTrain(X,y,C_value, @(x1, x2) gaussianKernel(x1, x2, sigma_value));
        predictions = svmPredict(model, Xval);
        pred_error = mean(double(predictions ~= yval));
        pred_error_matrix = [pred_error_matrix;[C_value sigma_value pred_error]];
    end
end

[dummy idx_pred_error] = min(pred_error_matrix(:,3));

C = pred_error_matrix(idx_pred_error,1);
sigma = pred_error_matrix(idx_pred_error,2);

% =========================================================================

end
