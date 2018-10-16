close all, format compact, clear all;
tic;

rng(1,'v4normal')
options = optimset('maxIter',1e6,'LargeScale','off','Display','off');

% suppress warnings when hessian matrix is not symetrical for quadprog
warning('off','optim:quadprog:HessianNotSym')

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 1. Linear Hard-margin SVM
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('P2_data.mat');

[l, dim] = size(X);

% See 2.16a from Springer book
H = (Y*Y').*(X*X');
H = H + eye(l)*1e-7;
P = -ones(size(Y)); % negative because matlab by default minimizes quadprod, but we want a maximization
% See 2.16b from Springer book
Aeq = Y';
beq = 0;
% See 2.16c
lb = zeros(size(Y));
ub = inf * ones(size(Y));

alpha = quadprog(H, P, [], [], Aeq, beq, lb, ub, [], options);

% Get weight
W = 0;
for i = 1:size(alpha,1)
    W = W + (alpha(i) * Y(i) * X(i,:)');
end

% Get bias
indices = (alpha > 1e-5); % only support vectors
b = 0;
X_sv = X(indices, :);
Y_sv = Y(indices);
l_sv = size(Y_sv, 1);
for i = 1:l_sv
    b = b + (1/Y_sv(i))-(X_sv(i,:)*W);
end
b = b / l_sv;


m = 1/norm(W);

"Part 1.a"
alphas_for_support_vectors = alpha(alpha > 1e-5)
b
M = m % As specified previously, scalar values should be lowercase, but the assignment states margin "M".

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Find results for given datapoints
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

"Part 1.b"
"o for [3 4]' = " + sign(W'*[3 4]'+b)
"o for [6 6]' = " + sign(W'*[6 6]'+b)

%%%%%%%%%%%%%%
% Plot Results
%%%%%%%%%%%%%%

figure(1);
hold on;

gscatter(X(:,1), X(:,2), Y, 'rb', '.+');

graph_line(W, b, m, 'b-', 'b--');

X_support_vectors = X(alpha > 1e-5, :);
plt = scatter(X_support_vectors(:,1), X_support_vectors(:,2), 'c', 'o');

grid on;
axis([-4 8 -4 8])
xlabel('x');
ylabel('y');
legend("-1", "+1", "decision hyperplane", "margin", "margin", "support vectors")
hold off;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 2. Multiclass Soft-margin SVM
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[Y, X] = libsvmread('glass'); X=full(X);

[l, dim] = size(X);

% Scale each dimension
for i = [1:dim]
    X(:,i) = (X(:,i) - mean(X(:,i))) / std(X(:,i));
end

% Shuffle
[~, indices] = sort(rand(l,1));
X = X(indices, :);
Y = Y(indices);

% Plot arbetrary first dimension v. seventh dimension.
gscatter(X(:,1),X(:,7), Y)
xlabel('First Dimension')
ylabel('Seventh Dimension')
title('First Dimension of glass dataset compared to Seventh Dimension')

C0 = [1e-2 1e-1 1 1e1 1e2 1e3 1e4];
parameters_polynomial = [1 2 3 4 5]; % for polynomial kernel classifier
parameters_gaussian = [1e-2 1e-1 1 1e1 1e2 1e3]; % for Gaussian (i.e. RBF) kernel classifier

% % 
% % 
% % 
% % TODO! Remove these testing lines!
% % 
% % 
% % 
% C0 = [1e-1 1 1e1 1e2];
% parameters_polynomial = [1 2 3]; % for polynomial kernel classifier

% labels for graph
label_strings = strings(length(C0) * 2, 1);
labels_i = 1;

% store errors and accuracy values for plotting later
accuracy_polynomial = zeros(size(parameters_polynomial));
accuracy_gaussian = zeros(size(parameters_gaussian));

% Hold onto all figures for plotting accuracy and # of errors
for i = 2:3
    figure(i)
    hold on
end

unique_classes = unique(Y);

% Collect best parameters FOR EACH CLASS
C_best_polynomial = inf * ones(size(unique_classes));
C_best_gaussian = inf * ones(size(unique_classes));
degree_best_polynomial = inf * ones(size(unique_classes));
sigma_best_gaussian = inf * ones(size(unique_classes));

percentErr_all_polynomial = zeros(length(C0), length(parameters_polynomial), length(unique_classes));

percentErr_best_polynomial = inf * ones(length(unique_classes), 1);
percentErr_best_gaussian = inf * ones(length(unique_classes), 1);

for kernel = 1:2 % 1: polynomial, 2: gaussian %TODO: re-add!
    for C_i = 1:length(C0)
        C = C0(C_i);
        if kernel == 1 
            % polynomial
            parameters = parameters_polynomial;
        else
            % gaussian
            parameters = parameters_gaussian;
        end

        for p_i = 1:length(parameters)
            param = parameters(p_i);

            % Create a classifier for each class
            for cl_i = 1:length(unique(Y))
                class = unique_classes(cl_i);
                % Make true-classes: +1 and false-classes: -1
                Y_class = ((Y == class) * 2) - 1; 
            
                % 5-fold-crossvalidation
                numErr = 0;
                indices = crossvalind('Kfold',Y,5);
                for i = 1:5
                    ind_test = (indices == i);
                    ind_train = ~ind_test;

                    X_train = X(ind_train, :);
                    Y_train = Y_class(ind_train, :);
                    X_test = X(ind_test, :);
                    Y_test = Y_class(ind_test, :);
                    
                    % Actual learning here
                    Y_pred = soft_margin_svm(X_train, Y_train, X_test, Y_test, kernel, param, C);

                    numErr = numErr + length(find(Y_test - sign(Y_pred)));
                end
                
                percentErr = 100 * (numErr / length(Y));
                
%                 percentErr + " kernel=" + kernel + " C=" + C + " param=" + param + " class=" + class % TODO: remove
                
                % IF these are the best parameters for this class, store them for later
                if kernel == 1 
                    % Store ALL for graphing purposes
                    percentErr_all_polynomial(C_i, p_i, cl_i) = percentErr;

                    if percentErr_best_polynomial(cl_i) > percentErr
                        percentErr_best_polynomial(cl_i) = percentErr;

                        accuracy_polynomial(p_i) = percentErr;
                        C_best_polynomial(cl_i) = C;
                        degree_best_polynomial(cl_i) = param;
                    end
                else
                    if percentErr_best_gaussian(cl_i) > percentErr
                        percentErr_best_gaussian(cl_i) = percentErr;

                        accuracy_gaussian(p_i) = percentErr;
                        C_best_gaussian(cl_i) = C;
                        sigma_best_gaussian(cl_i) = param;
                    end
                end
            end
        end
    end
end

hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create a model with the best parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % polynomial SVM C=0.01, d=2
% multi_class_soft_margin(X, Y, 1, 0.01, 2)
% 
% % Gaussian SVM C=0.1, sigma=1
% soft_margin_SVM(X, Y, 2, 0.1, 1)

% Polynomial
"3. Accuracy of best polynomial SVM"
accuracy = (multi_class_soft_margin(X, Y, 1, degree_best_polynomial, C_best_polynomial) / length(Y)) * 100

% Gaussian
"3. Accuracy of best gaussian SVM"
accuracy = (multi_class_soft_margin(X, Y, 2, sigma_best_gaussian, C_best_gaussian) / length(Y)) * 100


% Additional information
"3. Best parameters for each class"

for cl_i = 1:length(unique_classes)
    "Best parameters for class " + unique_classes(cl_i) + " (polynomial) C = " + C_best_polynomial(cl_i) + " degree = " + degree_best_polynomial(cl_i) + " percentError = " + percentErr_best_polynomial(cl_i)
end

for cl_i = 1:length(unique_classes)
    "Best parameters for class " + unique_classes(cl_i) + " (gaussian) C = " + C_best_polynomial(cl_i) + " degree = " + sigma_best_gaussian(cl_i) + " percentError = " + percentErr_best_gaussian(cl_i)
end

toc;

%%%%%%%%%%%%%%
% 
% Sub Routines
% 
%%%%%%%%%%%%%%

function numErr = multi_class_soft_margin(X, Y, kernel, params, Cs)
    [l, ~] = size(X);

    unique_classes = unique(Y);
    
    Y_pred_all = zeros(l,length(unique(Y)'));
    
    for cl_i = 1:length(unique_classes)
        class = unique_classes(cl_i);
        % Make true-classes: +1 and false-classes: -1
        Y_class = ((Y == class) * 2) - 1; 

        Y_pred = soft_margin_svm(X, Y_class, X, Y_class, kernel, params(cl_i), Cs(cl_i));
        Y_pred_all(:,cl_i) = Y_pred';
    end
    [~, ind] = max(Y_pred_all, [], 2);
    ind = unique_classes(ind);
    numErr = length(find((ind - Y) ~= 0));
end

function Y_pred = soft_margin_svm(X_train, Y_train, X_test, Y_test, kernel, param, C)
    [l_test, ~] = size(X_test);
    [l, ~] = size(X_train);

    if kernel == 1 % polynomial
        H = (Y_train * Y_train') .* (((X_train * X_train') + 1) .^ param);
    else           % gaussian
        H = (Y_train * Y_train') .* grbf_fast(X_train,X_train,param);
    end

    H = H + eye(l)*1e-7;

    P = -ones(size(Y_train)); % negative because matlab by default minimizes quadprod, but we want a maximization
    Aeq = Y_train';
    beq = 0;

    % See 2.16c
    lb = zeros(size(Y_train));
    ub = C * ones(size(Y_train));

    options = optimset('maxIter',1e6,'LargeScale','off','Display','off');
    alpha = quadprog(H, P, [], [], Aeq, beq, lb, ub, [], options);
%     alpha = quadprog(H, P, [], [], [], [], lb, ub, [], options);

    e = 1e-5;
    ind_Free = find(alpha >= e & alpha <= C - e);
    ind_Support_Vectors = find(alpha >= e);
    X_free = X_train(ind_Free,:);
    X_Support_Vectors = X_train(ind_Support_Vectors,:);
    Y_Support_Vectors = Y_train(ind_Support_Vectors,:);

    % Figure out the bias from the slideshow p.162/209
    b = 0;
    for j = ind_Free'
        sum = 0;
        for i = ind_Support_Vectors'
            if kernel == 1 % polynomial
                sum = sum + ((((X_train(j,:) * X_train(i,:)') + 1) .^ param));
%                 sum = sum + ((((X_train(j,:) * X_train(i,:)') + 1) .^ param) * alpha(i) * Y_train(i));
            else           % gaussian
                sum = sum + (grbf_fast(X_train(j,:),X_train(i,:),param));
%                 sum = sum + (grbf_fast(X_train(j,:),X_train(i,:),param) * alpha(i) * Y_train(i));
            end
        end
        b = b + (Y_train(j) - sum);
    end
    b = (1/length(ind_Free)) * b;

    % Calculate Y_pred for X_test
    Y_pred = zeros(size(Y_test));
    for j = 1:l_test
        for i = ind_Support_Vectors'
            if kernel == 1 % polynomial
                Y_pred(j) = Y_pred(j) + (((X_test(j,:) * X_train(i,:)') + 1) .^ param);
            else           % gaussian
                Y_pred(j) = Y_pred(j) + grbf_fast(X_test(j,:),X_train(i,:),param);
            end
        end
        Y_pred(j) = Y_pred(j) + b;
    end
end

%%%%%%%%%%%%%%%%%%%%
% 
% 
% Plotting functions
% 
% 
%%%%%%%%%%%%%%%%%%%%

function plt = graph_line(W, b, m, line_type, line_type_2)
    x_intercept = -(b/W(1));
    y_intercept = -(b/W(2));
    slope = -(b/W(2))/(b/W(1));

    x_matrix = -10:20;
    y_matrix = y_intercept + (slope * x_matrix);
    y_diff = m * sqrt(slope^2 + 1);

    plt = plot(x_matrix, y_matrix, line_type);
    plt = plot(x_matrix, y_matrix + y_diff, line_type_2);
    plt = plot(x_matrix, y_matrix - y_diff, line_type_2);
end