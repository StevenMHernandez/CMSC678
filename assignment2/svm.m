close all, format compact, clear all;
tic;

rng(1,'v4normal')
options = optimset('maxIter',1e6,'LargeScale','off','Display','off');

% suppress warnings when hessian matrix is not symetrical for quadprog
warning('off','optim:quadprog:HessianNotSym')

% %%%%%%%%%%%%%%%%%%%%%%%%%%%
% % 
% % 1. Linear Hard-margin SVM
% % 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% load('P2_data.mat');
% 
% [l, dim] = size(X);
% 
% % See 2.16a from Springer book
% H = (Y*Y').*(X*X');
% H = H + eye(l)*1e-7;
% P = -ones(size(Y)); % negative because matlab by default minimizes quadprod, but we want a maximization
% % See 2.16b from Springer book
% Aeq = Y';
% beq = 0;
% % See 2.16c
% lb = zeros(size(Y));
% ub = inf * ones(size(Y));
% 
% alpha = quadprog(H, P, [], [], Aeq, beq, lb, ub, [], options);
% 
% % Get weight
% W = 0;
% for i = 1:size(alpha,1)
%     W = W + (alpha(i) * Y(i) * X(i,:)');
% end
% 
% % Get bias
% indices = (alpha > 1e-5); % only support vectors
% b = 0;
% X_sv = X(indices, :);
% Y_sv = Y(indices);
% l_sv = size(Y_sv, 1);
% for i = 1:l_sv
%     b = b + (1/Y_sv(i))-(X_sv(i,:)*W);
% end
% b = b / l_sv;
% 
% 
% m = 1/norm(W);
% 
% "Part 1.a"
% alphas_for_support_vectors = alpha(alpha > 1e-5)
% b
% M = m % As specified previously, scalar values should be lowercase, but the assignment states margin "M".
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Find results for given datapoints
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% "Part 1.b"
% "o for [3 4]' = " + sign(W'*[3 4]'+b)
% "o for [6 6]' = " + sign(W'*[6 6]'+b)
% 
% %%%%%%%%%%%%%%
% % Plot Results
% %%%%%%%%%%%%%%
% 
% figure(1);
% hold on;
% 
% gscatter(X(:,1), X(:,2), Y, 'rb', '.+');
% 
% graph_line(W, b, m, 'b-', 'b--');
% 
% X_all = X(alpha > 1e-5, :);
% plt = scatter(X_all(:,1), X_all(:,2), 'c', 'o');
% 
% grid on;
% axis([-4 8 -4 8])
% xlabel('x');
% ylabel('y');
% legend("-1", "+1", "decision hyperplane", "margin", "margin", "support vectors")
% hold off;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 2. Multiclass Soft-margin SVM
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 1 - polynomial
% 2 - gaussian
kernel = 2;

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

unique_classes = unique(Y);

C0 = [1e-2 1e-1 1 1e1 1e2 1e3 1e4];
if kernel == 1 % polynomial
    parameters = [1 2 3 4 5];
else           % gaussian
    parameters = [1e-2 1e-1 1 1e1 1e2 1e3];
end

Y_pred_all = zeros(length(unique(Y)), l);

% Create a classifier for each class
for cl_i = 1:length(unique(Y))
    class = unique_classes(cl_i);
    % Make true-classes: +1 and false-classes: -1
    Y_class = ((Y == class) * 2) - 1; 
    
    numErr_best = inf;
    C_best = C0(1);
    param_best = parameters(1);

    for C_i = 1:length(C0)
        C = C0(C_i);
        for p_i = 1:length(parameters)
            param = parameters(p_i);

            % 5-fold-crossvalidation
            numErr = 0;
            indices = crossvalind('Kfold',Y_class,5);
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
            
            if numErr_best > numErr
                numErr_best = numErr;
                C_best = C;
                param_best = param;
            end
        end
    end
    
    disp("==========================");
    disp("Best Parameters for class:");
    class, param_best, C_best, accuracy = ((l - numErr) / l) * 100
    
    % Using the best attributes, create Y_pred
    Y_pred_all(cl_i, :) = soft_margin_svm(X, Y_class, X, Y_class, kernel, param_best, C_best);
end


[~, ind] = max(Y_pred_all, [], 1);
ind = unique_classes(ind);
numErr = length(find((ind - Y) ~= 0));

percentErr = 100 * (numErr / length(Y));

"3. Accuracy of SVM with best parameters:"
accuracy = ((l - numErr) / l) * 100

toc

function Y_pred = soft_margin_svm(X_train, Y_train, X_test, Y_test, kernel, param, C)
    [l, ~] = size(X_train);
    
    if kernel == 1 % polynomial
        K = ((X_train*X_train')+1) .^ param;
    else           % gaussian
        K = grbf_fast(X_train, X_train, param);
    end
    H = (Y_train * Y_train') .* K;
    H = H + eye(l)*1e-7;

    P = -ones(size(Y_train)); % negative because matlab by default minimizes quadprod, but we want a maximization
    Aeq = Y_train';
    beq = 0;

    % See 2.16c
    lb = zeros(size(Y_train));
    ub = C * ones(size(Y_train));

    options = optimset('maxIter',1e6,'LargeScale','off','Display','off');
    alpha = quadprog(H, P, [], [], Aeq, beq, lb, ub, lb, options);
%     alpha = quadprog(H, P, [], [], [], [], lb, ub, lb, options);

    eps = 1e-5;
    ind_free = find(alpha >= eps & alpha <= C - eps);
    ind_all = find(alpha >= eps);

    X_free = X_train(ind_free,:);
    Y_free = Y_train(ind_free);
    X_all = X_train(ind_all,:);
    Y_all = Y_train(ind_all);
    alpha_all = alpha(ind_all);
    
    % Calculate bias
    if kernel == 1 % polynomial
        K = ((X_free*X_all')+1) .^ param;
    else           % gaussian
        K = grbf_fast(X_free, X_all, param);
    end
    b = mean(Y_free - sum(K' .* alpha_all .* Y_all)');

    % Calculate Y_pred for X_test
    if kernel == 1 % polynomial
        K = ((X_test*X_train')+1) .^ param;
    else           % gaussian
        K = grbf_fast(X_test, X_train, param);
    end
    Y_pred = sum(Y_train .* alpha .* K')' + b;
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