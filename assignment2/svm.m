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
parameters_guassian = [1e-2 1e-1 1 1e1 1e2 1e3]; % for Gaussian (i.e. RBF) kernel classifier

% labels for graph
label_strings = strings(length(C0) * 2, 1);
labels_i = 1;

% store errors and accuracy values for plotting later
accuracy_polynomial = zeros(size(parameters_polynomial));
accuracy_guassian = zeros(size(parameters_guassian));

C_best_polynomial = C0(1);
C_best_guassian = C0(1);
degree_best_polynomial = parameters_polynomial(1);
sigma_best_guassian = parameters_guassian(1);

lowest_polynomial_error = inf;
lowest_guassian_error = inf;

% Hold onto all figures for plotting accuracy and # of errors
for i = 2:3
    figure(i)
    hold on
end

for kernel = 1:2 % 1: polynomial, 2: guassian
    for C = C0
        if kernel == 1 
            % polynomial
            parameters = parameters_polynomial;
        else
            % guassian
            parameters = parameters_guassian;
        end

        for p_i = 1:length(parameters)
            param = parameters(p_i);
            
            % 5-fold-crossvalidation
            numErr = 0;
            indices = crossvalind('Kfold',Y,5);
            for i = 1:5
                ind_test = (indices == i);
                ind_train = ~ind_test;

                X_train = X(ind_train, :);
                Y_train = Y(ind_train, :);
                X_test = X(ind_test, :);
                Y_test = Y(ind_test, :);

                % Store all Y_pred (before applying sign()) so that we then use
                % max() to determing the prediction for the model
                [l_test, ~] = size(X_test);
                Y_pred_all = zeros(l_test,length(unique(Y_test)'));
                [l, dim] = size(X_train);

                % Create a classifier for each class
                for class = unique(Y)'
                    % Make true-classes: +1 and false-classes: -1
                    Y_train_class = ((Y_train == class) * 2) - 1; 
                    Y_test_class = ((Y_test == class) * 2) - 1; 

                    if kernel == 1 % polynomial
                        H = (Y_train_class * Y_train_class') .* (((X_train * X_train') + 1) .^ param);
                    else           % guassian
                        H = (Y_train_class * Y_train_class') .* grbf_fast(X_train,X_train,param);
                    end

                    H = H + eye(l)*1e-3; % TODO: check if this is badly conditioned

                    P = -ones(size(Y_train_class)); % negative because matlab by default minimizes quadprod, but we want a maximization
                    Aeq = Y_train_class';
                    beq = 0;

                    % See 2.16c
                    lb = zeros(size(Y_train_class));
                    ub = C * ones(size(Y_train_class));

                    alpha = quadprog(H, P, [], [], Aeq, beq, lb, ub, [], options);
%                     alpha = quadprog(H, P, [], [], [], [], lb, ub, [], options);

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
                                sum = sum + (((X_train(j,:) * X_train(i,:)') + 1) .^ param);
                            else           % guassian
                                sum = sum + grbf_fast(X_train(j,:),X_train(i,:),param);
                            end
                        end
                        b = Y_train_class(j) + sum;
                    end
                    b = (1/length(ind_Free)) * b;

                    % Calculate Y_pred for X_test
                    Y_pred = zeros(size(Y_test));
                    for j = 1:l_test
                        for i = ind_Support_Vectors'
                            if kernel == 1 % polynomial
                                Y_pred(j) = Y_pred(j) + (((X_test(j,:) * X_train(i,:)') + 1) .^ param);
                            else           % guassian
                                Y_pred(j) = Y_pred(j) + grbf_fast(X_test(j,:),X_train(i,:),param);
                            end
                        end
                        Y_pred(j) = Y_pred(j) + b;
                    end
                    Y_pred_all(:,class) = Y_pred';
                end

                [~, ind] = max(Y_pred_all, [], 2);
                
                numErr = numErr + length(find(ind-Y_test));
            end

            totalErr = 100 * (numErr / length(Y));
            
            % Store num_errors and percent_accuracy for plotting
            if kernel == 1 
                accuracy_polynomial(p_i) = totalErr;
            else
                accuracy_guassian(p_i) = totalErr;
            end
        end

            % Plot num_errors and percent_accuracy
            if kernel == 1 
                label_strings(labels_i) = "Polynomial, C=" + C;
                labels_i = labels_i + 1;

                figure(2);
                plot(parameters_polynomial, accuracy_polynomial);
            else
                label_strings(labels_i) = "Guassian, C=" + C;
                labels_i = labels_i + 1;

                figure(3);
                plot(parameters_guassian, accuracy_guassian);
            end
    end

    % Complete num_errors and percent_accuracy plots
    if kernel == 1 
        figure(2)
        legend(label_strings(1:length(label_strings)/2));
        title("Percentage of Incorrectly Classified Elements through Crossvalidation")
        xlabel("Polynomial Parameter Used")
        ylabel("Percentage Error")
    else
        figure(3)
        legend(label_strings(length(label_strings)/2 + 1:length(label_strings)));
        title("Percentage of Incorrectly Classified Elements through Crossvalidation")
        xlabel("Guassian Parameter (Sigma) Used")
        ylabel("Percentage Error")
        set(gca, 'XScale', 'log')
    end      
end

hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create a model with the best parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% polynomial SVM C=0.01, d=2
soft_margin_SVM(X, Y, 1, 0.01, 2)

% Gaussian SVM C=0.1, sigma=1
soft_margin_SVM(X, Y, 2, 0.1, 1)

toc;

%%%%%%%%%%%%%%
% 
% Sub Routines
% 
%%%%%%%%%%%%%%

function percentErr = soft_margin_SVM(X, Y, kernel, C, param)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % soft_margin_SVM() trains the SVM on all data points X and Y 
    % returning the percent Err resulting from the parameters 
    % kernel, C and param.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    [l, ~] = size(X);
    Y_pred_all = zeros(l,length(unique(Y)')); % apply max() to this later on.

    % Create a classifier for each class
    for class = unique(Y)'
        % Make true-classes: +1 and false-classes: -1
        Y_class = ((Y == class) * 2) - 1;

        if kernel == 1 % polynomial
            H = (Y_class * Y_class') .* (((X * X') + 1) .^ param);
        else           % guassian
            H = (Y_class * Y_class') .* grbf_fast(X,X,param);
        end

        H = H + eye(l)*1e-7; % TODO: check if this is badly conditioned
        P = -ones(size(Y)); % negative because matlab by default minimizes quadprod, but we want a maximization
        Aeq = Y_class';
        beq = 0;
        % See 2.16c
        lb = zeros(size(Y));
        ub = C * ones(size(Y));

        options = optimset('maxIter',1e6,'LargeScale','off','Display','off');
        alpha = quadprog(H, P, [], [], Aeq, beq, lb, ub, [], options);
%         alpha = quadprog(H, P, [], [], [], [], lb, ub, [], options);

        e = 1e-5;
        ind_Free = find(alpha >= e & alpha <= C - e);
        ind_Support_Vectors = find(alpha >= e);

        % Figure out the bias from the slideshow p.162/209
        b = 0;
        for j = ind_Free'
            sum = 0;
            for i = ind_Support_Vectors'
                if kernel == 1 % polynomial
                    sum = sum + (((X(j,:) * X(i,:)') + 1) .^ param);
                else           % guassian
                    sum = sum + grbf_fast(X(j,:),X(i,:),param);
                end
            end
            b = Y(j) + sum;
        end
        b = (1/length(ind_Free)) * b;

        % Calculate Y_pred for X_test
        Y_pred = zeros(size(Y));
        for j = 1:l
            for i = ind_Support_Vectors'
                if kernel == 1 % polynomial
                    Y_pred(j) = Y_pred(j) + (((X(j,:) * X(i,:)') + 1) .^ param);
                else           % guassian
                    Y_pred(j) = Y_pred(j) + grbf_fast(X(j,:),X(i,:),param);
                end
            end
            Y_pred(j) = Y_pred(j) + b;
        end
        Y_pred_all(:,class) = Y_pred';
    end

    [~, ind] = max(Y_pred_all, [], 2);

    numErr = length(find(ind-Y));
    percentErr = 100 * (numErr / length(Y));
end

% 2.17a
function W = get_weight(alpha, Y, X)
end

% 2.17b
function b = get_bias(Y, X, W)
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