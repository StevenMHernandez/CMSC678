close all, format compact, clear all;

% rng(1,'v4normal')

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 1. Linear Hard-margin SVM
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('P2_data.mat');

[l, dim] = size(X); % TODO: remove and size(Y), replace with simply `l`.

% See 2.16a from Springer book
H = (Y*Y').*(X*X');
H = H + eye(l)*1e-7; % cond(H) = 9.5629e+20, so this is badly conditioned
P = -ones(size(Y)); % negative because matlab by default minimizes quadprod, but we want a maximization

% See 2.16b from Springer book
Aeq = Y'; % TODO
beq = 0;
% See 2.16c
lb = zeros(size(Y));
ub = inf * ones(size(Y));

alpha = quadprog(H, P, [], [], Aeq, beq, lb, ub);

W = get_weight(alpha, Y, X);
b = get_bias(Y(alpha > 1e-5), X(alpha > 1e-5, :), W);
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
grid on;
axis([-4 8 -4 8])
xlabel('x');
ylabel('y');
plt = gscatter(X(:,1), X(:,2), Y, 'rb', '.+');
graph_line(W, b, m, 'b-', 'b--');


X_support_vectors = X(alpha > 1e-5, :);
plt = scatter(X_support_vectors(:,1), X_support_vectors(:,2), 'c', 'o');



legend("-1", "+1", "decision hyperplane", "margin", "margin", "support vectors")

ax = gca;
ax.XAxisLocation = 'origin';
ax.YAxisLocation = 'origin';
hold off;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 2. Multiclass Soft-margin SVM
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[Y, X] = libsvmread('glass'); X=full(X);

[l, dim] = size(X);

% Scale each dimention
for i = [1:dim]
    X(:,i) = (X(:,i) - mean(X(:,i))) / std(X(:,i));
end

% Shuffle
[~, indices] = sort(rand(l,1));
X = X(indices, :);
Y = Y(indices);

% % Preview the relationship of the data
% % WARNING, this will create a dim*dim number of plot windows!
% for i = [1:dim]
%     for j = [1:dim]
%         figure((i*dim)+j)
%         plt = gscatter(X(:,i), X(:,j), Y);
%     end
% end
% return

C0 = [1e-2 1e-1 1 1e1 1e2 1e3 1e4];
parameters_polynomial = [1 2 3 4 5]; % for polynomial kernel classifier
parameters_guassian = [1e-2 1e-1 1 1e1 1e2 1e3]; %for Gaussian (i.e. RBF) kernel classifier

% labels for graph
label_strings = strings(length(C0) * 2, 1);
labels_i = 1;


% C0*poly+C0*guassian
% 7*5+7*

err_polynomial = zeros(size(parameters_polynomial));
err_guassian = zeros(size(parameters_guassian));

figure(2)
hold on

for kernel = [1] % 1: polynomial, 2: guassian % TODO: add guassian
    for C = C0
        if kernel == 1 
            % polynomial
            parameters = parameters_polynomial;
        else
            % guassian
            parameters = parameters_guassian;
        end

        % index for tracking which parameter # we are using
        p_i = 1;

        for param = parameters
            numErr = 0;

            % Store all Y_pred (before applying sign()) so that we then use
            % max() to determing the prediction for the model
            Y_pred_all = zeros(l,length(unique(Y)'));

            % Create a classifier for each class
            for c = unique(Y)'
                Yc = ((Y==c)*2)-1; % Ugly way to make true-classes: +1 and false-classes: -1, TODO: find a better method for this

                if kernel == 1 % polynomial
                    H = (Yc * Yc') .* (((X * X') + 1) .^ param);
                else
    %               % TODO
                end

                H = H + eye(l)*1e-7; % TODO: check if this is badly conditioned

                P = -ones(size(Yc)); % negative because matlab by default minimizes quadprod, but we want a maximization

                Aeq = Yc'; % TODO
                beq = 0;

                % See 2.16c
                lb = zeros(size(Yc));
                ub = C * ones(size(Yc));

                alpha = quadprog(H, P, [], [], Aeq, beq, lb, ub);
%                 alpha = quadprog(H, P, [], [], [], [], lb, ub);

                e = 1e-5;
                ind_Free = find(alpha >= e & alpha <= C - e);

                W = get_weight(alpha, Yc, X);
                b = get_bias(Yc(ind_Free), X(ind_Free, :), W);

                M = 1/norm(W);
                
                
%                 Y_pred = (W'*((X' + 1) .^ 2))' + b;

                if kernel == 1 
                    % polynomial
%                     Y_pred = W'*X'+b;
%                     W'*(((X * X') + 1) .^ param);
                    Y_pred = (W'*((X' + 1) .^ 2))' + b;
%                     Y_pred = (dot(Y,alpha)*(X' + 1) .^ 2)'+b;
                else
                    % guassian
                    % TODO
                end

                Y_pred_all(:,c) = Y_pred';

                Y_pred_o = sign(Y_pred);
            end

            [~, indices] = max(Y_pred_all'); length(Y); length(find(i'-Y));


            if kernel == 1 
                % polynomial
                err_polynomial(p_i) = length(find(indices'-Y));
            else
                % guassian
                err_guassian(p_i) = length(find(indices'-Y));
            end
            
            % TODO: don't worry about this

            p_i = p_i + 1;
        end
        
        label_strings(labels_i) = "Polynomial, C=" + C;
        labels_i = labels_i + 1;
        plot(parameters_polynomial, err_polynomial)
    end
end

legend(label_strings)
grid on;
title("Number of Misclassified Points per Polynomial Parameter Used")
xlabel("Polynomial Parameter Used")
ylabel("Number of misclassified points")
hold off


%%%%%%%%%%%%%%
% 
% Sub Routines
% 
%%%%%%%%%%%%%%

% 2.17a
function w = get_weight(alpha, Y, X)
    % TODO: do this in one line (matrix multiplication)?
    w = 0;
    for i = 1:size(alpha,1)
        w = w + (alpha(i) * Y(i) * X(i,:)');
    end
end

% 2.17b
function b = get_bias(Y, X, W)
    b = 0;
    count = 0;
    for i = 1:size(X,1)
        b = b + (1/Y(i))-(X(i,:)*W);
        count = count + 1;
    end
    
    b = b / count;
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
    % TODO: figure out, and name
    y_diff = m * sqrt(slope^2 + 1);

    plt = plot(x_matrix, y_matrix, line_type);
    plt = plot(x_matrix, y_matrix + y_diff, line_type_2);
    plt = plot(x_matrix, y_matrix - y_diff, line_type_2);
end