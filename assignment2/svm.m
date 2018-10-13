close all, format compact, clear all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 1. Linear Hard-margin SVM
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('P2_data.mat');

[l, dim] = size(X);

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
b = get_bias(alpha, Y, X, W);
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

range = -3:1;
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

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % 
% % 2. Multiclass Soft-margin SVM
% % 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% [Y, X]=libsvmread('glass'); X=full(X);
% 
% % Create a classifier for each class
% for c = unique(Y)'
%     Y_class = ((Y==c)*2)-1; % Ugly way to make true-classes: +1 and false-classes: -1, TODO: find a better method for this
% 
%     % See 2.16a from Springer book
%     H = get_hessian_matrix(X,Y_class);
%     P = -ones(size(Y_class)); % negative because matlab by default minimizes quadprod, but we want a maximization
% 
%     % See 2.16b from Springer book
%     Aeq = Y_class'; % TODO
%     beq = 0;
% 
% 
%     C = 1e-4; % TODO
% 
%     % See 2.16c
%     lb = zeros(size(Y_class));
%     ub = C * ones(size(Y_class));
% 
%     alpha = quadprog(H, P, [], [], Aeq, beq, lb, ub)
% 
%     W = get_weight(alpha, Y_class, X)
%     B = get_bias(alpha, Y_class, X, W)
% 
%     M = 1/norm(W)
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 1. Linear Hard-Margin SVM Functions
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 2.16a
% function H = get_hessian_matrix(X, Y)
% %     TODO: figure out which method is correct darrn you!
% % 
% 
% 
%     [l dim] = size(X);
%     H = zeros(l, l);
%     Y_s = zeros(l,l);
%     X_s = zeros(l,l);
%     
%     for i = 1:l
%         
%         for j = 1:l
%             H(i,j) = Y(i) * Y(j) * (X(i,:) * X(j,:)' + 1)^2;
%             Y_s(i,j) = Y(i) * Y(j);
%             X_s(i,j) = (X(i,:) *X(j,:)' + 1)^2;
%         end
%     end
%     
%     H_first = H
%     X_first = X_s
%     Y_first = Y_s
% 
%     H = (Y*Y').*(((X*X') + 1).^2);
%     
%     H_second = H
%     X_first = (((X*X') + 1).^2)
%     Y_first = (Y*Y')
%     
%     "ok"
% 
% %     [l dim] = size(X);
% %     H = zeros(l, l);
% %     
% %     for i = 1:l
% %         
% %         for j = 1:l
% %             H(i,j) = Y(i) * Y(j)* X(i,:) *X(j,:)';
% %         end
% %     end
%     
% %     H = (Y*Y').*(X*X');
%     
% %     if cond(H) == inf % if we have a poorly conditioned matrix, regularize it
% %         c = 1e-7;
% %         H = H + (c*eye(size(H,1)));
% %     end
% end

% 2.17a
function w = get_weight(alpha, Y, X)
    % TODO: do this in one line (matrix multiplications
    w = [0 0]';
    for i = 1:size(alpha,1)
        w = w + (alpha(i) * Y(i) * X(i,:)');
    end
end

% 2.17b
function b = get_bias(alpha, Y, X, W)
%     alpha = alpha(alpha > 1e-5)

    b = 0;
    count = 0;
    for i = 1:size(X,1)
        if alpha(i) > 1e-5 % only handle support vectors
            b = b + (1/Y(i))-(X(i,:)*W);
            count = count + 1;
        end
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