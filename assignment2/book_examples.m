close all, format compact, clear all;

svm;

% Testing example 
% TODO: remove

X = [-2;-1;0];
Y = [-1;-1;1];
H = get_hessian_matrix(X,Y)
P = -ones(size(Y));

% See 2.16b from Springer book
Aeq = Y'; % TODO
beq = 0;

% See 2.16c
lb = zeros(size(Y));
ub = inf * ones(size(Y));

alpha = quadprog(H, P, [], [], Aeq, beq, lb, ub)
W = get_weight(alpha, Y, X)
B = get_bias(alpha, Y, X, W)

"Testing the example from the book"
W + " should = 2"
B + " should = 1"



% % Testing example from Fig. 2.10 from Springer book
% % TODO: remove
% 
% X = [
%     % negatives
%     1 5;
%     2 1;
%     2 6;
%     3.5 6;
%     4 5;
%     4 1;
%     5 3;
%     7 3.5;
%     11 7; % outlier!
%     % positives
%     2 9;
%     5 9;
%     5.5 6.5;
%     6 11;
%     8 11; 
%     8 7;
%     9 9;
%     9 10;
%     10 7;
%     11 11;
%     12 9;
% ];
% Y = cat(1, -1*ones(9, 1), 1*ones(11, 1));
% 
% 
% 
% figure(99);
% 
% % for C = [1e-2 1e-1 1 1e1 1e2 1e3 1e4]
% for C = [2]
%     H = get_hessian_matrix(X,Y)
%     P = -ones(size(Y));
% 
%     % See 2.16b from Springer book
%     Aeq = Y'; % TODO
%     beq = 0;
% 
%     % See 2.16c
%     lb = zeros(size(Y));
%     ub = C * ones(size(Y));
% 
%     alpha = quadprog(H, P, [], [], Aeq, beq, lb, ub)
%     W = get_weight(alpha, Y, X)
%     B = get_bias(alpha, Y, X, W)
% 
%     M = 1/norm(W)
% 
%     %%%%%%%%%%%%%%
%     % Plot Results
%     %%%%%%%%%%%%%%
% 
%     range = -3:1;
%     hold on;
%     grid on;
%     axis([-1 13 -1 13])
%     xlabel('x');
%     ylabel('y');
%     plt = gscatter(X(:,1), X(:,2), Y, 'rb', '.+');
%     graph_line(W, B, 'b-');
% 
% 
%     X_support_vectors = X(find(alpha > 0.000001), :);
%     plt = scatter(X_support_vectors(:,1), X_support_vectors(:,2), 'c', 'o');
% 
%     graph_line(W, B-M, 'b--');
%     graph_line(W, B+M, 'b--');
% 
%     legend("-1", "+1", "decision hyperplane", "support vectors")
% 
%     ax = gca;
%     ax.XAxisLocation = 'origin';
%     ax.YAxisLocation = 'origin';
% end
% hold off;
% 
% "ok"

% %%%%%%
% % Test figure 2.12 from the book
% %%%%%%
% X = [-1 0 1]';
% Y = [-1 1 -1]';
% 
% H = get_hessian_matrix(X,Y)
% P = -ones(size(Y));
% 
% % See 2.16b from Springer book
% Aeq = Y'; % TODO
% beq = 0;
% 
% % See 2.16c
% C = 1.0
% lb = zeros(size(Y));
% ub = C * ones(size(Y));
% 
% alpha = quadprog(H, P, [], [], Aeq, beq, lb, ub)
% W = get_weight(alpha, Y, X)
% B = get_bias(alpha, Y, X, W)
% 
% %     "Testing the example from the book"
% 
% M = 1/norm(W)
% M = M/2
% 
% %%%%%%%%%%%%%%
% % Plot Results
% %%%%%%%%%%%%%%
% 
% range = -3:1;
% hold on;
% grid on;
% axis([-1 13 -1 13])
% xlabel('x');
% ylabel('y');
% gscatter(X(:,1), X(:,2), Y, 'rb', '.+');
% graph_line(W, B, '-');
% 
% 
% X_support_vectors = X(find(alpha > 0.000001), :);
% plt = scatter(X_support_vectors(:,1), X_support_vectors(:,2), 'c', 'o');
% 
% graph_line(W, B-M, 'c--');
% graph_line(W, B+M, 'c--');
% 
% legend("-1", "+1", "decision hyperplane", "support vectors")
% 
% ax = gca;
% ax.XAxisLocation = 'origin';
% ax.YAxisLocation = 'origin';
% hold off;

% % Testing example from Fig. 2.14 from Springer book
% % TODO: remove
% 
% X = [1 2 5 6]';
% Y = [-1 -1 1 -1]';
% 
% % for C = [1e-2 1e-1 1 1e1 1e2 1e3 1e4]
% for C = [inf]
%     H = get_hessian_matrix(X,Y)
%     P = -ones(size(Y));
% 
%     % See 2.16b from Springer book
%     Aeq = Y'; % TODO
%     beq = 0;
% 
%     % See 2.16c
%     lb = zeros(size(Y));
%     ub = C * ones(size(Y));
% 
%     alpha = quadprog(H, P, [], [], Aeq, beq, lb, ub)
% %     alpha = quadprog(H, P, [], [], [], [], lb, ub) % Get rid of the equality constraint as specified in page 45
%     W = get_weight(alpha, Y, X)
%     B = get_bias(alpha, Y, X, W)
% 
%     M = 1/norm(W)
% 
%     %%%%%%%%%%%%%%
%     % Plot Results
%     %%%%%%%%%%%%%%
% 
%     range = -3:1;
%     hold on;
%     grid on;
%     axis([-1 13 -1 13])
%     xlabel('x');
%     ylabel('y');
%     plt = gscatter(X(:,1), X(:,2), Y, 'rb', '.+');
%     graph_line(W, B, 'b-');
% 
% 
%     X_support_vectors = X(find(alpha > 0.000001), :);
%     plt = scatter(X_support_vectors(:,1), X_support_vectors(:,2), 'c', 'o');
% 
%     graph_line(W, B-M, 'b--');
%     graph_line(W, B+M, 'b--');
% 
%     legend("-1", "+1", "decision hyperplane", "support vectors")
% 
%     ax = gca;
%     ax.XAxisLocation = 'origin';
%     ax.YAxisLocation = 'origin';
% end
% hold off;
% 
% "ok"


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 1. Linear Hard-Margin SVM Functions
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 2.16a
function H = get_hessian_matrix(X, Y)
%     TODO: figure out which method is correct darrn you!
% 


    [l dim] = size(X);
    H = zeros(l, l);
    Y_s = zeros(l,l);
    X_s = zeros(l,l);
    
    for i = 1:l
        
        for j = 1:l
            H(i,j) = Y(i) * Y(j) * (X(i,:) * X(j,:)' + 1)^2;
            Y_s(i,j) = Y(i) * Y(j);
            X_s(i,j) = (X(i,:) *X(j,:)' + 1)^2;
        end
    end
    
    H_first = H
    X_first = X_s
    Y_first = Y_s

    H = (Y*Y').*(((X*X') + 1).^2);
    
    H_second = H
    X_first = (((X*X') + 1).^2)
    Y_first = (Y*Y')
    
    "ok"

%     [l dim] = size(X);
%     H = zeros(l, l);
%     
%     for i = 1:l
%         
%         for j = 1:l
%             H(i,j) = Y(i) * Y(j)* X(i,:) *X(j,:)';
%         end
%     end
    
%     H = (Y*Y').*(X*X');
    
%     if cond(H) == inf % if we have a poorly conditioned matrix, regularize it
%         c = 0.00000001;
%         H = H + (c*eye(size(H,1)));
%     end
end

% 2.17a
function w = get_weight(alpha, Y, X)
    % TODO: do this in one line (matrix multiplications
    w = 0;
    for i = 1:size(alpha,1)
        w = w + (alpha(i) * Y(i) * X(i,:));
    end
    
    w = w' % TODO: I don't even know
end

% 2.17b
function b = get_bias(alpha, Y, X, W)
%     TODO: can we just use the find() subroutine?
    b = 0;
    count = 0;
    for i = 1:size(X,1)
        if alpha(i) > 0.000001 % only handle support vectors
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

function plt = graph_line(W, B, line_type)
    x_intercept = -(B/W(1));
    y_intercept = -(B/W(2));
    slope = -(B/W(2))/(B/W(1));

    x_matrix = -10:20;
    y_matrix = y_intercept + (slope * x_matrix);

    plt = plot(x_matrix, y_matrix, line_type);
end