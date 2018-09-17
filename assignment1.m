% Steven Hernandez
% CMSC 678

close all, format compact

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
% PART 1:
%     Basic Perceptron Learning
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%% 
% General Setup
%%%%%%%%%%%%%%%

% set seed
seed = 10;
rng(seed);

%%%%%%%%%%%%%%%%%%%%%
% Create Base Dataset
%%%%%%%%%%%%%%%%%%%%%

X = [];
y = [];

for i = 1:20
    X = cat(1, X, [normrnd(0,2), normrnd(0,2), 1.0]);
    y = cat(1, y, 1);
end

for i = 1:10
    X = cat(1, X, [normrnd(5,2), normrnd(5,2), 1.0]);
    y = cat(1, y, -1);
end

%%%%%%%%%%%%%%%%%%%%%%%%
% Add a negative outlier
%%%%%%%%%%%%%%%%%%%%%%%%

X_with_outlier = cat(1, X, [20, 20, 1.0]);
y_with_outlier = cat(1, y, -1.0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set matrix into correct orientation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

X = X';
y = y';
X_with_outlier = X_with_outlier';
y_with_outlier = y_with_outlier';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Train by default algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

learning_rates = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4];
learning_rate_epochs = [];
final_weights = [];

learning_rate_epochs_with_outlier = [];
final_weights_with_outlier = [];

for learning_rate = learning_rates
    %%%%%%%%%%%%%%%%%
    % without outlier
    %%%%%%%%%%%%%%%%%

    tuple = run_learning_algorithm(X, y, learning_rate);
    epochs_taken = tuple(1);
    final_weight_vector = tuple(2:size(tuple, 2));
    
    learning_rate_epochs = cat(1, learning_rate_epochs, epochs_taken);
    final_weights = cat(1, final_weights, final_weight_vector);

    %%%%%%%%%%%%%%
    % with outlier
    %%%%%%%%%%%%%%

    tuple = run_learning_algorithm(X_with_outlier, y_with_outlier, learning_rate);
    epochs_taken = tuple(1);
    final_weight_vector = tuple(2:size(tuple, 2));
    
    learning_rate_epochs_with_outlier = cat(1, learning_rate_epochs_with_outlier, epochs_taken);
    final_weights_with_outlier = cat(1, final_weights_with_outlier, final_weight_vector);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Graph Results of Training
%%%%%%%%%%%%%%%%%%%%%%%%%%%

w = final_weights(1, :);
w_with_outlier = final_weights_with_outlier(1, :);

figure(1)
hold on
title("Final Predictions")
plt = graph(X_with_outlier', y_with_outlier);
line1 = graph_line(w, ':');
line2 = graph_line(w_with_outlier, '-');
legend([line1 line2], "weights without outlier", "weight learned with outlier")
% set(findobj(gca, 'Type', 'Line', 'Linestyle', ':'), 'LineWidth', 2);
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Graph effect of Learning Rate on Epoch
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(2)
hold on
title("Effect of Learning Rate on Epoch")
xlabel("Learning Rate")
set(gca, 'XScale', 'log')
ylabel("# of Epochs Taken")
scatter1 = scatter(learning_rates, learning_rate_epochs);
scatter2 = scatter(learning_rates, learning_rate_epochs_with_outlier, '.');
legend([scatter1 scatter2], "epochs taken without outlier", "epochs taken with outlier")
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
% PART 2
%     Learning with Pseudo-inverse
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

penalty = 1.0;
w = learn_psuedoinverse(X, y, penalty);

%%%%%%%%%%%%%%%%%%%%%
% Learn with outliers
%%%%%%%%%%%%%%%%%%%%%

penalty = 1.0;
w_with_outliers = learn_psuedoinverse(X_with_outlier, y_with_outlier, penalty);

figure(3)
hold on
title("Final Predictions with Psuedo Inverse and with Outlier");
plt = graph(X_with_outlier', y_with_outlier);
line1 = graph_line(w, ':');
line2 = graph_line(w_with_outliers, '-');
legend([line1 line2], "weights without outlier", "weight learned with outlier")
set(findobj(gca, 'Type', 'Line', 'Linestyle', ':'), 'LineWidth', 2);
hold off

%%%%%%%%%%%%%%%%%%%%%%
%
%
% PART 3
%     Cross-validation
%
%
%%%%%%%%%%%%%%%%%%%%%%









%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
% Perceptron Learning Functions
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function tuple = run_learning_algorithm(X, y, learning_rate)
    w = [0 0 0];
    e = Inf;
    epoch = 0;

    while ((e > 0) && (epoch < 500))
        epoch = epoch + 1;
        for i = 1:size(X,2)
            w = learn(X(:,i), y(:,i), w, learning_rate);
        end

        % Check if learning is now complete.
        e = 0;

        for i = 1:size(X,2)
            e = e + (error(y(:,i), output(X(:,i), w)))^2;
        end

        e = sum(e); 

        if e == 0
%             learning_rate_epochs = [learning_rate_epochs, epoch];
%             final_weights = cat(1, final_weights, w);
            break;
        end
    end
    
    % Return a tuple of (# of epochs taken, weight)
    tuple = [epoch, w];
end

function o = output(x, w)
    o = sign(w*x);
end

function e = error(d, o)
    e = d - o;
end

function w = learn(x, d, w, learning_rate)
    change = learning_rate * ((error(d, output(x, w))) * x');
    w = w + change;
%     learning_rate
%     change
%     w
end

function w = learn_psuedoinverse(x, d, penalty)
    X = x';
    Y = d;
    % w = (X'X + ?I)-1X'Y
    w = (inv((X'*X) + (penalty * eye(size(X, 2)))) * X' * Y')'
end

%%%%%%%%%%%%%%%%%%%%
% 
% 
% Plotting functions
% 
% 
%%%%%%%%%%%%%%%%%%%%

function plt = graph(X,y)
    axis([-5 21 -5 21])
    xlabel('x');
    ylabel('y');
    plt = gscatter(X(:,1), X(:,2), y, 'rb', 'o+')
end

function plt = graph_line(w, line_type)
    x_intercept = -(w(3)/w(1));
    y_intercept = -(w(3)/w(2));
    slope = -(w(3)/w(2))/(w(3)/w(1));

    x_matrix = -10:20;
    y_matrix = y_intercept + (slope * x_matrix);

    plt = plot(x_matrix,y_matrix, line_type);
end