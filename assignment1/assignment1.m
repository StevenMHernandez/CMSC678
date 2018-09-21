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
rng(1,'v4normal')

%%%%%%%%%%%%%%%%%%%%%
% Create Base Dataset
%%%%%%%%%%%%%%%%%%%%%

X = cat(1, normrnd(0,2,20,2), normrnd(5,2,10,2));
y = cat(1, repmat(1, [20 1]), repmat(-1, [10 1]));

% Add bias
X = [X ones(size(X(:, 1)))];

%%%%%%%%%%%%%%%%%%%%%%%%
% Add a negative outlier
%%%%%%%%%%%%%%%%%%%%%%%%

X_with_outlier = cat(1, [20, 20, 1.0], X);
y_with_outlier = cat(1, -1.0, y);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set matrix into correct orientation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

X = X';
y = y';
X_with_outlier = X_with_outlier';
y_with_outlier = y_with_outlier';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Train by default algorithm with ?=0.1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tuple = run_learning_algorithm(X, y, 0.1, [0 0 0]);
epochs_taken = tuple(1);
w = tuple(2:size(tuple, 2));

"===="
"1.1.1: What are the final weight and # of epochs when eta=0.1"
"final weights with eta=0.1"
w
"number of epochs taken"
epochs_taken
"----"

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Train by default algorithm for different learning rates
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

learning_rates = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4];
learning_rate_epochs = [];
final_weights = [];

learning_rate_epochs_with_outlier = [];
final_weights_with_outlier = [];

for learning_rate = learning_rates
    %%%%%%%%%%%%%%%%%
    % without outlier
    %%%%%%%%%%%%%%%%%

    tuple = run_learning_algorithm(X, y, learning_rate, [1 1 1]);
    epochs_taken = tuple(1);
    final_weight_vector = tuple(2:size(tuple, 2));
    
    learning_rate_epochs = cat(1, learning_rate_epochs, epochs_taken);
    final_weights = cat(1, final_weights, final_weight_vector);

    %%%%%%%%%%%%%%
    % with outlier
    %%%%%%%%%%%%%%

    tuple = run_learning_algorithm(X_with_outlier, y_with_outlier, learning_rate, [1 1 1]);
    epochs_taken = tuple(1);
    final_weight_vector = tuple(2:size(tuple, 2));
    
    learning_rate_epochs_with_outlier = cat(1, learning_rate_epochs_with_outlier, epochs_taken);
    final_weights_with_outlier = cat(1, final_weights_with_outlier, final_weight_vector);
end

% Note: final_weights(6, :) is when learning_rate is 1.0

"===="
"2.1.1: What are the weights of psuedoinverse learning with and without outlier."
final_weights(6, :)
final_weights_with_outlier(6, :)
"----"

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Graph Results of Training
%%%%%%%%%%%%%%%%%%%%%%%%%%%

w = final_weights(6, :);
w_with_outlier = final_weights_with_outlier(6, :);

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

%%%%%%%%%%%%%%%%%%%%%%%%
% Learn without outliers
%%%%%%%%%%%%%%%%%%%%%%%%

penalty = 1.0;
w = learn_psuedoinverse(X, y, penalty);

%%%%%%%%%%%%%%%%%%%%%
% Learn with outliers
%%%%%%%%%%%%%%%%%%%%%

penalty = 1.0;
w_with_outliers = learn_psuedoinverse(X_with_outlier, y_with_outlier, penalty);

%%%%%%%%%%%%%%%
% Graph Figures
%%%%%%%%%%%%%%%

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

% crossvalind

penalties = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4];
errors = [];

for penalty = penalties
    err = 0;
    for i = 1:10
        % Using the built in crossvalidation

        indices = crossvalind('Kfold',y_with_outlier,10);
        test = (indices == i);
        train = ~test;

        X_train = X_with_outlier(:, train);
        y_train = y_with_outlier(:, train);

        X_test = X_with_outlier(:, test);
        y_test = y_with_outlier(:, test);

        w = learn_psuedoinverse(X_train, y_train, penalty);

        for i = 1:size(X_test,2)
            err = err + error(y_test(:,i), output(X_test(:,i), w))^2;
        end
    end
    
    errors = [errors 100 * (err/(size(y_test, 2) * 2 * 10))];
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Graph effect of penalty on error
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(4)
hold on
title("Effect of penalty on error percentage")
xlabel("Penalty")
set(gca, 'XScale', 'log')
ylabel("Error (%)")
plt = plot(penalties, errors);
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The best penalty appears to be 1e-5
% What is the final weight with ?
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

w = learn_psuedoinverse(X_train, y_train, 1e-5);
"Weight with penalty=1e-5 is:"
w

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
% Perceptron Learning Functions
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function tuple = run_learning_algorithm(X, y, learning_rate, w)
    e = Inf;
    epoch = 0;

    % ensure training doesn't go on forever (more than 1000 epochs)
    while ((e > 0) && (epoch < 1000))
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
            break;
        end
    end
    
    % Return a tuple of (# of epochs taken, weight) for use in graphing
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
end

function w = learn_psuedoinverse(x, d, penalty)
    X = x';
    Y = d;
    % w = (X'X + ?I)^-1 * X'Y
    w = (inv((X'*X) + (penalty * eye(size(X, 2)))) * X' * Y')';
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
    plt = gscatter(X(:,1), X(:,2), y, 'rb', 'o+');
end

function plt = graph_line(w, line_type)
    x_intercept = -(w(3)/w(1));
    y_intercept = -(w(3)/w(2));
    slope = -(w(3)/w(2))/(w(3)/w(1));

    x_matrix = -10:20;
    y_matrix = y_intercept + (slope * x_matrix);

    plt = plot(x_matrix,y_matrix, line_type);
end