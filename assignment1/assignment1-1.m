close all, format compact

% set seed
seed = 10;
rng(seed);

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

% Add a negative outlier
X = cat(1, X, [20, 20, 1.0]);
y = cat(1, y, -1);

% % Render Original Graph
% figure(1)
% hold on
% graph(X, y)
% title("Original graph")
% hold off

% Start Rendering 
figure(2)
hold on
graph(X, y)

X = X';
y = y';

learning_rates = [1];
% learning_rates = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4];
learning_rate_epochs = [];
final_weights = []

for learning_rate = learning_rates
    w = [0 0 0];
%     w = [-10 -10 -10];
    e = Inf;
    epoch = 0;

    while ((e > 0) && (epoch < 500))
        epoch = epoch + 1;
        for i = 1:size(X,2)
            
%             epoch = epoch + 1
            w = learn(X(:,i), y(:,i), w, learning_rate);

            % Check if learning is now complete.
            e = 0;

            for i = 1:size(X,2)
                e = e + (error(y(:,i), output(X(:,i), w)))^2;
            end

            e = sum(e); 

            if e == 0
                learning_rate_epochs = [learning_rate_epochs, epoch];
                final_weights = cat(1, final_weights, w);
                break;
            end
        end

        % Plot each learned line
        graph_line(w)
    end
end

learning_rates
learning_rate_epochs
final_weights

final_number_of_epoch = epoch

title("Weight changes during learning")
hold off

% Plot the predicted values

predictions = [];

for i = 1:size(X,2) 
    predictions = [predictions, output(X(:,i), w)];
end


figure(3)
hold on
title("Final Predictions")
graph(X', y)
graph_line(w)
hold off

final_weights = w

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Perceptron Learning Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function o = output(x, w)
    o = sign(w*x);
end

function e = error(d, o)
    e = d - o;
end

function w = learn(x, d, w, learning_rate)
    change = learning_rate * ((error(d, output(x, w))) * x');
    w = w + change;
    learning_rate
    change
    w
end

%%%%%%%%%%%%%%%%%%%%
% Plotting functions
%%%%%%%%%%%%%%%%%%%%

function plt = graph(X,y)
    axis([-5 21 -5 21])
    gscatter(X(:,1), X(:,2), y, 'rb', 'o+')
    xlabel('x');
    ylabel('y');
end

function plt = graph_line(w)
    x_intercept = -(w(3)/w(1));
    y_intercept = -(w(3)/w(2));
    slope = -(w(3)/w(2))/(w(3)/w(1));

    x_matrix = -10:10;
    y_matrix = y_intercept + (slope * x_matrix);

    plot(x_matrix,y_matrix);
end