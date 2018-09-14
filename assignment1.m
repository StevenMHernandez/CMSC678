% set seed
seed = 10;
rng(seed);

learning_rate = 0.1;

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

% Render Original Graph
figure(1)
hold on
graph(X, y)
title("Original graph")
hold off

% Start Rendering 
% figure(2)
% hold on
% graph(X, y)

w = [0 0 0];

X = X';
y = y';

e = Inf;
epoch = 0;

while ((e > 0) && (epoch < 500))
    epoch = epoch + 1;
    for i = 1:size(X,2)
        w = learn(X(:,i), y(:,i), w);
        
        % Check if learning is now complete.
        e = 0;

        for i = 1:size(X,2)
            e = e + (error(y(:,i), output(X(:,i), w)))^2;
        end

        e = sum(e);

        if sum(e) == 0
            break;
        end
    end

    % Plot each learned line
%     graph_line(w)
end

final_number_of_epoch = epoch

% title("Weight changes during learning")
% hold off

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

% figure(2)
% hold on
% X = X';
% gscatter(X(:,1), X(:,2), predictions, 'rb', 'o+')
% hold off

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

function w = learn(x, d, w)
    change = learning_rate * (error(d, output(x, w))) * x';
    w = w + change;
end

function plt = graph(X,y)
    axis([-5 10 -5 10])
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