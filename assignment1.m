% set seed
% rng(1)

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

gscatter(X(:,1), X(:,2), y, 'rb', 'o+')
hold on
xlabel('x');
ylabel('y');

w = [0 0 0]';

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
            "test";
            X(:,i);
            y(:,i);
            e = e + error(X(:,i), y(:,i));
        end

        e = sum(e);

        if sum(e) == 0
            break;
        end
    end

    % Plot it A BUNCH!
    
    % 0 = w(1)*x + w(2)*y + w(3)*1
    % -w(1)*x =  w(2)*y + w(3)*1
    % x = -((w(2)*y)/w(1)) - (w(3)/w(1))

    % -w(2)*y = w(1)*x + w(3)*1
    % y = -(w(1)*x)/w(2)) - (w(3)/w(2))

    slope = -((w(3)/w(1))/(w(2)/w(1)));
    intercept = -(w(3)/w(1));

    x_matrix = -10:10;
    y_matrix = intercept + slope*x_matrix;

    plot(x_matrix,y_matrix);
end

axis([-5 10 -5 10])

hold off

w

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Perceptron Learning Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function o = output(x, w)
    o = sign(w'*x);
end

function e = error(d, o)
    e = d - o;
end

function w = learn(x, d, w)
    actual_output = output(x, w);
    err = error(d, actual_output);

    learn_rate = 0.1;
    w = w + (learn_rate * err * x);
end
