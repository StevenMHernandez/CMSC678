% set seed
% rng(1)

X = [];
y = [];

for i = 1:20
    X = cat(1, X, [normrnd(0,2),normrnd(0,2)]);
    y = cat(1, y, 1);
end

for i = 1:10
    X = cat(1, X, [normrnd(5,2),normrnd(5,2)]);
    y = cat(1, y, -1);
end

gscatter(X(:,1), X(:,2), y, 'rb', 'o+')
hold on
xlabel('x');
ylabel('y');
hold off