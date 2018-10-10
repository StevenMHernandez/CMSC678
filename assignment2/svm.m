clear all; close all;

x = [-2 -1 0]';
y = [-1 -1 1]';

% See 2.16a from Springer book
H = get_hessian_matrix(x,y);
p = -ones(size(x)); % negative because matlab by default minimizes quadprod, but we want a maximization

% We have no inequalities to be concerned about
A = [];
b = [];

% See 2.15c/2.16b from Springer book
Aeq = y';
beq = 0;

% See 2.16c
lb = zeros(size(x));
ub = inf * ones(size(x));

a = quadprog(H, p, A, b, Aeq, beq, lb, ub)

w = get_weight(a, y, x)
b = get_bias(a, y, x, w)


range = -3:1;

figure(1);
hold on;
grid on;
plot(range, (range .* w) + b);
plot(x, y, 'o');
ax = gca;
ax.XAxisLocation = 'origin';
ax.YAxisLocation = 'origin';
hold off;

% 2.16a
function H = get_hessian_matrix(x, y)
    H = (y*y').*(x*x');
    if cond(H) == inf
        c = 0.00000001;
        H = H + (c*eye(size(H,1)));
    end
end

% 2.17a
function w = get_weight(a, y, x)
    w = 0;
    for i = 1:size(a,1)
        get = (a(i) * y(i) * x(i));
        w = w + get;
    end
end

% 2.17b
function b = get_bias(a, y, x, w)
    b = 0;
    count = 0;
    for i = 1:size(x,1)
        if a(i) > 0.000001 % only handle support vectors
            b = b + (1/y(i))-(x(i)'*w);
            count = count + 1;
        end
    end
    
    b = b / count;
end