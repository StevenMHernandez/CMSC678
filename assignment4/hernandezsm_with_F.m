close all; clear all;

% Quality: [terrible poor ok good excellent]
Q = [0 25 50 75 100];

% Number of Errors: [low few many]
E = [2 5 8];

% Power of Figures: [ugly medium very-fine/VF]
F = [50 75 100];

%  Grades: [F D C B A]
G = [40 55 70 85 100];

% To make it easier to play around with these values in `Y`, 
% Let's defined them as MATLAB variables.
grade_F = G(1);
grade_D = G(2);
grade_C = G(3);
grade_B = G(4);
grade_A = G(5);

% Y = [
%     grade_C % Q:terrible, E: low
%     grade_D % Q:terrible, E: few
%     grade_F % Q:terrible, E: many
%     grade_C % Q:poor, E: low
%     grade_D % Q:poor, E: few
%     grade_F % Q:poor, E: many
%     grade_B % Q:ok, E: low
%     grade_B % Q:ok, E: few
%     grade_C % Q:ok, E: many
%     grade_A % Q:good, E: low
%     grade_B % Q:good, E: few
%     grade_C % Q:good, E: many
%     grade_A % Q:excellent, E: low
%     grade_A % Q:excellent, E: few
%     grade_C % Q:excellent, E: many
% ];

Y = [
    grade_D % Q:terrible, E: low, F: ugly
    grade_C % Q:terrible, E: low, F: medium
    grade_C % Q:terrible, E: low, F: VF
    grade_D % Q:terrible, E: few, F: ugly
    grade_D % Q:terrible, E: few, F: medium
    grade_D % Q:terrible, E: few, F: VF
    grade_F % Q:terrible, E: many, F: ugly
    grade_F % Q:terrible, E: many, F: medium
    grade_F % Q:terrible, E: many, F: VF
    grade_C % Q:poor, E: low, F: ugly
    grade_C % Q:poor, E: low, F: medium
    grade_C % Q:poor, E: low, F: VF
    grade_D % Q:poor, E: few, F: ugly
    grade_D % Q:poor, E: few, F: medium
    grade_D % Q:poor, E: few, F: VF
    grade_F % Q:poor, E: many, F: ugly
    grade_F % Q:poor, E: many, F: medium
    grade_D % Q:poor, E: many, F: VF
    grade_B % Q:ok, E: low, F: ugly
    grade_B % Q:ok, E: low, F: medium
    grade_A % Q:ok, E: low, F: VF
    grade_B % Q:ok, E: few, F: ugly
    grade_B % Q:ok, E: few, F: medium
    grade_A % Q:ok, E: few, F: VF
    grade_C % Q:ok, E: many, F: ugly
    grade_C % Q:ok, E: many, F: medium
    grade_C % Q:ok, E: many, F: VF
    grade_A % Q:good, E: low, F: ugly
    grade_A % Q:good, E: low, F: medium
    grade_A % Q:good, E: low, F: VF
    grade_B % Q:good, E: few, F: ugly
    grade_B % Q:good, E: few, F: medium
    grade_A % Q:good, E: few, F: VF
    grade_D % Q:good, E: many, F: ugly
    grade_C % Q:good, E: many, F: medium
    grade_C % Q:good, E: many, F: VF
    grade_B % Q:excellent, E: low, F: ugly
    grade_A % Q:excellent, E: low, F: medium
    grade_A % Q:excellent, E: low, F: VF
    grade_A % Q:excellent, E: few, F: ugly
    grade_A % Q:excellent, E: few, F: medium
    grade_A % Q:excellent, E: few, F: VF
    grade_D % Q:excellent, E: many, F: ugly
    grade_D % Q:excellent, E: many, F: medium
    grade_C % Q:excellent, E: many, F: VF
];

% Distance from center of triangles to either left or right

% % BROAD MFs
% Q_triangle_width = 50;
% E_triangle_width = 10;
% F_triangle_width = 40;

% MEDIUM MFs
Q_triangle_width = 25;
E_triangle_width = 3;
F_triangle_width = 20;

% % NARROW MFs
% Q_triangle_width = 15;
% E_triangle_width = 2;
% F_triangle_width = 10;

%%%%%%%%%%%%%%
% 
% PLOT FIGURES
% 
%%%%%%%%%%%%%%

figure(1)

% Plot Q
subplot(141)
hold on
for i = 1:length(Q)
    plot([Q(i)-Q_triangle_width Q(i) Q(i)+Q_triangle_width],[0 1 0])
end
xlim([-10 110])
ylim([0 1.1])
title("Quality Membership Functions")


% Plot E
subplot(142)
hold on
for i = 1:length(E)
    plot([E(i)-E_triangle_width E(i) E(i)+E_triangle_width],[0 1 0])
end
xlim([-1 11])
ylim([0 1.1])
title("Errors Membership Functions")


% Plot F
subplot(143)
hold on
for i = 1:length(F)
    plot([F(i)-F_triangle_width F(i) F(i)+F_triangle_width],[0 1 0])
end
xlim([45 105])
ylim([0 1.1])
title("Power of Figures Membership Functions")


% Plot G
subplot(144)
hold on
for i = 1:length(G)
    plot([G(i) G(i)],[0 1], 'b')
    plot([G(i)],[1], 'bo')
end
xlim([30 110])
ylim([0 1.1])
title("Grade Membership Functions")


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Calculate Surface of Knowledge
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

knowledge_surface = zeros(101, 11, 51);
for x = 1:101
    for y = 1:11
        for z = 50:101
            mu_1 = zeros(1,length(Q));
            mu_2 = zeros(1,length(E));
            mu_3 = zeros(1,length(F));

            for i = 1:length(Q)
                d = abs(Q(i) - x); % distance from triangle's center
                if d > Q_triangle_width
                    mu_1(i) = 0;
                else
                    slope = 1 / Q_triangle_width;
                    mu_1(i) = slope * (Q_triangle_width-d);
                end
            end

            for i = 1:length(E)
                d = abs(E(i) - y); % distance from triangle's center
                if d > E_triangle_width
                    mu_2(i) = 0;
                else
                    slope = 1 / E_triangle_width;
                    mu_2(i) = slope * (E_triangle_width-d);
                end
            end

            for i = 1:length(F)
                d = abs(F(i) - z); % distance from triangle's center
                if d > F_triangle_width
                    mu_3(i) = 0;
                else
                    slope = 1 / F_triangle_width;
                    mu_3(i) = slope * (F_triangle_width-d);
                end
            end

            H = kron(kron(mu_1, mu_2),mu_3)';
            knowledge_surface(x,y,z) = (Y' * H) / sum(H);
        end
    end
end


figure(2)
subplot(221)
ks = permute(knowledge_surface, [3 2 1]);
surf(0:10, 50:100, ks(50:100,:,1))
title("Surface of Knowledge when Quality Q=0")
xlabel("Number of Errors")
ylabel("Power of Figures")
ylim([50,100])
zlabel("Grade")

subplot(222)
ks = permute(knowledge_surface, [3 1 2]);
surf(0:100, 50:100, ks(50:100,:,1))
title("Surface of Knowledge when Number of Errors = 0")
xlabel("Quality")
ylabel("Power of Figures")
ylim([50,100])
zlabel("Grade")

subplot(223)
ks = permute(knowledge_surface, [3 2 1]);
surf(0:10, 50:100, ks(50:100,:,100))
title("Surface of Knowledge when Quality Q=100")
xlabel("Number of Errors")
ylabel("Power of Figures")
ylim([50,100])
zlabel("Grade")

subplot(224)
ks = permute(knowledge_surface, [3 1 2]);
surf(0:100, 50:100, ks(50:100,:,10))
title("Surface of Knowledge when Number of Errors = 10")
xlabel("Quality")
ylabel("Power of Figures")
ylim([50,100])
zlabel("Grade")


% % surf(0:10, 0:100, knowledge_surface)
% % surf(0:10, 0:100, knowledge_surface)
% surf(knowledge_surface)
% title("Surface of Knowledge")
% xlabel("Number of Errors")
% ylabel("Quality")
% zlabel("Grade")
