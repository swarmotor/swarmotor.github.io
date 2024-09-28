mu1 = mean(data1);
sigma1 = std(data1);
mu2 = mean(data2);
sigma2 = std(data2);


x = linspace(min([data1, data2]), max([data1, data2]), 1000); 


pdf1 = normpdf(x, mu1, sigma1);
pdf2 = normpdf(x, mu2, sigma2);


figure;


hold on; 


alpha = 0.05;
z = norminv(1 - alpha/2);
upper1 = mu1 + z * sigma1;
upper2 = mu2 + z * sigma2;


plot(x, pdf1, 'k-', 'LineWidth', 2); 
plot(x, pdf2, 'b-', 'LineWidth', 2); 

plot([upper2, upper2], [0, max(pdf2) * 1.1], 'r--', 'LineWidth', 1);
plot([upper1, upper1], [0, max(pdf1) * 1.1], 'k--', 'LineWidth', 1);
idx1 = x >=0 & x <= upper1;
area(x(idx1), pdf1(idx1), 'FaceColor', [0.3, 0.5, 0.2], 'FaceAlpha', 0.3);
idx2 = x >= 0 & x <= upper2;
area(x(idx2), pdf2(idx2), 'FaceColor', [0, 0, 1], 'FaceAlpha', 0.3);

legend('CLASSICAL VISIBILITY MODEL', 'ATTENTION SELECTED MODEL');


xlabel('TOTAL NEIGHBORS EACH TURN', 'FontSize', 14);
ylabel('Probability Density', 'FontSize', 14);
title('Gaussian Fit Distribution with 95% Confidence Interval', 'FontSize', 16);


ylim([0, max([pdf1, pdf2]) * 1.1]);


set(gca, 'FontSize', 12);


hold off;