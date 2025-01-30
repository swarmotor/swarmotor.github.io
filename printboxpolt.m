rng('default');
data = {randn(100,1)*0.5+1,  randn(150,1)*0.8+2, randn(200,1)*1.2+3};
groupLabels = {'Group A', 'Group B', 'Group C'};

groups = arrayfun(@(k) ones(length(data{k}),1)*k, 1:numel(data), 'un',0);
groups = cat(1, groups{:});

allData = cat(1, data{:});

figure('Color', [1 1 1]);
boxplot(allData, groups, 'Labels', groupLabels, 'Whisker', 1.5);
hold on;

% ===== 样式调整 =====
boxColors = [
    0.2 0.6 0.8   % 蓝色
    0.8 0.4 0.2   % 橙色
    0.4 0.8 0.4   % 绿色
];

h = findobj(gca,'Tag','Box');
for k=1:length(h)
    patch(get(h(k),'XData'), get(h(k),'YData'), boxColors(k,:),...
        'FaceAlpha',0.5, 'EdgeColor', boxColors(k,:)*0.7, 'LineWidth',2);
end

% 中位数线
medLines = findobj(gca, 'Tag', 'Median');
set(medLines, 'Color', [0.9 0.2 0.2], 'LineWidth', 2);

% 触须样式
whiskers = findobj(gca, 'Tag', 'Whisker');
set(whiskers, 'LineStyle', '-', 'Color', [0.4 0.4 0.4], 'LineWidth', 1.2);

% 异常值样式
outliers = findobj(gca, 'Tag', 'Outliers');
set(outliers, 'Marker', 'o', 'MarkerEdgeColor', [0.6 0 0],...
    'MarkerFaceColor', [1 0.8 0.8], 'MarkerSize', 5, 'LineWidth', 0.8);

set(gca, 'FontName', 'Arial', 'FontSize', 12,...
    'LineWidth', 1.2, 'XColor', [0.3 0.3 0.3], 'YColor', [0.3 0.3 0.3]);
grid on;
set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.3);
xlabel('Experimental Groups', 'FontSize', 14, 'Color', [0.2 0.2 0.2]);
ylabel('Measurement Values', 'FontSize', 14, 'Color', [0.2 0.2 0.2]);
title('Customized Boxplot with Style Enhancements',...
    'FontSize', 16, 'FontWeight', 'bold', 'Color', [0.2 0.2 0.2]);
set(gcf, 'Position', [100 100 800 600]); % 设置图形大小
hold off;
