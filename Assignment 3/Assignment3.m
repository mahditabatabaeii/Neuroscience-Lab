% Mahdi Tabatabaei
clc; clear;

%% Q2 - a

% Load data from .mat file
Data = load('Data_Search_Time.mat');  % Assuming your .mat file contains variables DS, TD, and ST

DS = Data.Data.DS;
TD = Data.Data.TD;
ST = Data.Data.SearchTime;

% Define independent variables (predictors) and dependent variable (response)
X = [DS, TD];  % DS and TD are assumed to be column vectors
Y = ST;        % ST is assumed to be a column vector

% Fit multiple linear regression model
mdl = fitlm(X, Y, 'linear');

% Display regression results
disp(mdl);

%% Q2 - b


% Fit linear regression model for search time vs. display size / training duration
mdl_ds = fitlm(DS, ST, 'linear');
mdl_td = fitlm(TD, ST, 'linear');

% Plot 2D scatter plot for search time vs. display size and the fitted line
figure;
scatter(DS, ST);
xlabel('Display Size (DS)');
ylabel('Search Time (ST)');
title('Search Time vs. Display Size');
hold on;
plot(DS, predict(mdl_ds),'r', 'LineWidth', 2);
legend('Data','Fitted Line');
hold off;

% Plot 2D scatter plot for search time vs. training duration and the fitted line
figure;
scatter(TD, ST);
xlabel('Training Duration (TD)');
ylabel('Search Time (ST)');
title('Search Time vs. Training Duration');
hold on;
plot(TD, predict(mdl_td),'r', 'LineWidth', 2);
legend('Data','Fitted Line');
hold off;

% Plot 2D scatter plot for whole model
figure;
plot(mdl, 'LineWidth', 2);
xlabel('Whole model');
ylabel('Search Time (ST)');
title('Search Time vs. Other Variables');
hold off;

% Plot 3D scatter plot for search time vs. display size and training duration
figure;
scatter3(DS, TD, ST);
xlabel('Display Size (DS)');
ylabel('Training Duration (TD)');
zlabel('Search Time (ST)');
title('Search Time vs. Display Size and Training Duration');

% Create a grid for the surface plot
[DS_grid, TD_grid] = meshgrid(min(DS):0.1:max(DS), min(TD):0.1:max(TD));
ST_grid = mdl.Coefficients.Estimate(1) + mdl.Coefficients.Estimate(2) * DS_grid + mdl.Coefficients.Estimate(3) * TD_grid;

% Plot the fitted plane
hold on;
surf(DS_grid, TD_grid, ST_grid, 'FaceAlpha', 0.5);
legend('Data','Fitted Plane');
hold off;

%% Q3 - a

% Obtain residuals
residuals = mdl.Residuals.Raw;

% Create Q-Q plot of residuals
figure;
qqplot(residuals);
xlabel('Theoretical Quantiles');
ylabel('Sample Quantiles');
title('Q-Q Plot of Residuals');

% Create histogram of residuals
figure;
histogram(residuals);
xlabel('Bins');
ylabel('Residuals');
title('Histogram of Residuals');

%% Q3 - b

% Obtain predicted values and residuals
predicted = mdl.Fitted;
residuals = mdl.Residuals.Raw;

% Create plot of residuals vs predicted values
figure;
scatter(predicted, residuals);
hold on;  % Add this line to hold the plot
plot([min(predicted), max(predicted)], [0, 0], 'r--');  % Plot line y = 0
hold off;  % Release the hold
xlabel('Predicted Values');
ylabel('Residuals');
title('Residuals vs Predicted Values');

%% Q3 - c

% Obtain residuals
residuals = mdl.Residuals.Raw;

% Plot residuals over observation index
figure;
plot(1:length(residuals), residuals);
xlabel('Observation Index');
ylabel('Residuals');
title('Residuals Over Observation Index');

% Scatter plot of residuals against observation index
figure;
scatter(1:length(residuals), residuals);
xlabel('Observation Index');
ylabel('Residuals');
title('Scatter Plot of Residuals Against Observation Index');

% Durbin-Watson test
durbinWatsonStat = sum(diff(residuals).^2) / sum(residuals.^2);

% Ljung-Box test
lags = 10; % Number of lags for Ljung-Box test
[h, pValue, QStat, CriticalValue] = lbqtest(residuals, 'Lags', lags);

% Display results of statistical tests
fprintf('Durbin-Watson test statistic: %.4f\n', durbinWatsonStat);
fprintf('Ljung-Box test statistic: %.4f\n', QStat);
fprintf('p-value: %.4f\n', pValue);

%% Q4
clc;

% Step-wise regression: Fit ST with DS first, then use residuals to fit TD
mdl_DS_first = fitlm(DS, ST, 'linear'); 
residuals_DS_first = mdl_DS_first.Residuals.Raw;  
mdl_TD_after_DS = fitlm(residuals_DS_first, TD, 'linear'); 

disp(mdl_DS_first);
disp(mdl_TD_after_DS);

% Step-wise regression: Fit ST with TD first, then use residuals to fit DS
mdl_TD_first = fitlm(TD, ST, 'linear');  
residuals_TD_first = mdl_TD_first.Residuals.Raw;  
mdl_DS_after_TD = fitlm(DS, residuals_TD_first, 'linear');  

disp(mdl_TD_first)
disp(mdl_DS_after_TD);

%% Q5 - a

% Create Q-Q plot of ST
figure;
qqplot(ST);
xlabel('Theoretical Quantiles');
ylabel('Sample Quantiles');
title('Q-Q Plot of ST');

% Create histogram of ST
figure;
histogram(ST);
xlabel('Bins');
ylabel('ST');
title('Histogram of Residuals');

%% Q5 - b

% Apply logarithmic transformation
ST_log = log(ST);

% Apply square root transformation
ST_sqrt = sqrt(ST);

% Apply inverse transformation
ST_inv = 1 ./ ST;

% Plot Q-Q plots and histograms of transformed variables
figure;

subplot(3, 2, 1);
qqplot(ST_log);
title('Q-Q Plot of Log Transformed ST');
subplot(3, 2, 3);
qqplot(ST_sqrt);
title('Q-Q Plot of Square Root Transformed ST');
subplot(3, 2, 5);
qqplot(ST_inv);
title('Q-Q Plot of Inverse Transformed ST');

subplot(3, 2, 2);
histogram(ST_log);
title('Histogram of Log Transformed ST');
subplot(3, 2, 4);
histogram(ST_sqrt);
title('Histogram of Square Root Transformed ST');
subplot(3, 2, 6);
histogram(ST_inv);
title('Histogram of Inverse Transformed ST');

mld_transformed = fitlm(X, ST_inv);
inv_resuals = mld_transformed.Residuals.Raw;
disp(mld_transformed);

% Create Q-Q plot of ST
figure;
qqplot(inv_resuals);
xlabel('Theoretical Quantiles');
ylabel('Sample Quantiles');
title('Q-Q Plot of ST');

% Create histogram of ST
figure;
histogram(inv_resuals);
xlabel('Bins');
ylabel('ST');
title('Histogram of Residuals');

%% Q6 - b

% Perform one-way ANOVA
[p, tbl, stats, terms] = anovan(ST, {DS, TD});

% Display ANOVA table
disp('One-Way ANOVA Results:');
disp(tbl);

% Report significance of factors and their interaction
if p < 0.05
    disp('At least one factor or interaction is significant (p < 0.05)');
else
    disp('No significant factors or interactions (p >= 0.05)');
end

%% Q6 - c
clc;

% Perform post-hoc comparisons using Tukey's method
figure(1)
comparison_tukey = multcompare(stats, 'CType', 'tukey-kramer');

% Perform post-hoc comparisons using Scheffé's method
figure(2)
comparison_scheffe = multcompare(stats, 'CType', 'scheffe');

% Perform post-hoc comparisons using Bonferroni's method
figure(3)
comparison_bonferroni = multcompare(stats, 'CType', 'bonferroni');

% Display post-hoc comparison results
disp('Tukey Post-Hoc Comparison:');
disp(comparison_tukey);
disp('Scheffé Post-Hoc Comparison:');
disp(comparison_scheffe);
disp('Bonferroni Post-Hoc Comparison:');
disp(comparison_bonferroni);

%% Q7 - b
clc;

subject = Data.Data.Subject;

% Perform ANOVA including subject as a factor
[p, tbl, stats] = anovan(ST, {subject, DS, TD}, 'model', 'interaction', 'varnames', {'DS', 'TD', 'subject'});

% Display ANOVA table
disp('ANOVA Results:');
disp(tbl);

% Report significance of factors and interactions
if any(p < 0.05)
    disp('At least one factor or interaction is significant (p < 0.05)');
else
    disp('No significant factors or interactions (p >= 0.05)');
end
