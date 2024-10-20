%% Q02
clc; clear;

% Given data points
hit = 0.75;
false_alarm = 0.11;

% Plot the ROC curve
figure;
plot([0, 1], [0, 1], '--');  % Diagonal line (random classifier)
hold on;
plot(false_alarm, hit, 'go', 'MarkerSize', 4);  % Data point
title('ROC Curve', 'Interpreter', 'latex');
xlabel('False alarm', 'Interpreter', 'latex');
ylabel('Hit', 'Interpreter', 'latex');
grid on;
legend('Our Trial', 'Data Point');
hold off;

%% Q05

clc; clear;

% Specify the path to your CSV file
csvFilePath = 'study.csv';

% Read the CSV file into a table
table = readtable(csvFilePath);

% Extract relevant columns
stim_value = table2array(table(:,9));
correct = table2array(table(:,7));
reaction_time = table2array(table(:,6));

% Create logical indices for reward and loss-associated stimuli
reward_indices = (((stim_value == 1) | (stim_value == 2)) & (~isnan(correct)) & (~isnan(reaction_time)));
loss_indices = ((stim_value == 0) & (~isnan(correct)) & (~isnan(reaction_time)));

reward_accuracy = table{reward_indices, 'correct'};
loss_accuracy = table{loss_indices, 'correct'};

% Calculate mean reaction times for reward and loss-associated stimuli
reward_mean_reaction_time = mean(table{reward_indices, 'reactiontime'});
loss_mean_reaction_time = mean(table{loss_indices, 'reactiontime'});

% Calculate mean accuracy for reward and loss-associated stimuli
reward_mean_accuracy = mean(reward_accuracy);
loss_mean_accuracy = mean(loss_accuracy);

% Display the results
disp(['Reward Mean Reaction Time: ', num2str(reward_mean_reaction_time)]);
disp(['Loss Mean Reaction Time: ', num2str(loss_mean_reaction_time)]);
disp(['Reward Mean Accuracy: ', num2str(reward_mean_accuracy)]);
disp(['Loss Mean Accuracy: ', num2str(loss_mean_accuracy)]);

% Data
categories = {'Reward', 'Loss'};
mean_reaction_times = [reward_mean_reaction_time, loss_mean_reaction_time];
mean_accuracy = [reward_mean_accuracy, loss_mean_accuracy];

% Create a bar plot
subplot(1,2,1)
bar(mean_reaction_times, 'FaceColor', [0.2 0.2 0.5]);
xticks(1:2);
xticklabels(categories);
xlabel('Condition', 'Interpreter','latex');
ylabel('Mean Reaction Time', 'Interpreter','latex');
title('Comparison of Mean Reaction Time', 'Interpreter','latex');
ylim([0.6, 1.0]);

% Create a bar plot
subplot(1,2,2)
bar(mean_accuracy, 'FaceColor', [0.2 0.2 0.5]);
xticks(1:2);
xticklabels(categories);
xlabel('Condition', 'Interpreter','latex');
ylabel('Mean Reaction Time', 'Interpreter','latex');
title('Comparison of Mean Accuracy', 'Interpreter','latex');
ylim([0.6, 1.0]);

% Define your training set sizes
training_reward = cumsum(reward_accuracy); 
training_loss = cumsum(loss_accuracy);

training_reward_mean = cumulative_mean(reward_accuracy); 
training_loss_mean = cumulative_mean(loss_accuracy);

% Create a figure
figure;

% Plot reward model accuracy
plot([1:length(reward_accuracy)], training_reward, 'LineWidth', 2);
hold on;

% Plot loss-associated model accuracy
plot([1:length(loss_accuracy)], training_loss, 'LineWidth', 2);

% Add labels and title
xlabel('Training Set Size', 'Interpreter','latex');
ylabel('Cumulative', 'Interpreter','latex');
title('Learning Curves: Reward vs. Loss-Associated', 'Interpreter','latex');
legend('Reward Model', 'Loss-Associated Model', 'Interpreter','latex');


% Create a figure
figure;

% Plot reward model accuracy
plot([1:length(reward_accuracy)], training_reward_mean, 'LineWidth', 2);
hold on;

% Plot loss-associated model accuracy
plot([1:length(loss_accuracy)], training_loss_mean, 'LineWidth', 2);

% Add labels and title
xlabel('Training Set Size', 'Interpreter','latex');
ylabel('Accuracy', 'Interpreter','latex');
title('Learning Curves: Reward vs. Loss-Associated', 'Interpreter','latex');
legend('Reward Model', 'Loss-Associated Model', 'Interpreter','latex');



%% Functions

function cum_mean = cumulative_mean(data)
    mean = 0;
    for i=1:length(data)
        sum = mean*(i-1) + data(i);
        mean =  sum/i;
        cum_mean(i) = mean;
    end
    cum_mean = cum_mean';
end


