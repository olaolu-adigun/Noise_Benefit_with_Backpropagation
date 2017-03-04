% Main code for running the backpropagation training

clear all; 
close all; 
config_no_noise;

% Run the training

logfile = [logdir logfile_prefix '_no_Noise_numhid_'  '_40_40_40_Number2' '.txt'];

fid = fopen(logfile,'w');
backpropclassify_no_noise;
fclose(fid);
test_acc = 1 - test_err;
train_acc = 1 - train_err;
    
figure;
plot(train_acc, 'r-','LineWidth',2); hold on;
plot(test_acc, 'b-','LineWidth',2);
  
title('CLASSIFICATION ACCURACY FOR BACKPROPAGATION NO NOISE');
xlabel('Iteration'); % x-axis label
ylabel('Classification Accuracy'); % y-axis label
legend('Training','Testing','Location','southwest');
   
figure;
plot(train_crerr, 'r-','LineWidth',2); hold on;
plot(test_crerr, 'b-','LineWidth',2);
title('CROSS-ENTROPY FOR BACKPROPAGATION NO NOISE');
xlabel('Iteration'); % x-axis label
ylabel('Cross-Entropy'); % y-axis label
legend('Training','Testing','Location','southwest');
   