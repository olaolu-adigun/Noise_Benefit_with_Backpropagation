% Main code for running the backpropagation training

clear all; 
close all; 
clc; 
config_blind_noise;

%% Blind Noise: Output Only
logfile = [logdir 'noisevar_' num2str(noisevar) '_annfact_' num2str(annfactor)...
          '_numhid_' '_40_40_40_Number2' '.txt'];
fid = fopen(logfile,'w');


backpropclassify_blind_noise;
fclose(fid);
%test_acc = 1 - test_err;
%train_acc = 1 - train_err;
    
figure;
plot(train_acc, 'r-','LineWidth',2); hold on;
plot(test_acc, 'b-','LineWidth',2);
  
title('CLASSIFICATION ACCURACY FOR BACKPROPAGATION WITH BLIND NOISE');
xlabel('Iteration'); % x-axis label
ylabel('Classification Accuracy'); % y-axis label
legend('Training','Testing','Location','southwest');
   
figure;
plot(train_crerr, 'r-','LineWidth',2); hold on;
plot(test_crerr, 'b-','LineWidth',2);
title('CROSS-ENTROPY FOR BACKPROPAGATION WITH BLIND NOISE');
xlabel('Iteration'); % x-axis label
ylabel('Cross-Entropy'); % y-axis label
legend('Training','Testing','Location','southwest');           
           