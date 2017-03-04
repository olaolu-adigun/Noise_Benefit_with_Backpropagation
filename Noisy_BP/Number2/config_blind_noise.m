% Add lightspeed toolbox to path
addpath('../lightspeed/');

% Path of the log directory
logdir = '../logdir/blind_noise/';

% Path of the directory containing the initial neural network parameters
init_param_dir = '../init_params/';

% Prefix of all log file names
logfile_prefix = 'mnist_backprop';

% Max number of Monte-Carlo samples for EM-BP
maxMC = 15;

% Number of neurons in each of the 3 hidden layers
numhid = 40;
numpen = numhid;
numpen2 = numhid;
size_hidden = [40,40,40];
%% Adjustable Parameters
% Maximum number of training epochs
maxepoch = 50;

% Noise variance
noisevar = 0.48;  %[0.4, 0.44, 0.48, 1]

% Noise annealing factor
annfactor = 1;  % [0.5, 1.0, 1.5, 2.0];
%%%%%%%%%%%%%%%%%%%%%END OF ADJUSTABLE PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%

%% Load initial parameters for neural network
load(strcat(init_param_dir,'mnistvhclassify_nhid',num2str(numhid),'.mat'));
load(strcat(init_param_dir,'mnisthpclassify_nhid',num2str(numhid),'.mat'));
load(strcat(init_param_dir,'mnisthp2classify_nhid',num2str(numhid),'.mat'));
load(strcat(init_param_dir,'w_class_init_nhid',num2str(numhid),'.mat'));

% Load the data
load('../data/mnist_rand_batches.mat');
