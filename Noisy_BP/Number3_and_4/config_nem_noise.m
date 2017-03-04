
% Add lightspeed toolbox to path
addpath('../lightspeed/');

% Path of the log directory
logdir = '../logdir/nem_noise/';

% Path of the directory containing the initial neural network parameters
init_param_dir = '../init_params/';

% Prefix of all log file names
logfile_prefix = 'mnist_backprop';

% Max number of Monte-Carlo samples for EM-BP
maxMC = 10;

%% Adjustable Parameters

%%%%%%%%%%%%%%%%%%%%%%%% ADJUSTABLE PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%
% Maximum number of training epochs
maxepoch = 50;

% Number of neurons in each of the 3 hidden layers
size_hidden = [40, 40];
wgt_bound = 1.0;

% Noise variance
noisevar = 0.48;  %[0.4, 0.44, 0.48, 1]

% Noise annealing factor
annfactor = 1.0;  % [0.8, 1.5, 2.0, 3.0];
%%%%%%%%%%%%%%%%%%%% END ADJUSTABLE PARAMETERS %%%%%%%%%%%%%%%%%%%%%%

%% Define the Network

nn = {};
nn.e = wgt_bound;
nn.W = {};
nn.Wprobs = {};
nn.hsamp = {};
nn.Wprobs_samp = {};
nn.Wprobs_test = {};

%% Load the data
load('../data/mnist_rand_batches.mat');

[numcases numdims numbatches]= size(batchdata);
[numcases1 numdims1, numbatchtes1] = size(batchtargets);

L = [numdims, size_hidden, numdims1];
num1 = numel(L);

%% Initialize weights

for idx = 1:1:num1 -1
    nn.W{idx} = nn.e * randn(L(idx) + 1, L(idx+1));
end;

numhid = '';
for k = 1:1:numel(size_hidden)
    numhid = strcat(numhid,'_',num2str(size_hidden(k)));
end;
