% Add lightspeed toolbox to path
addpath('../lightspeed/');

% Path of the log directory
logdir = '../logdir/no_noise/';

% Path of the directory containing the initial neural network parameters
init_param_dir = '../init_params/';

% Prefix of all log file names
logfile_prefix = 'mnist_backprop';

% Max number of Monte-Carlo samples for EM-BP
maxMC = 10;

%% Define the Network
nn = {};
nn.W = {};
nn.Wprobs = {};
nn.hsamp = {};
nn.Wprobs_samp = {};
nn.Wprobs_test = {};
%% Adjustable Parameters

%%%%%%%%%%%%%%%%%%%%%%%% ADJUSTABLE PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%
% Maximum number of training epochs
maxepoch = 50;

% Number of neurons in each of the 3 hidden layers
size_hidden = [24 , 30, 40 , 50];
nn.bound = 1; % vary this

%%%%%%%%%%%%%%%%%%%% END ADJUSTABLE PARAMETERS %%%%%%%%%%%%%%%%%%%%%%
%% Load the data
load('../data/mnist_rand_batches.mat');

[numcases numdims numbatches]= size(batchdata);
[numcases1 numdims1, numbatchtes1] = size(batchtargets);

L = [numdims, size_hidden, numdims1];
num1 = numel(L);

%% Initialize weights for all the layers 
for idx = 1:1:num1 -1
    nn.W{idx} = (nn.bound * randn(L(idx) + 1, L(idx+1))) ;
end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numhid = '';
for k = 1:1:numel(size_hidden)
    numhid = strcat(numhid,'_',num2str(size_hidden(k)));
end;
