%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%THE CODE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
This folder contains the MATLAB code for training a classification neural network with
backpropagation (BP). There are 3 different modes of training namely:
--- Training without noise
--- Training with NEM noise
--- Training with Blind noise

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FOLDERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

The NBP folder contains 5 folders. The folders are:   
1). data -- This folder contains the dataset for training and testing. We are using the
            MNIST dataset for this task. The MNIST dataset is a large dataset of handwritten digits.
            You can find more details about the MNIST dataset on https://en.wikipedia.org/wiki/MNIST_database.
    
    Action--  DON'T CAHNGE ANYTHING

2). init_params -- This folder contains my best training weights for a neural network 3 hidden layers 
                   and 40 neurons each.
    
    Action--  DON'T CAHNGE ANYTHING

3). lightspeed -- This folder contains helper function for training the network.

    Action -- DON'T CHANGE ANYTHING IN THIS FOLDER.

4). logdir -- The logdir holds subfolders storing the log files of BP training.
              The subfolders are:
              *** blind_noise -- Holds the log files for BP training with blind noise.
              *** nem_noise --   Holds the log files for BP training with NEM noise
              *** no_noise --    Holds the log files for BP training with no noise.

              The log files are .txt files a log file contains these 5 columns:
              Epoch	   Training CE	 Testing CE		Training Acc	Test Acc
        
             where CE stands for mean cross entropy between the predicted and true class label
             posterior probability mass functions, and "Acc" denotes the classification accuracy.

    Action -- You can view your training log in the subfolders after training.

5). Number_2 -- This folder contains the file for the 3 BP training modes.
                You will need the stored weights in init_param for Number2.
    
    Action -- For Blind_noise training, you can only change some parameters in config_blind_noise.m  
              You can change the paramters under "ADJUSTABLE PARAMETERS" section in this file. 

              For NEM_noise training, you can only change some parameters in config_nem_noise.m
              You can change the paramters under "ADJUSTABLE PARAMETERS" section in the file. 

              For No_noise training, you can only change some parameters in config_no_noise.m 
              You can change the paramters under "ADJUSTABLE PARAMETERS" section in the file.

6). Number_3_and_4 -- This folder contains the file for the 3 BP training modes.
                       You will need to specify the size of hidden layers and number of neurons.
                       For example [20,40,50,60] means 20, 30, 40, and 50 neurons at the first, 
                       second, third, and fourth hidden layers respectively.

    Action --  For Blind_noise training, you can only change some parameters in config_blind_noise.m  
               Change the paramters under "ADJUSTABLE PARAMETERS" section in this file. 
              
               For NEM_noise training, you can only change some parameters in config_nem_noise.m
               Change the paramters under "ADJUSTABLE PARAMETERS" section in the file. 

               For No_noise training, you can only change some parameters in config_no_noise.m 
               Change the paramters under "ADJUSTABLE PARAMETERS" section in the file.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NUMBER 2%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% HOW TO RUN BP TRAINING WITH NO NOISE MODE %%%%%%%%%%%%%%%%%%%%%%%%%
1). Open the NBP folder.
2). Open Number2 subfolder.
3). Open config_no_noise.m and change the adjustable parameters. (Optional)
4). Open mnistclassify_no_noise.m and click Run. 
5). Check the logfile in directory logdir/no_noise.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% HOW TO RUN BP TRAINING WITH NEM NOISE MODE %%%%%%%%%%%%%%%%%%%%%%
1). Open the NBP folder.
2). Open Number2 subfolder.
3). Open config_nem_noise.m and change the adjustable parameters. (Optional)
4). Open mnistclassify_nem_noise.m and click Run. 
5). Check the logfile in directory logdir/nem_noise. 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% HOW TO RUN BP TRAINING WITH BLIND NOISE MODE %%%%%%%%%%%%%%%%%%%%%%%
1). Open the NBP folder.
2). Open Number2 subfolder.
3). Open config_blind_noise.m and change the adjustable parameters. (Optional)
4). Open mnistclassify_blind_noise.m and click Run.
5). Check the logfile in directory logdir/blind_noise. 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NUMBER_3_and _4 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

You can also try different initialization techniques.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% HOW TO RUN BP TRAINING WITH NO NOISE MODE %%%%%%%%%%%%%%%%%%%%%%%%%
1). Open the NBP folder.
2). Open Number3_and_4 subfolder.
3). Open config_no_noise.m and change the adjustable parameters. (Optional)
4). Open mnistclassify_no_noise.m and click Run. 
5). Check the logfile in directory logdir/no_noise.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% HOW TO RUN BP TRAINING WITH NEM NOISE MODE %%%%%%%%%%%%%%%%%%%%%%
1). Open the NBP folder.
2). Open Number3_and_4 subfolder.
3). Open config_nem_noise.m and change the adjustable parameters. (Optional)
4). Open mnistclassify_nem_noise.m and click Run. 
5). Check the logfile in directory logdir/nem_noise. 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% HOW TO RUN BP TRAINING WITH BLIND NOISE MODE %%%%%%%%%%%%%%%%%%%%%%%
1). Open the NBP folder.
2). Open Number3_and_4 subfolder.
3). Open config_blind_noise.m and change the adjustable parameters. (Optional)
4). Open mnistclassify_blind_noise.m and click Run.
5). Check the logfile in directory logdir/blind_noise. 
