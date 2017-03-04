[numcases numdims numbatches]= size(batchdata);
[numcases1 numdims1, numbatchtes1] = size(batchtargets);

N=numcases;

% Load the weights
w1=[vishid; hidrecbiases];
w2=[hidpen; penrecbiases];
w3=[hidpen2; penrecbiases2];

test_err=[];
train_err=[];
test_acc =[];
train_acc=[];

%% Define the network 
nn = {};
nn.W = {};
nn.Wprobs = {};
nn.hsamp = {};
nn.Wprobs_samp = {};
nn.Wprobs_test = {};

L = [numdims, size_hidden, numdims1];
num1 = numel(L);

% Initialize weights
nn.W{1} = w1;
nn.W{2} = w2;
nn.W{3} = w3;
nn.W{4} = w_class;

for epoch = 1:maxepoch
    %%%%%%%%%%%%%%%%%%%% COMPUTE TRAINING MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    err=0;
    err_cr=0;
    counter=0;
    [numcases numdims numbatches]=size(batchdata);
    
    N=numcases;
    for batch = 1:10
        data = [batchdata(:,:,batch)];
        target = [batchtargets(:,:,batch)];
        data = [data ones(N,1)];
        val = 1./(1 + exp(-data * nn.W{1}));
        nn.Wprobs{1} = [val ones(N,1)];
        
        for idx = 2:1:(num1 - 2)
            val =    1./(1 + exp(-nn.Wprobs{idx-1}*nn.W{idx}));
            nn.Wprobs{idx} = [val  ones(N,1)];
        end;
        
        Targetout = exp(nn.Wprobs{num1-2}*nn.W{num1-1});
        Targetout = Targetout./repmat(sum(Targetout,2),1,10); 
        
        [I J] = max(Targetout,[],2);
        [I1 J1]= max(target,[],2);
        counter = counter+length(find(J == J1));
        err_cr = err_cr- sum(sum( target(:,1:end).*log(Targetout)));

    end
    train_err(epoch)=(numcases*10-counter)/(numcases*10);
    train_acc(epoch) = 1 - train_err(epoch);
    train_crerr(epoch)= err_cr/(numcases*10);
    
    %%%%%%%%%%%%%% END OF COMPUTING TRAINING MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%% COMPUTE TEST MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    err=0;
    err_cr=0;
    counter=0;
    [testnumcases testnumdims testnumbatches]=size(testbatchdata);
    N=testnumcases;
    for batch = 1:testnumbatches
        data = [testbatchdata(:,:,batch)];
        target = [testbatchtargets(:,:,batch)];
        data = [data ones(N,1)];
        val = 1./(1 + exp(-data * nn.W{1}));
        nn.Wprobs_test{1} = [val ones(N,1)];
        
        for idx = 2:1:(num1 - 2)
            val =    1./(1 + exp(-nn.Wprobs_test{idx-1}*nn.W{idx}));
            nn.Wprobs_test{idx} = [val  ones(N,1)];
        end;
        
        Targetout = exp(nn.Wprobs_test{num1-2}*nn.W{num1-1});
        Targetout = Targetout./repmat(sum(Targetout,2),1,10); 
        
        [I J]= max(Targetout,[],2);
        [I1 J1]= max(target,[],2);
        counter = counter + length(find(J == J1));
        err_cr = err_cr- sum(sum( target(:,1:end).*log(Targetout)));
        
    end
    test_err(epoch)=(testnumcases*testnumbatches-counter)/(testnumcases*testnumbatches);
    test_acc(epoch) = 1 - test_err(epoch);
    test_crerr(epoch)=err_cr/(testnumcases*testnumbatches);
    
	if epoch > 1
	    fprintf(fid,'%d\t%f\t%f\t%f\t%f\n',epoch-1,train_crerr(epoch),test_crerr(epoch),train_acc(epoch),test_acc(epoch));
	end
    
    if epoch == maxepoch
        Dat = [];
        Targ = [];
        for id = 1:1:100
            Dat = [Dat; testbatchdata(:,:,id)];
            Targ = [Targ; testbatchtargets(:,:,id)];
        end;
        N1 = size(Dat,1);
        Dat = [Dat , ones(N1,1)];
        val = 1./(1 + exp(-Dat * nn.W{1}));
        nn.Wprobs_test{1} = [val ones(N1,1)];
        
        for idx = 2:1:(num1 - 2)
            val =    1./(1 + exp(-nn.Wprobs_test{idx-1}*nn.W{idx}));
            nn.Wprobs_test{idx} = [val  ones(N1,1)];
        end;
        
        Targetout = exp(nn.Wprobs_test{num1-2}*nn.W{num1-1});
        Targetout = Targetout./repmat(sum(Targetout,2),1,10); 
        [I J]= max(Targetout,[],2);
        [I1 J1]= max(Targ,[],2);

        for i = 0:1:9
          tag = find(J == i+1);
          pred = find(J1 == i+1);
          Y1 = numel(pred);
          correct = intersect(tag, pred);
          X1 = numel(correct);
          text = ['The classification accuracy of digit ', num2str(i) , ' is: ', num2str(X1/Y1)];
          disp(text);
        end;
    end;
    
    %%%%%%%%%%%%%% END OF COMPUTING TEST MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    tt=0;
    for batch = 1
        tt=tt+1;
        data=[];
        targets=[];
        for kk=1:10
            data=[data
                batchdata(:,:,(tt-1)*10+kk)];
            targets=[targets
                batchtargets(:,:,(tt-1)*10+kk)];
        end
        
        %%%%%%%%%%%%%%% PERFORM CONJUGATE GRADIENT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        max_iter=1;
        N = size(data,1);

        VV = [];
        for i = 1:1:num1-1
            val = nn.W{i};
            VV = [VV; val(:)];
        end;
        Dim = L;     
        XX = [data ones(N,1)];
        nn.Wprobs{1} = 1./(1 + exp(-XX*nn.W{1}));
        nn.Wprobs{1} = [nn.Wprobs{1} ones(N,1)];
        nn.hsamp{1} = zeros(N,L(2)+1,maxMC);
        for idx = 2:1:num1-2
            nn.Wprobs_samp{idx} = zeros(N,L(idx+1)+1,maxMC);
            nn.hsamp{idx} = zeros(N,L(idx)+1,maxMC);
        end;
        Targetout_samp = zeros(N,L(num1),maxMC);
        Sampimp = zeros(N,maxMC);
        
        %% Get MC samples and their importance
        for iterMC = 1:maxMC
            nn.hsamp{1}(:,:,iterMC) = sparse((rand(N,L(2)+1) >= 1-nn.Wprobs{1}));	% hidden1 MC samples
            
            for idx = 2:1:num1-2
                nn.Wprobs_samp{idx}(:,1:end-1,iterMC) = 1./(1+exp(-nn.hsamp{idx-1}(:,:,iterMC)*nn.W{idx}));	% hidden2 act using hidden1 MC samples
                nn.Wprobs_samp{idx}(:,end,iterMC) = ones(N,1);
                nn.hsamp{idx}(:,:,iterMC) = sparse((rand(N,L(idx+1)+1) >= 1-nn.Wprobs_samp{idx}(:,:,iterMC)));	% hidden2 MC samples      
            end
            
            Targetout_samp(:,:,iterMC) = exp(nn.hsamp{num1-2}(:,:,iterMC)*nn.W{num1-1});
            Targetout_samp(:,:,iterMC) = Targetout_samp(:,:,iterMC)./repmat(row_sum(Targetout_samp(:,:,iterMC)),1,L(5));
              
            Sampimp(:,iterMC) = row_sum(Targetout_samp(:,:,iterMC) .* targets);
              
        end
 
        Sampimp = Sampimp ./ repmat(row_sum(Sampimp),1,maxMC);

   
        %% Optimize input-hidden weights
        [X,fX] = minimize(nn.W{1}(:),'compute_qfun',max_iter,[L(1);L(2)],repmat(data,[1,1,maxMC]),nn.hsamp{1}(:,1:end-1,:),maxMC,Sampimp,0);
        w1 = reshape(X,L(1)+1,L(2));
        
        for idx = 2:1:num1-2
            % Optimize hidden-hidden weights
            [X,fX] = minimize(nn.W{idx}(:),'compute_qfun',max_iter,[L(idx);L(idx+1)],nn.hsamp{idx-1}(:,1:end-1,:),nn.hsamp{idx}(:,1:end-1,:),maxMC,Sampimp,0);
            nn.W{idx} = reshape(X,L(idx)+1,L(idx+1));          
        end;
        
        % Optimize hidden-output weights
        [X,fX] = minimize(nn.W{num1-1}(:),'compute_qfun',max_iter,[L(num1-1);L(num1)],nn.hsamp{num1-2}(:,1:end-1,:),repmat(targets,[1,1,maxMC]),maxMC,Sampimp,1);
        nn.W{num1-1} = reshape(X,L(num1-1)+1,L(num1));
  
        %%%%%%%%%%%%%%% END OF CONJUGATE GRADIENT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    end
end
