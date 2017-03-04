% Returns MC-EM Q-function and its gradient

function [f, df] = compute_qfun(VV,Dim,input,target,maxMC,sampimp,issoftmax);

l1 = Dim(1);	% Layer-1 dimension
l2 = Dim(2);	% Layer-2 dimension
N = size(input,1);	% Number instances

% Get param matrix from parameter vector
w = reshape(VV,l1+1,l2);

eps = 1e-10;
f = 0;
df = zeros(size(w));
for iterMC=1:maxMC
	% Append 1s to the input and compute output activations
	temp = [input(:,:,iterMC) repmat(1,[N,1])];

	if issoftmax
		% Softmax cross entropy for output layer
		act = exp(temp*w);
		act = act ./ repmat(row_sum(act),1,l2);
		act(find(act <= eps)) = eps;
		act(find(act >= 1-eps)) = 1-eps;
		
		f = f - sampimp(:,iterMC)'*row_sum(target(:,:,iterMC).*log(act));
	else
		act = 1./(1+exp(-temp*w));
		act(find(act <= eps)) = eps;
		act(find(act >= 1-eps)) = 1-eps;
		
		% Logistic cross entropy for other layers
		f = f - sampimp(:,iterMC)'*row_sum(target(:,:,iterMC).*log(act)+(1-target(:,:,iterMC)).*log(1-act));
	end

	% Gradient
	D_sampimp = sparse(1:N,1:N,sampimp(:,iterMC));
	df = df - temp'*D_sampimp*(target(:,:,iterMC)-act);
end
df = df(:);
