function topfeatures = rankingfeat(TrainMat, LabelTrain)
%% input: TrainMat - a NxM matrix that contains the full list of features
%% of training data. N is the number of training samples and M is the
%% dimension of the feature. So each row of this matrix is the face
%% features of a single person.
%%        LabelTrain - a Nx1 vector of the class labels of training data

%% output: topfeatures - a Kx2 matrix that contains the information of the
%% top 1% features of the highest variance ratio. K is the number of
%% selected feature (K = ceil(M*0.01)). The first column of this matrix is
%% the index of the selected features in the original feature list. So the
%% range of topfeatures(:,1) is between 1 and M. The second column of this
%% matrix is the variance ratio of the selected features.
X=TrainMat;
Y=LabelTrain;
C=length(countcats(categorical(Y)))-1; %classes
%Var = 0;%zeros(1,width(X));
VR = zeros(1,width(X));
Var = var(X);
Var_k = zeros(1,size(X,2));
for k=1:C
    %Var_k = X(Y==k);
    Var_temp(1,:)=var(X(Y==k));
    Var_k = Var_k+Var_temp;
    
end
for i=1:width(X)
    if VR(1,i) ==0
        continue
    end
    VR(1,i)=C*Var(1,i)/Var_k(1,i);
end

    
K = ceil(width(X)*1); 
[variance,I] = sort(VR);
topfeatures = transpose([(I(1:K)); (variance(1:K))]);


return

            