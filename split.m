function [TrainMat, TrainLabel, TestMat, TestLabel] = split(data, label, sub_info, i)
% Input: 
%   data - a NxM matrix that contains the full list of features of Taiji data. 
% N is the number of frames and M is the dimension of the feature. 
%   label - a N vector that contains form label of each frame.
%   sub_info - a Nx2 matrix that contains [subject_id, take_id] of each
%   frame
%   i - a selected subject id to be used for testing

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% get test data 
test_list = sub_info(:,1) == i;
train_list = sub_info(:,1) ~= i;

TrainMat = data(train_list,:);
TrainLabel = label(train_list,:);

TestMat = data(test_list,:);
TestLabel = label(test_list,:);

end