%%Term Project PRML

clear all;
close all;
clc;

%% load the Taiji data 
load('Taiji_data_cls_upd.mat')

frame_num = size(Taiji_data,1);
form_num = size(keyframes,1);
form_list = linspace(1,form_num,form_num);

%% Labelling each frame
M = 100; N = 20;
labels = zeros(frame_num, 1);
for i=1:frame_num
    take_id = sub_info(i,2); 
    frame_idx = Taiji_data(i,end);      % the frame index
    % label the frame to a specific Taiji key form (or "0" indicates the NON KEY FRAME)
    form_idx = 0;
    [val,idx] = min(abs(frame_idx - keyframes(:,take_id)));
    if ((frame_idx - keyframes(idx,take_id) < 0 && val <= M) || frame_idx - keyframes(idx,take_id) >= 0 && val <= N)
        form_idx = form_list(idx);
    end
    labels(i) = form_idx;
end



for i=1:1
    
    %% divide into train and test sets with "Leave one subject out"
    [TrainMat, LabelTrain, TestMat, LabelTest]= split(Taiji_data, labels, sub_info,i);
    
    %Split into traianing and validation data
    
    valMat = TrainMat(end-0.4*length(TrainMat):end,:); 
    Labelval = LabelTrain(end-0.4*length(LabelTrain):end,1);
    TrainMat =TrainMat(1:0.6*length(TrainMat),:);
    LabelTrain= LabelTrain(1:0.6*length(LabelTrain),1);
%     TestMat = TestMat(:,1:220);
    

    %% start feature ranking
    topfeatures = rankingfeat(TrainMat, LabelTrain);
    %topfeatures =topfeatures(1:220,:);

    
    %% start classification
    height = 1;
    width = length(topfeatures);
    channels = 1;
    %sampleSize = 114450;
    TrainMat = TrainMat(:,topfeatures(:,1));
    TestMat = TestMat(:,topfeatures(:,1));
    valMat = valMat(:,topfeatures(:,1));
    CNN_TrainingData = reshape(TrainMat,[height, width, channels, length(TrainMat)]);
    CNN_TrainingLabels = categorical(LabelTrain);
    CNN_TestingData = reshape(TestMat,[height, width, channels, length(TestMat)]);
    CNN_TestingLabels = categorical(LabelTest);
    CNN_valData = reshape(valMat,[height, width, channels, 45781]);
    CNN_valLabels = categorical(Labelval);

    numEpochs = 20; 
    batchSize = 160;
    % Build the CNN layers
    
    InputLayer = imageInputLayer([height,width,channels]); %'DataAugmentation', 'none'); %'Normalization', 'none');
    c1 = convolution2dLayer([1 5], 60,'stride',[1 1]); %Filter window size = [1 5], No of filters = 80, stride = [1 10]
    b1 =batchNormalizationLayer;
    r1 = reluLayer();
    p1 = maxPooling2dLayer([1 10],'stride',[1 1]); %PoolSize = [1 20], Stride = [1 10]
    
    c2 = convolution2dLayer([1 3], 70,'stride',[1 1]); %Filter window size = [1 5], No of filters = 90, stride = [1 10]
    b2 =batchNormalizationLayer;
    r2 = reluLayer();
    p2 = maxPooling2dLayer([1 10],'stride',[1 1]); %PoolSize = [1 20], Stride = [1 10]
    
    c3 = convolution2dLayer([1 3], 120,'stride',[1 1]); %Filter window size = [1 5], No of filters = 120, stride = [1 10]
    b3 =batchNormalizationLayer;
    r3 = reluLayer();
    p3 = maxPooling2dLayer([1 10],'stride',[1 1]); %PoolSize = [1 20], Stride = [1 10]
    f2 = fullyConnectedLayer(100);
    d1= dropoutLayer(.25);
    f1 = fullyConnectedLayer(44); %Reduce to three output classes
    
    s1 = softmaxLayer();
    
    outputLayer=classificationLayer();
    convnet = [InputLayer; c1;b1; r1; p1;c2;b2; r2; p3;c3;b3; r3; p3;d1; f1; s1; outputLayer] %c2;b2;r2;p2;c3;b3;r3;p3;f2;

    opts = trainingOptions('sgdm','MaxEpochs',20,'InitialLearnRate',1e-3,'MaxEpochs',numEpochs); %Optimise using stochastic gradient descent with momentum
    [convnet, info1] = trainNetwork(CNN_TrainingData, CNN_TrainingLabels, convnet, opts);

    YTrain = classify(convnet,CNN_TrainingData);
    train_acc = mean(YTrain==CNN_TrainingLabels)
    cfmt=confusionmat(CNN_TrainingLabels,YTrain);
    classfmat_train = cfmt./(meshgrid(countcats(categorical(CNN_TrainingLabels)))');
    conf_mat(classfmat_train)
    f1=figure(1);
    heatmap(classfmat_train);
    title('Train Classification Matrix');
    saveas(f1,'Train Classifiaction Matrix.png');

    f2=figure(2);
    heatmap(cfmt);
    title('Train Confusion Matrix');
    saveas(f2,'Train Confusion Matrix.png');
    
    YTest = classify(convnet,CNN_TestingData);
    test_acc = mean(YTest==CNN_TestingLabels)
    cfmt=confusionmat(CNN_TestingLabels,YTest);
    classfmat_test = cfmt./(meshgrid(countcats(categorical(CNN_TestingLabels)))');
    conf_mat(classfmat_test)
    
    f3=figure(3);
    heatmap(classfmat_test);
    title('Test Classification Matrix');
    saveas(f3,'Test Classification Matrix.png');

    f4=figure(4);
    heatmap(cfmt);
    title('Train Confusion Matrix');
    saveas(f4,'Train Confusion Matrix.png');
    
    f11=figure(11)
    plotTrainingAccuracy_All(info1,numEpochs);
    title('Training Accuracy vs loss');
    saveas(f11,'accuracy.png');


    % Test on the validation data
    Yval = classify(convnet,CNN_valData);
    val_acc = mean(Yval==CNN_valLabels)
    cfmt=confusionmat(CNN_valLabels,Yval);
    classfmat_val = cfmt./(meshgrid(countcats(categorical(CNN_valLabels)))');
    conf_mat(classfmat_val)
   
    f5=figure(5);
    heatmap(cfmt);
    title('Val Confusion Matrix');
    saveas(f5,'Val Confusion Matrix .png');


    f6=figure(6);
    heatmap(classfmat_val);
    title('Val Classification Matrix');
    saveas(f6,'Val Classification Matrix.png');
    
    

    %%%%%%tsne visualization
    feat_train = activations(convnet,CNN_TrainingData,1,'OutputAs','rows');
    feat_test = activations(convnet,CNN_TestingData,1,'OutputAs','rows');
    feat_val = activations(convnet,CNN_valData,1,'OutputAs','rows');

    y_train = tsne(feat_train,'Standardize',true);

    h1=figure(8);
    grid on;
    gscatter(y_train(:,1),y_train(:,2),CNN_TrainingLabels(:));
    xlabel('feature 1');
    ylabel('feature 2');
    saveas(h1,'TSNE_visualization_train.png');

    y_train = tsne(feat_test,'Standardize',true);

    h2=figure(9);
    grid on;
    gscatter(y_train(:,1),y_train(:,2),CNN_TestingLabels(:));
    xlabel('feature 1');
    ylabel('feature 2');
    saveas(h2,'TSNE_visualization_test.png');

    y_train = tsne(feat_val,'Standardize',true);

    h3=figure(10);
    grid on;
    gscatter(y_train(:,1),y_train(:,2),CNN_valLabels(:));
    xlabel('feature 1');
    ylabel('feature 2');
    saveas(h3,'TSNE_visualization_val.png');


    
end
    
    
    
    
    
