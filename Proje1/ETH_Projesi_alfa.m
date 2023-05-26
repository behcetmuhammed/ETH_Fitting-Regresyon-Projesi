clc; clear; clear all; close all;

%uiopen('C:\Users\emira\OneDrive\Masaüstü\ETH_Fitting Regresyon Projesi\Proje2\applerevenue .csv',1)
%load('applerevenue.mat') %10559x6
ETHfitting2 = table2array(applerevenue);

s1 = normalize(ETHfitting2(:,1),'range');
ETHfitting2(:,1)=s1;
s2 = normalize(ETHfitting2(:,2),'range');
ETHfitting2(:,2)=s2;
s3 = normalize(ETHfitting2(:,3),'range');
ETHfitting2(:,3)=s3;
s4 = normalize(ETHfitting2(:,4),'range');
ETHfitting2(:,4)=s1;
s5 = normalize(ETHfitting2(:,5),'range');
ETHfitting2(:,5)=s5;


n = randperm(10559);

input_train = ETHfitting2(n(1:7500),1:5)';
input_test = ETHfitting2(n(7501:end),1:5)';
target_train = ETHfitting2(n(1:7500),6)';
target_test = ETHfitting2(n(7501:end),6)';

%nnstart;

hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize);
net.trainFcn='traingda';
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
net.trainParam.epochs = 1000;
net.trainParam.goal = 0;
net.trainParam.lr = 0.01;
net.performFcn = 'mse';

[net,tr] = train(net,input_train,target_train);

op_test = net(input_test);
err = mse(target_train,op_test)
figure, plotregression(target_test,op_test)
performance = perform(net,target_test,op_test)

%%
net1 = net_1.Network;

%5. satır
deneme1 = net1([0	0.688442211055276	0.327868852459016	0	0.198581560283688	0.943637916310845	0.200000000000000	1]');
deneme1_s = vec2ind(round(deneme1));

%88. satır
deneme2 = net1([0.117647058823529	0.502512562814070	0.557377049180328	0.117647058823529	0.0839243498817967	0.105038428693424	0.0833333333333333	0]');
deneme2_s = vec2ind(round(deneme2));

%368. satır
deneme3 = net1([0	0.507537688442211	0.524590163934426	0	0	0.0742954739538856	0	0]');
deneme3_s = vec2ind(round(deneme3));

%659. satır
deneme4 = net1([0.647058823529412	0.638190954773869	0.868852459016394	0.647058823529412	0	0.0478223740392827	0.500000000000000	0]');
deneme4_s = vec2ind(round(deneme4));

%762. satır
deneme5 = net1([0.529411764705882	0.854271356783920	0.606557377049180	0.529411764705882	0	0.138770281810418	0.366666666666667	1]');
deneme5_s = vec2ind(round(deneme5));

