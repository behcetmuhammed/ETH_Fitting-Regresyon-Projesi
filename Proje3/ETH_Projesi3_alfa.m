clc; clear; clear all; close all;

%uiopen('C:\Users\emira\OneDrive\Masaüstü\ETH_MLP Sınıflandırma Projesi\Diyabet_Verileri.xls',1)
load('Syr_Table_Fitting.mat') %35x7
ETHfitting3 = table2array(MLPSnf);

s1 = normalize(ETHfitting3(:,1),'range');
ETHfitting3(:,1)=s1;
s2 = normalize(ETHfitting3(:,2),'range');
ETHfitting3(:,2)=s2;
s3 = normalize(ETHfitting3(:,3),'range');
ETHfitting3(:,3)=s3;
s4 = normalize(ETHfitting3(:,4),'range');
ETHfitting3(:,4)=s1;
s5 = normalize(ETHfitting3(:,5),'range');
ETHfitting3(:,5)=s5;
s6 = normalize(ETHfitting3(:,6),'range');
ETHfitting3(:,6)=s6;

n = randperm(35);

input_train = ETHfitting3(n(1:25),1:6)';
input_test = ETHfitting3(n(26:end),1:6)';
target_train = ETHfitting3(n(1:25),7)';
target_test = ETHfitting3(n(26:end),7)';

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

%1. satır
deneme1 = net1([0	0.0313111545988258	0.0999367488931056	0	0.0163934426229508	0.307692307692308]');
deneme1_s = vec2ind(round(deneme1));

%7. satır
deneme2 = net1([0.176470588235294	0.253424657534247	0.00379506641366224	0.176470588235294	0	0.0615384615384615]');
deneme2_s = vec2ind(round(deneme2));

%22. satır
deneme3 = net1([0.617647058823529	0	0.00759013282732448	0.617647058823529	0	0.123076923076923]');
deneme3_s = vec2ind(round(deneme3));


