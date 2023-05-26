clc; clear; clear all; close all;

uiopen('C:\Users\emira\OneDrive\Masaüstü\ETH_Fitting Regresyon Projesi\Proje2\Orman Yangınları Veri Kümesi.xlsx',1)
%load('OrmanYangnlarVeriKmesi.mat') %517x13
ETHfitting2 = table2array(OrmanYangnlarVeriKmesi);
a
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
s6 = normalize(ETHfitting2(:,6),'range');
ETHfitting2(:,6)=s6;
s7 = normalize(ETHfitting2(:,7),'range');
ETHfitting2(:,7)=s7;
s8=normalize(ETHfitting2(:,8),'range');
ETHfitting2(:,8)= s8;
s9=normalize(ETHfitting2(:,9),'range');
ETHfitting2(:,9)= s9;
s10=normalize(ETHfitting2(:,10),'range');
ETHfitting2(:,10)= s10;
s11=normalize(ETHfitting2(:,11),'range');
ETHfitting2(:,11)= s11;
s12=normalize(ETHfitting2(:,12),'range');
ETHfitting2(:,12)= s12;

n = randperm(517);

input_train = ETHfitting2(n(1:350),1:12)';
input_test = ETHfitting2(n(351:end),1:12)';
target_train = ETHfitting2(n(1:350),13)';
target_test = ETHfitting2(n(351:end),13)';

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

%4. satır
deneme1 = net1([0.875000000000000	0.571428571428571	0	0.875000000000000	0.953222453222450	0.333333333333331	0.792433537832304	9.09090909090909e-15	0.846938775510203	3.82978723404256e-15	3.12500000000000e-16	0]');
deneme1_s = vec2ind(round(deneme1));

%80. satır
deneme2 = net1([0	0	0.181818181818182	0	2.28690228690245e-14	0.121321321321319	0.574233128834341	7.07070707070707e-15	0.220408163265300	0.712765957446809	0	0]');
deneme2_s = vec2ind(round(deneme2));

%368. satır
deneme3 = net1([0.750000000000000	0.285714285714286	0.545454545454545	0.750000000000000	0.974012474012472	0.101401401401399	0.432924335378303	0.148484848484848	0.185714285714279	0.478723404255319	0	0.221887550200803]');
deneme3_s = vec2ind(round(deneme3));

%401. satır
deneme4 = net1([1	0.428571428571429	0.454545454545455	1	0.969854469854468	0.495495495495494	0.304396728016336	1.41414141414141e-14	2.04081632653063e-14	0.478723404255319	0	0.819277108433735]');
deneme4_s = vec2ind(round(deneme4));

%511. satır
deneme5 = net1([0.625000000000000	0.428571428571429	0.181818181818182	0.625000000000000	2.28690228690245e-14	0.167067067067065	0.769529652351730	0.717171717171717	0.185714285714279	0.574468085106383	0	4.31726907630522e-16]');
deneme5_s = vec2ind(round(deneme5));

