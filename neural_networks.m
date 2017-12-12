clc,clear
load data.txt
input_train=data(1:506,1:13)';
output_train=data(1:506,14)';
input_test=data(1:506,1:13)';
output_test=data(1:506,14)';
[inputn,inputps]=mapminmax(input_train);
[outputn,outputps]=mapminmax(output_train);
inputn_test=mapminmax('apply',input_test,inputps);
net2=feedforwardnet(9,'trainbr');             %训练函数还可选feedforwardnet(10,'trainlm')
                                                                %网络类型还可以选fitnet(9,'trainbr'); 
net2=train(net2,inputn,outputn);               
bn=net2(inputn_test);
BPoutput=mapminmax('reverse',bn,outputps)
delta=abs(output_test-BPoutput)./output_test
g=1:506
figure(1),plot(g,delta,'r-')
figure(2),plot(g,output_test,'-')
hold on
plot(g,BPoutput,'r-')

