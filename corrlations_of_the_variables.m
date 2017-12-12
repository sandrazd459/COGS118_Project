%the corrlations of the variables

clc,clear
load data.txt
m=size(data,1)
x0=data(:,[1:13]);
y=data(:,14);
x1=x0(:,1);
x2=x0(:,2);
x3=x0(:,3);
x4=x0(:,4);
x5=x0(:,5);
x6=x0(:,6);
x7=x0(:,7);
x8=x0(:,8);
x9=x0(:,9);
x10=x0(:,10);
x11=x0(:,11);
x12=x0(:,12);
x13=x0(:,13);
figure(1),plot(x1,y,'*');corrcoef(x1,y)
figure(2),plot(x2,y,'*');corrcoef(x2,y)
figure(3),plot(x3,y,'*');corrcoef(x3,y)
figure(4),plot(x4,y,'*');corrcoef(x4,y)
figure(5),plot(x5,y,'*');corrcoef(x5,y)
figure(6),plot(x6,y,'*');corrcoef(x6,y)
figure(7),plot(x7,y,'*');corrcoef(x7,y)
figure(8),plot(x8,y,'*');corrcoef(x8,y)
figure(9),plot(x9,y,'*');corrcoef(x9,y)
figure(10),plot(x10,y,'*');corrcoef(x10,y)
figure(11),plot(x11,y,'*');corrcoef(x11,y)
figure(12),plot(x12,y,'*');corrcoef(x12,y)
figure(13),plot(x13,y,'*');corrcoef(x13,y)
corrcoef(data)
