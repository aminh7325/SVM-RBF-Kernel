% Opening The svm file
%clear;
clc;
tic;
File = fopen('svm_train.txt' , 'r');
A = fscanf(File , '%f');
lengthA = length(A);
count = 1;
X1 = zeros(lengthA/3 , 1);
X2 = zeros(lengthA/3 , 1);
YTrain = zeros(lengthA/3 , 1);
for i=1:3:lengthA
   X1(count) = A(i);
   X2(count) = A(i+1);
   YTrain(count) = A(i+2);
   count = count + 1;
end
XTrain = cat(2 ,  X1 , X2);


GammaList = [10 , 50 , 100 , 500] ;
CList = [0.01 , 0.1 , 0.5 , 1] ;
%Choose Gamma and C
i_G=4;
i_C=2;
        Gamma = GammaList(i_G);
        C = CList(i_C);
        N = size(XTrain,1);
        K = zeros(N,N);
            for i=1:N % RBF kernel
                for j = 1:N
                    K(i,j) = exp(-1*Gamma*sum((XTrain(i,:)-XTrain(j,:)).^2));
                end 
            end
%CVX optimization
T_Start = cputime;
cvx_begin
    variable alpham(N);
    maximize (-0.5.*quad_form(YTrain.*alpham,K) + ones(N,1)'*(alpham));
    subject to
        alpham >= 0;
        YTrain'*alpham == 0;
        alpham <= C;
cvx_end
TCVX = cputime - T_Start;
%Displaying the results.
disp('Results of Optimization with CVX:')
disp({'C =' , num2str(C) ; 'Gamma=' , num2str(Gamma) ; 'OptimalValue =' , num2str(cvx_optval)});

%Calculating w'x + b
w = 0;
for i=1:N
   if alpham(i) > 1e-5 && alpham(i) < 1/2*C
        S = i;
        break;
   end
end
for i=1:N
   if alpham(i) > 1e-5
    w = w + alpham(i)*YTrain(i)*exp(-1*Gamma*sum((XTrain(S,:)-XTrain(i,:)).^2));
   end
end

b = YTrain(S) - w;

% plot train data

SupVec = zeros(N,1);
countcvx = 0;
for i = 1:N
    if(alpham(i) > 1e-5)
       SupVec(i) = 1;
       countcvx = countcvx + 1;
    end
end
figure;

gscatter(XTrain(:,1),XTrain(:,2),YTrain);
hold on

X1Plot = linspace(0 , 1.1 , 200);
X2Plot = linspace(0.2 , 1.1 , 200);
[X,Y] = meshgrid(X1Plot,X2Plot);
Z = b;
f = zeros(size(X));
for i=1:size(X,1)
    for j=1:size(Y,1)
            xnew = [X(i,j) , Y(i,j)];
            wx = 0;
            for k=1:863
                if alpham(k) > 1e-5
                    wx = wx + alpham(k) *YTrain(k) * exp(-Gamma * ((xnew(1)-XTrain(k,1))^2 + (xnew(2)-XTrain(k,2))^2));
                end
            end
            f(i,j) = wx + b;
     end
end
contour(X,Y,f,[0,0] , 'k');
contour(X,Y,f,[-1,-1] , 'r--');
contour(X,Y,f,[1,1]  , 'b--' );
for i = 1:N
    if(SupVec(i))
        plot(XTrain(i,1), XTrain(i,2), 'ko', 'Markersize', 10);
        hold on;
    end
end
legend('y = 1' , 'y = -1' , 'DecisionBorder' , 'Margin y = 1' , 'Margin y = -1' , 'SV');
title(['CVX Optimization: SVM Model with C=' , num2str(C) , ' Gamma =' , num2str(Gamma)]);
disp({'Support Vectors =' , num2str(countcvx)});


%%
%%Part4,5
File = fopen('svm_train.txt' , 'r');
A = fscanf(File , '%f');
lengthA = length(A);
count = 1;
X1 = zeros(lengthA/3 , 1);
X2 = zeros(lengthA/3 , 1);
YTrain = zeros(lengthA/3 , 1);
for i=1:3:lengthA
   X1(count) = A(i);
   X2(count) = A(i+1);
   YTrain(count) = A(i+2);
   count = count + 1;
end
XTrain = cat(2 ,  X1 , X2);
%Parameters
AlphaBT = 0.01;
BetaBT = 0.5;
Epsilone = 1e-8;
Mu = 1e-4;
C = 0.1;
Gamma = 500;
Alpha_Initial = 9*C/10*ones(863, 1) + 0.01*rand(863, 1); 
T_Initial = 1e-4;
GF = zeros(863, 1);
HF = zeros(863, 863);
Iteration = 1;
T_Start = cputime;
%newton's method
while(2*863 * Mu > Epsilone)
    T_Initial = 1e-4;
    while(1)
        [GF, HF] = Comp_H_G(XTrain, YTrain, Alpha_Initial, Mu, C, Gamma);
        V = zeros(863, 1);
        W = 0;
        A = [HF YTrain; YTrain' 0];
        Temp = inv(A) * [-GF ; 0];
        V = Temp(1:863);
        T_Initial = 1e-4;
        t = BackTracking(XTrain,YTrain,Alpha_Initial , C , Mu , AlphaBT , BetaBT ,Gamma , T_Initial , V);
        Alpha_Initial = Alpha_Initial + t*V;
        if norm(t*V) < 1e-6
            Alpha_Opt = Alpha_Initial;
            break;
        end
    end
    OptVal_Barrier(Iteration) = Comp_F(XTrain,YTrain,Alpha_Opt , C , Mu , Gamma);
    Iteration = Iteration + 1;
    Mu = Mu * 0.5;
end
disp('Results of Optimization with Barrier:')
disp({'C =' , num2str(C) ; 'Gamma=' , num2str(Gamma) ; 'OptimalValue =' , num2str(OptVal_Barrier(25))});
TBarrier = cputime - T_Start;
%%
w = 0;
N = 863;
for i=1:863
   if Alpha_Opt(i) > 1e-5 
        S = i;
        break;
   end
end
for i=1:863
   if Alpha_Opt(i) > 1e-5
    w = w + Alpha_Opt(i)*YTrain(i)*exp(-1*Gamma*sum((XTrain(S,:)-XTrain(i,:)).^2));
   end
end

b = YTrain(S) - w;

% plot train data

SupVec = zeros(N,1);
count = 0;
for i = 1:N
    if(Alpha_Opt(i) > 1e-5)
       SupVec(i) = 1;
       count = count + 1;
    end
end
figure;

gscatter(XTrain(:,1),XTrain(:,2),YTrain);
hold on

X1Plot = linspace(0 , 1.1 , 200);
X2Plot = linspace(0.2 , 1.1 , 200);
[X,Y] = meshgrid(X1Plot,X2Plot);
Z = b;
f = zeros(size(X));
for i=1:size(X,1)
    for j=1:size(Y,1)
            xnew = [X(i,j) , Y(i,j)];
            wx = 0;
            for k=1:863
                if Alpha_Opt(k) > 1e-5
                    wx = wx + Alpha_Opt(k) *YTrain(k) * exp(-Gamma * ((xnew(1)-XTrain(k,1))^2 + (xnew(2)-XTrain(k,2))^2));
                end
            end
            f(i,j) = wx + b;
     end
end
contour(X,Y,f,[0,0] , 'k');
contour(X,Y,f,[-1,-1] , 'r--');
contour(X,Y,f,[1,1]  , 'b--' );
legend('y = 1' , 'y = -1' , 'DecisionBorder' , 'Margin y = 1' , 'Margin y = -1');
title(['Barrier Optimization: SVM Model with C=' , num2str(C) , ' Gamma =' , num2str(Gamma)]);
figure;
Iteration_X = 1:1:Iteration-1;
plot(Iteration_X , OptVal_Barrier);
title('Value Vs Iteration');
xlabel('Iteration');
ylabel('Function Value');
%%
%Part 6
FCVX = Comp_F(XTrain, YTrain, alpham, C, Mu, Gamma);
FBarrier = Comp_F(XTrain, YTrain, Alpha_Opt, C, Mu, Gamma);
disp('Compare:');
disp({'CVXTime =' , num2str(TCVX) ; 'BarrierTime=' , num2str(TBarrier)});
disp({'CVXOptVal =' , num2str(FCVX) ; 'BarrierOptVal=' , num2str(FBarrier)});
disp({'CVX_SV =' , num2str(countcvx) ; 'Barrier_SV=' , num2str(count)});
