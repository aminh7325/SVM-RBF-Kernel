function [GradientF,HesianF] = Comp_H_G(XTrain, YTrain, Alpha, Mu, C, Gamma)
%   Detailed explanation goes here
  
    N = size(XTrain,1);
    K = zeros(N,N);
    for i=1:N
        for j = 1:N
                K(i,j) = exp(-1*Gamma*sum((XTrain(i,:)-XTrain(j,:)).^2));
        end 
    end
    
    Barrier_Grad = zeros(863, 1);
    Barrier_Hesian = zeros(863, 1);
    for i = 1:N
        Barrier_Grad(i) = (-1)/Alpha(i) + (-1)/(C - Alpha(i));
        Barrier_Hesian(i,i) = 1/Alpha(i)^2 + 1/((C - Alpha(i))^2);
    end
    XTild = diag(YTrain) * K; 
    GradientF = -diag(YTrain) * K * diag(YTrain) * Alpha + ones(863,1) + Mu * Barrier_Grad;
    HesianF = -diag(YTrain) * K * diag(YTrain) + Mu * Barrier_Hesian;
    
end

