function F = Comp_F(XTrain,YTrain,Alpha , C , Mu , Gamma)
%   Detailed explanation goes here
Sum = 0;
for i=1:length(Alpha)

        Sum = Sum - Mu * log(Alpha(i)) - Mu * log(C - Alpha(i));

end
N = size(XTrain,1);
    K = zeros(N,N);
    for i=1:N % RBF kernel
        for j = 1:N
                K(i,j) = exp(-1*Gamma*sum((XTrain(i,:)-XTrain(j,:)).^2));
        end 
    end

F = (-1/2)*Alpha'*(diag(YTrain) * K * diag(YTrain))*Alpha + ones(863, 1)' * Alpha + Sum;
end

