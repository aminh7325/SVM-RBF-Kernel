function t = BackTracking(XTrain,YTrain,Alpha , C , Mu , AlphaBT , BetaBT ,Gamma , t_Initial , V)
%   Detailed explanation goes here
[GF HF] = Comp_H_G(XTrain, YTrain, Alpha, Mu, C, Gamma);
Alphaprime = Alpha + t_Initial *V;
while(1)
    F = Comp_F(XTrain,YTrain,Alphaprime , C , Mu , Gamma);
    while(1)
        if(imag(F) == 0)
            break;
        end
        t_Initial = BetaBT * t_Initial;
        Alphaprime = Alpha + t_Initial *V;
        F = Comp_F(XTrain,YTrain,Alphaprime , C , Mu , Gamma);
    end
    if (Comp_F(XTrain,YTrain,Alphaprime , C , Mu , Gamma) <= Comp_F(XTrain,YTrain,Alpha , C , Mu , Gamma) + AlphaBT*t_Initial*GF'*V)
        break;
    end
    t_Initial = BetaBT * t_Initial;
    Alphaprime = Alpha + t_Initial * V;
   
end
t = t_Initial;
end

