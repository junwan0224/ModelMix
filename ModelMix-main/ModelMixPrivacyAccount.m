clc;
clear;


%% number of samples in the training dataset e.g. CIFAR 10 has 50000 training samples 
n = 50000;
%% l_infty sensitivity guarantee if used 
p = 100;

%% minibatch size
B = 1500;
%% sampling rate 
q = B/n;
%% clipping norm
c = 20;

%% number of iteration
comp = 3400;


%% sensitivity of gradient in each iteration
sen =  c/sqrt(p)/(q*n); 

%% noise ratio
nr  = 0.335;

%% magnitude of Gaussian noise added to the gradient in each iteration 
sigma = nr*c/(q*n);

step = sigma/25;

%% ModelMix coordinate-wise distance 
tau = 0.15;

%% desired delta parameter in (eps, delta)-DP gaurantee
delta = 1/(2*n);



index = 0;
Alpha = 19;
RenyiMMKey = zeros(1,Alpha);
RenyiMMDP = zeros(1,Alpha);
RenyiPureDP = zeros(1,Alpha);
    

%% unbounded 
for k = 1:1:Alpha

    x = -8*sigma-tau/2:step:tau/2+8*sigma;
    num = denGau(x,tau,sen,sigma).^k;
    den = denGau(x,tau,0,sigma).^(k-1);
    item = sum(num./den) * step;
    
    RenyiMMKey(k) = item^p;
    RenyiPureKey(k) = exp(p*((sen)^2*(k^2-k)/(2*sigma^2)));

end


for alpha = 1:1:Alpha
    index = index +1;
    RenyiMMDP(index) = (1-q)^alpha;
    RenyiPureDP(index) = (1-q)^alpha;
    for k  = 1:alpha
        RenyiMMDP(index) = RenyiMMDP(index) + nchoosek(alpha, k)*(1-q)^(alpha-k)*q^(k)*RenyiMMKey(k);
        RenyiPureDP(index) = RenyiPureDP(index)+ nchoosek(alpha, k)*(1-q)^(alpha-k)*q^(k)*RenyiPureKey(k);
    end

    MM(index) = (log(RenyiMMDP(index))*(comp)+log(1/delta))/(alpha-1);
    Or(index) = (log(RenyiPureDP(index))*(comp)+log(1/delta))/(alpha-1);
end


%% eps for modelmix
EspModelMix = min(MM(1:index))

%% eps for original DP-SGD 
EspOriginalDPSGD = min(Or(1:index))





function Ren = RenyiMixSim(alpha,sen,sigma)
         fun = @(x,y) 1/(sqrt(2*pi*sigma^2)) * (1/(sen)* exp(-((x-y).^2)/(2*sigma^2))).^(alpha)./(exp(-x.^2/(2*sigma^2))).^(alpha-1);
         Ren = integral2(fun,-5*(sigma),5*(sigma),-sen/2,sen/2);
end

function Ren = RenyiOrSim(alpha,sen,sigma)
         fun = @(x) (1/(sqrt(2*pi*sigma^2)))* (exp(-(x-sen).^2/(2*sigma^2))).^(alpha)./((exp(-x.^2/(2*sigma^2))).^(alpha-1));
         Ren = integral(fun,-10*(sigma),10*(sigma));
end




function Ren = Renyimix(alpha,sen,sigma,p)
         fun = @(x,y) ((1-p)/(sqrt(2*pi*sigma^2))* exp(-x.^2/(2*sigma^2))+p/(sen*sqrt(2*pi*sigma^2)).* exp(-((x-y-sen).^2)/(2*sigma^2))).^(alpha+1)./(exp(-x.^2/(2*sigma^2))).^(alpha);
         Ren = integral2(fun,-10*(sen+sigma),10*(sen+sigma),0,sen);
end

function Prob = RenyiProbpure(sen,sigma,p)
         fun = @(x) ((1-p)./(sqrt(2.*pi.*sigma.^2)) .* exp(-x.^2/(2.*sigma.^2))+p./(sqrt(2.*pi.*sigma.^2)) .* exp(-(x-sen).^2/(2.*sigma.^2)));
         Prob = integral(fun,-30*(sen+sigma),30*(sen+sigma));
end



function Ren = Renyipure(alpha,sen,sigma,p)
         fun = @(x) ((1-p)./(sqrt(2.*pi.*sigma.^2)) .* exp(-x.^2/(2.*sigma.^2))+p./(sqrt(2.*pi.*sigma.^2)) .* exp(-(x-sen).^2/(2.*sigma.^2))).^(alpha+1)./(1/(sqrt(2*pi*sigma^2)) .* exp(-x.^2/(2*sigma^2))).^(alpha);
         Ren = integral(fun,-5*(sen+sigma),5*(sen+sigma));
end


function den = denGau(x,tau,a,sigma)
%          fun = @(y) 1/(tau*sqrt(2*pi*sigma^2)) .* exp(-((x-y-a).^2)/(2*sigma^2));
%          den = integral(fun,-tau/2,tau/2);
           den =  integral(@(y) 1/(tau*sqrt(2*pi*sigma^2)) .* exp(-((x-y-a).^2)/(2*sigma^2)),...
                           -tau/2,tau/2,'ArrayValued',true);

end




function den = denGausuperMix(x,tau,sen,a,sigma)
          fun = @(y,z) 1/(sen*tau*sqrt(2*pi*sigma^2)) .* exp(-((x-y-a-z).^2)/(2*sigma^2));
          den = integral2(fun,-tau/2,tau/2,0,sen);
end


function den = GauRenyiKey(tau,sigma,sen,alpha)
          fun = @(x,y) (1/(tau*sqrt(2*pi*sigma^2)) .* exp(-((x-y-sen).^2)/(2*sigma^2))).^(alpha)./((1/(tau*sqrt(2*pi*sigma^2)).* exp(-((x-y).^2)/(2*sigma^2))).^(alpha-1));
          den = integral2(fun,(-5*sigma-tau/2),(5*sigma+tau/2),-tau/2,tau/2);
end


function den = denGauPure(x,sigma)
         den =  1/(sqrt(2*pi*sigma^2)) * exp(-x^2/(2*sigma^2));
end



function den = denLap(x,tau,a,sigma)
         fun = @(y) 1/(2*sigma) .* exp(-(abs(x-y-a))./sigma);
         den = integral(fun,0,tau);
end

function C = combi(n,k)
%          c =1;
%          for i=1:k
%              c = c*(n-i+1)/i;
%          end
%          C = c;
         C = gamma(n+1)/gamma(k+1)/gamma(n-k+1);
end