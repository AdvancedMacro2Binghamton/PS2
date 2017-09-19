close all
delta=0.025              %depreciation rate
alpha=0.35               %capital share of income
beta=0.99                %discount rate
sigma=2                  %utility parameter


nbk=20                 %number of data points int he grid
nba=2                    %number of value for the shocks
crit=1                   %convergence criterion
epsi=1e-1                %convergence parameter


PI=[phh 1-phh;1-pll,pll] %transition matrix
pihh=0.977               %this is pi hh
pill=0.926               %p ll
ah=1.1
al=0.678
A=[ah al]                %stochastic matrix


kmin=0.1
kmax=5
k=linspace(kmin,kmax,nbk)% k grid
c=zeros(nbk,nba)
util=zeros(nbk,nba)
v=zeros(nbk,nba)
Tv=zeros(nbk,nba)


%%%%%%%%%%%%%%%%%%set up%%%%%%%%%%%%%%%%%%%%%

while crit>epsi
    for i=1:nbk
        for j=1:nba
            c=A(j)*k(i)^alpha+(1-delta)*k(i)-k
            neg=find(c<0)
            c(neg)=NaN
            util(:,j)=(c.^(1-sigma))/(1-sigma)
            util(neg,j)=-1e12
        end
        [Tv(i,:),pl(1,:)]=max(util+beta*(v*PI))
    end
    crit=max(abs(Tv-v))
    v=Tv
    i=i+1
end

plot(v(:,1))
hold on 
plot(v(:,2))


