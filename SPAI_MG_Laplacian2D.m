%Solve 2D Poisson equation with Geometric Multgrid + SAI smoother
clear
set(0, 'defaultaxesfontsize',18,'defaultaxeslinewidth',1.5,...
    'defaultlinelinewidth',2,'defaultpatchlinewidth',1.5,...
    'defaulttextfontsize',18,'defaulttextInterpreter','latex');
lsty ={'x-','d--','^-.','v:','>-','<-','p-','h-','-o','-*'};
global LEVEL0 omega
fprintf('N\t Rel. Res.\t  Error \t iter \t CPU \t Rate\n');
maxit=100;tol=1e-10;  LEVEL0=2; 
omega=2;cycle='W';mu1=1; mu2=0; %W(1+0) cycle
%omega=1;cycle='V';mu1=1; mu2=1; %V(1+1) cycle
%Example 1
exname='Ex1';
U0=@(x,y) (x.^2-x.^4).*(y.^4-y.^2); %BC=0
fF=@(x,y) 2*(1-6*x.^2).*(y.^2-y.^4)+2*(1-6*y.^2).*(x.^2-x.^4);
%Example 2, uncomment below to run example 2
% exname='Ex2';
% U0=@(x,y) x.*log(x).*y.*log(y);
% fF=@(x,y) - (x.*log(x))./y - (y.*log(y))./x;

smlist={'M_J','M_5','M_9'};%smoother list
for kk=[1:3]
    smtype=smlist{kk}; %ASe, M_J, GS, RB-GS
    fprintf('--------------Smoother type: [%s]-------------------\n',smtype)
    for fineL=12:12
        tic
        xmin=0;xmax=1;   mg=[];
        for Level=fineL:-1:LEVEL0 %define the matrix at each level
            n=2^Level; h=(xmax-xmin)/n;m=n-1;e=ones(m,1); E=speye(m^2); 
            mg(Level).A=(1/h^2)*gallery('poisson',n-1); %5-pint stencil
            %interpolation operator in matrix form
            Pn=(1/2)*spdiags([e 2*e e],-2:0,m,m);
            mg(Level).P=kron(Pn(:,1:2:end-2),Pn(:,1:2:end-2));

            mg(Level).Mfun=[];mg(Level).w=1;
            switch(smtype)
                case 'M_J' %weighted Jacobi smoother
                    mg(Level).w=4/5;
                    mg(Level).Mfun=@(v) (h^2/4)*v;
                case 'M_5' %optimal 5-point SAI smoother
                    mg(Level).w=1/4;
                    Me2=spdiags([e 3*e e],-1:1,m,m);  
                    mg(Level).Mfun=@(v) (h^2*8/41)*reshape((reshape(v,m,m)*Me2+Me2*reshape(v,m,m)),[],1);
                case 'M_9' %optimal 9-point SAI smoother 
                    mg(Level).w=(309-12*sqrt(10))/1720; %~0.1576
                    Me3=spdiags([e (10/3)*e e],-1:1,m,m); %
                    %M_9=(h^2/24)*(3*kron(Me3,Me3)+(32/3)*speye(m^2)) 
                    mg(Level).Mfun=@(v) (h^2/24)*(3*reshape((Me3*reshape(v,m,m)*Me3),[],1)+(32/3)*v);               
            end
        end
        A=mg(fineL).A; N=2^fineL;   h=(xmax-xmin)/N;  
        xgrid=xmin+(1:N-1)'*h;  ygrid=xgrid;
        [X,Y] = meshgrid(xgrid,ygrid);
        F=fF(X,Y);  b=F(:); 
        x=rand(size(b)); r0=h*norm(b-A*x);rk=r0;
        err0=[]; it=1; err0(it)=rk;
        while((it<=maxit)&&(rk>=r0*tol))
            it=it+1;
            x=mg_iter_2d(mg,x,b,fineL,mu1,mu2);
            rk=h*norm(b-A*x); err0(it)=rk;
        end
        cpu=toc;%CPU time  
        iter=length(err0)-1;%total iterations
        res=err0(end)/r0; rate=res^(1/iter);%convergence rate
        S0=U0(X,Y); err=norm(S0(:)-x,inf);% true solution, max error
        fprintf('%d&\t %1.1e & \t %1.1e &\t %d&\t  %1.2f\t%1.3f \\\\ \n',N,res,err,iter,cpu,rate);
    end
    semilogy(0:length(err0)-1,err0/r0,lsty{kk},'DisplayName',sprintf('%s(CPU=%1.2f s, iter=%d, rate=%1.3f)',smtype,cpu,iter,rate));axis tight; hold on
end
set(gcf, 'Position', get(0, 'Screensize').*[1 1 1/2 1/2]);
legend('-DynamicLegend','Location', 'NorthEast');
title(sprintf('5-point 2D Lapalcian: GMG with $N=%d$ and %d-levels',N,fineL-LEVEL0+1));
xlabel(sprintf('%s(%d+%d)-cycles',cycle,mu1,mu2)); ylabel('Relative residual norms')
figname=sprintf('../Manuscript/OptimalSmoother/MGplot2D_%s_%s',exname,cycle);
%print(figname,'-depsc')

function [x]=mg_iter_2d(mg,x0,b,level,pre,post) %multigrid algoritm
global LEVEL0 omega
if(level==LEVEL0) %Coarest level
    x=mg(level).A\b;
else
    x = mg_smooth(mg,x0,b,pre,level);
    r = b - mg(level).A*x; rc=(mg(level).P)'*r/4;
    cc = mg_iter_2d(mg,zeros(size(rc)),rc,level-1,pre,post);% coarse grid correction
    if(omega==2)
        cc = mg_iter_2d(mg,cc,rc,level-1,pre,post); %add this for W cycle
    end
    x = x + mg(level).P*cc; %
    x = mg_smooth(mg,x,b,post,level);
end
end
function [x]=mg_smooth(mg,x,b,nv,lev) %smoothing iteration
for k = 1:nv
    x=x + mg(lev).w*mg(lev).Mfun(b - mg(lev).A*x);
end
end



