%Solve 3D Poisson equation with Geometric Multgrid method + SAI smoother
clear
set(0, 'defaultaxesfontsize',16,'defaultaxeslinewidth',1.5,...
    'defaultlinelinewidth',2,'defaultpatchlinewidth',1.5,...
    'defaulttextfontsize',16,'defaulttextInterpreter','latex');
global LEVEL0 omega
fprintf('N\t Rel. Res.\t  Error \t iter \t CPU \t Rate\n');
maxit=100;tol=1e-10; mu1=1; mu2=0;
LEVEL0=2; omega=2; cycle='W'; %W(1+0) cycle
%omega=1;cycle='V'; mu2=1; %uncomment for testing V(1+1) cycle
smlist={'M_J','M_7'};
lsty ={'x-','d--','^-.','v:','>-','<-','p-','h-','.-'};
%Example 3
fF=@(x,y,z) 3*pi^2*sin(pi*x).*sin(pi*y).*sin(pi*z);
U0=@(x,y,z) sin(pi*x).*sin(pi*y).*sin(pi*z);

for kk=[1:2]
    smtype=smlist{kk};
    fprintf('--------------Smoother type: [%s]-------------------\n',smtype)
    for fineL=7
        tic;  xmin=0;xmax=1;ymin=0;ymax=1;  mg=[];           
        for Level=fineL:-1:LEVEL0 %define the matrix at each level
            nn=2^Level; h=(xmax-xmin)/nn; m=nn-1; e=ones(m,1);
            E=speye(m); E2=speye(m^2); L=spdiags([-e 2*e -e], -1:1, m, m);
            mg(Level).A=(1/h^2)*(kron(E2,L)+kron(kron(E,L),E)+kron(L,E2));

            %interpolation operator  in matrix form
            Pn=(1/2)*spdiags([e 2*e e],-2:0,m,m);
            mg(Level).P=kron(kron(Pn(:,1:2:end-2),Pn(:,1:2:end-2)),Pn(:,1:2:end-2));

            switch(smtype)
                case 'M_7' %7 point stencil
                    mg(Level).w=20/73;
                    S=h^2*spdiags([0.1*e 0.8/3*e 0.1*e], -1:1, m, m);
                    ES=kron(E,S); SE=kron(S,E);
                    %mg(Level).M=(kron(E,ES)+kron(E,SE)+kron(S,E2));
                    %mg(Level).Mfun=@(v) mg(Level).M*v;%matrix form of M_7
                    mg(Level).Mfun=@(v) reshape(ES*reshape(v,m^2,m)+SE*reshape(v,m^2,m)+reshape(v,m^2,m)*S,m^3,1);
                case 'M_J'
                    mg(Level).w=6/7;
                    mg(Level).M=1./diag(mg(Level).A);
                    mg(Level).Mfun=@(v) mg(Level).M.*v;
            end
        end
        A=mg(fineL).A;N=2^fineL;hx=(xmax-xmin)/N;
        xgrid=xmin+(1:N-1)'*hx;%interior nodes
        [X,Y,Z] = meshgrid(xgrid,xgrid,xgrid);
        S0=U0(X,Y,Z); F=fF(X,Y,Z); b=F(:);

        x=rand(size(b)); r0=hx*norm(b-A*x);rk=r0;
        err0=[]; it=1; err0(it)=rk;
        while((it<=maxit)&&(rk>=r0*tol)) %Begin Multigrid Cycles
            it=it+1;
            x=mg_iter_2d(mg,x,b,fineL,mu1,mu2);
            rk=hx*norm(b-A*x);
            err0(it)=rk;
        end
        cpu=toc; iter=length(err0)-1;%total V-cycle iterations
        res=err0(end)/r0;  rate=res^(1/iter);
        semilogy(0:length(err0)-1,err0/r0,lsty{kk},'DisplayName',sprintf('%s(CPU=%1.2f s, iter=%d, rate=%1.3f)',smtype,cpu,iter,rate));axis tight; hold on

        err=norm(S0(:)-x,inf);%solution max error, decreM_e by 4 times
        fprintf('%d&\t %1.1e & \t %1.1e &\t %d&\t  %1.2f\t%1.3f \\\\ \n',N,res,err,iter,cpu,rate);
    end
end
set(gcf, 'Position', get(0, 'Screensize').*[1 1 1/2 1/2]);
legend('-DynamicLegend','Location', 'NorthEast');
title(sprintf('7-point 3D Poisson: GMG with $N$=%d and %d-levels',N,fineL-LEVEL0+1));
xlabel(sprintf('%s(%d+%d)-cycles',cycle,mu1,mu2));
ylabel('Relative residual norms')
figname=sprintf('../Manuscript/OptimalSmoother/MGplot3D_%s',cycle);
%print(figname,'-depsc')

function [x]=mg_iter_2d(mg,x0,b,level,pre,post)
A=mg(level).A;
global LEVEL0 omega
if(level==LEVEL0) %Coarest level
    x=A\b;
else
    x = mg_smooth(mg,A,x0,b,pre,level);
    r = b - A*x;
    rc=(mg(level).P)'*r/8;
    % coarse grid correction
    cc = mg_iter_2d(mg,zeros(size(rc)),rc,level-1,pre,post);
    if(omega==2)
        cc = mg_iter_2d(mg,cc,rc,level-1,pre,post); %add this for W cycle
    end
    x = x + mg(level).P*cc; %
    x = mg_smooth(mg,A,x,b,post,level);
end
end
function [x]=mg_smooth(mg,A,x,b,nv,level)
for k = 1:nv
    x=x + mg(level).w*mg(level).Mfun(b - A*x);
end
end




