clear all; 
%this is to generate figures for the rate models in joglekar et al neuron
%2018
load subgraphData.mat %loading fln and hierarchy data. 

netwParams=struct('nNodes',nNodes,'alpha',0.68);

netwParams.hier=hierVals/max(hierVals);%hierarchy normalized. 
betaE = .066 ; betaI = .351 ; %Hz/pA

muEE = 33.7 ; muIE = 25.3; omegaEE = 24.3 ; omegaEI = 19.7 ;  netwParams.alpha=.68;omegaIE = 12.2 ; omegaII = 12.5 ; 
%omegaEI = 25.2 ; muEE = 51.5;
cv(1,1:3) = [0.4660, 0.6740, 0.1880]; cv(2,1:3) = [0.4940,0.1840,0.5560];%color scheme.

%various cases below plot figures in neuron paper. 

caseval = 0; %normal fln values (not removing fb, symmetrizing fln matrix etc.) %netwParams.alpha=0
%caseval = 1; flnMat = tril(flnMat);  %removing fb worsens propagation 
%caseval = -1; netwParams.alpha=0;  %no hierarchy 
%caseval = 2; flnMat = (flnMat+flnMat')/2; netwParams.alpha=.6;%symmetrizing fln matrix -- using arithmetic mean. 
%caseval = 3; flnMat = sqrt(flnMat.*flnMat'); %symmetrizing fln -- geometric mean. 
%caseval = 4; %change threshold and connection density .. measure propagation ratio
%caseval = 5; nofb=0;%plot how muEE changes and activity in 24c blows up or too low 
%flnMat = tril(flnMat); %to test case5 w/out fb. % other tests: nofb=1; flnMat = tril(flnMat);%netwParams.alpha=0;
%caseval = 6; %histogram showing few strong connections vs all flns. 
%caseval = 7; %for figure about balanced amp. in local circuit  
%caseval = 8; %for figure about either blowup or badprop in 24c vs V1. along with input to V1. 
%caseval = 9; %jorge's mejias sci advances figure 
%caseval = 10; %jorge's mejias sci advances figure
%caseval = 11; %compare max rate control case / no fb / symmetrized fln with arithmetic / geometric mean. 
%caseval = 12; %caret figs. for brain. 
%caseval = 13; %jorge mejias science advances -- fig oscillations. 
%caseval = 14 ; %contour plot large scale system  
%caseval = -2; flnMat = (flnMat+flnMat')/2; netwParams.alpha=0; 
%caseval = -3; flnMat = (flnMat+flnMat')/2; netwParams.alpha=0; 
%caseval = 15; %burkhalter result implement in supplementary. vary muIE

%%%%%%%%%%%% pulse to V1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%basic network setup below

localParams.bgInh=zeros(netwParams.nNodes,1);%background inhibitory
localParams.tauE=2e-2; localParams.tauI=1e-2; % time constants
tParams.nSteps=5e4;tParams.dt=2e-4;
tSteps=(1:tParams.nSteps)*tParams.dt;
toplotstart=round(1.75/tParams.dt);toplotend=round(5/tParams.dt);
%input to v1
inpArea='V1'; inpType='whiteNoise'; areaInput.idx=areaMap(inpArea);nNodes=29;
currParams.desiredSs=[10*ones(nNodes,1); 35*ones(nNodes,1)]; 
localParams.noiseStd=0;  % No local noise here for pulse input to V1

lw = 1; fsize=7;fsizelab=7;msize=7; poster = 0;

%poster=1;
if poster==1
    fsize=10;fsizelab=10;msize=10;
end

if caseval < 4  
    for count=1:2
        if count==1
            omegaEI = 19.7 ; muEE = 33.7 ; %weak GBA
        else
            omegaEI = 25.2 ; muEE = 51.5;%strong GBA
        end
        
        localParams.ee=betaE*omegaEE*(1+netwParams.alpha*netwParams.hier);
        localParams.ie=betaI*omegaIE*(1+netwParams.alpha*netwParams.hier);
        localParams.ei=-betaE*omegaEI; 
        localParams.ii=-betaI*omegaII;         
             
        ldParams.ee=betaE*muEE*(1+netwParams.alpha*netwParams.hier);
        ldParams.ie=betaI*muIE*(1+netwParams.alpha*netwParams.hier);     
        
        if caseval == -3
          localParams.ee=betaE*omegaEE*(1+.68*netwParams.hier); %note eta = .68 as in chaudhuri 2015 neuron
          localParams.ie=betaI*omegaIE*(1+.68*netwParams.hier);
        end    
       
        ldConns.ee= bsxfun(@times,flnMat,ldParams.ee);
        ldConns.ie= bsxfun(@times,flnMat,ldParams.ie);

        
%         %if caseval == 15
%             burkmat = zeros(nNodes, nNodes);
%             mf = .1;  %vary mf. 
%             %burkhalter like scaling of long range E to I. 
%             for r = 1:nNodes 
%                 for s = 1:nNodes
%                     burkmat(r,s) = max(0,netwParams.hier(r,1) - netwParams.hier(s,1));
% %                    burkmat(r,s) = netwParams.hier(r,1) - netwParams.hier(s,1);
%                 end ;
%             end;
%             burkmat = 1 + mf*burkmat;
%             ldConns.ie = bsxfun(@times,ldConns.ie,burkmat);
%         %end
        
        
        nNodes=size(ldConns.ee,1);%29 nodes. 
        %build local and long range connections. 
        wEe=diag(-1+localParams.ee)+ldConns.ee;
        wEi=localParams.ei*eye(nNodes);%these ws would be 29x29 identity matrices. 
        wIe=diag(localParams.ie)+ldConns.ie;
        wIi=(-1+localParams.ii)*eye(nNodes);%compare this to eqn 4 - where does -1 come from??

        A = [wEe/localParams.tauE wEi/localParams.tauE;...
            wIe/localParams.tauI wIi/localParams.tauI];

        currParams.A=A; 

        B=currParams.A;%B=params.A;
                B(1:nNodes,:)=B(1:nNodes,:)*localParams.tauE;
                B((nNodes+1):end,:)=B((nNodes+1):end,:)*localParams.tauI;
                currs=-B*currParams.desiredSs; %currs=-B*params.desiredSs;

        localParams.bgExc=currs(1:nNodes); localParams.bgInh=currs((nNodes+1):end);
        bgCurr=[localParams.bgExc; localParams.bgInh];%bg = background

        combMat=A;
        combMat(1:nNodes,:)=combMat(1:nNodes,:)*localParams.tauE;
        combMat((nNodes+1):end,:)=combMat((nNodes+1):end,:)*localParams.tauI;
        sstate=-combMat\bgCurr;%which is -bgCurr/combMat %this is steady state. 

        netwParams.initConds.exc=sstate(1:nNodes); netwParams.initConds.inh=sstate((nNodes+1):end);

        if (caseval==0 || caseval == -1)
            if (count == 1)
                %inputs set in various cases to set firing rate to 100 Hz. 
            inpParams=struct('tStart',2,'tEnd',2.25,'ampl',22.05*1.9);
            else
            inpParams=struct('tStart',2,'tEnd',2.25,'ampl',11.54*1.9);    
            end
        end
        if caseval==1
            if (count == 1)
            inpParams=struct('tStart',2,'tEnd',2.25,'ampl',22.68*1.9);
            else
            inpParams=struct('tStart',2,'tEnd',2.25,'ampl',37.8*1.9);    
            end
        end
        if caseval==2
            if (count == 1)
            inpParams=struct('tStart',2,'tEnd',2.25,'ampl',22.07*1.9);
            else
            inpParams=struct('tStart',2,'tEnd',2.25,'ampl',11.18*1.9);    
            end
        end
        if caseval==3
            if (count == 1)
            inpParams=struct('tStart',2,'tEnd',2.25,'ampl',22.07*1.9);
            else
            inpParams=struct('tStart',2,'tEnd',2.25,'ampl',13.63*1.9);    
            end
        end
        
        if caseval== -2 || caseval== -3
            if (count == 1)
            inpParams=struct('tStart',2,'tEnd',2.25,'ampl',22.07*1.9);
            else
            inpParams=struct('tStart',2,'tEnd',2.25,'ampl',13.63*1.9);    
            end
        end
        
        areaInput.inp=zeros(tParams.nSteps,1);
        areaInput.inp(round(inpParams.tStart/tParams.dt):round(inpParams.tEnd/tParams.dt))=inpParams.ampl;


        if (count == 1)
            popRatesbad=runNetworkAreaInputNEW(tParams,localParams,netwParams,ldConns,areaInput,'threshLinear');
            maxpeakbad=zeros(nNodes,1);
            for r=1:nNodes
            maxpeakbad(r,1)=max(popRatesbad.exc(r,(toplotstart:toplotend))-popRatesbad.exc(r,toplotstart));
            end
        else
            popRatesgood=runNetworkAreaInputNEW(tParams,localParams,netwParams,ldConns,areaInput,'threshLinear');
            maxpeakgood=zeros(nNodes,1);
            for r=1:nNodes
            maxpeakgood(r,1)=max(popRatesgood.exc(r,(toplotstart:toplotend))-popRatesgood.exc(r,toplotstart));
            end
        end    

        if (caseval==0 || caseval==1 || caseval==2 || caseval == -1 || caseval == -2 || caseval == -3)
            if (count == 1)
            s=1000;%weak gba
            else
            s=100;%strong gba 
            end
        end
        if caseval==3
            if (count == 1)
            s=10000;%weak gba
            else
            s=100;%strong gba 
            end
        end
        
        lw=1;
        x1 = -98.25; x2 = -96; cv(1,1:3) = [0.4660, 0.6740, 0.1880]; cv(2,1:3) = [0.4940,0.1840,0.5560];

        hFig = figure();
        if count == 1
          set(hFig,'units', 'centimeters', 'Position', [0 0 5.7 6],'PaperUnits','centimeters','PaperPosition',[.1 -.4 5.7 6],'PaperSize',[6. 5.8])          
        else
          set(hFig,'units', 'centimeters', 'Position', [0 0 5.7 6],'PaperUnits','centimeters','PaperPosition',[.1 -.4 5.7 6],'PaperSize',[6. 5.8])  
        end

        if count ==1 
            maxpeak = maxpeakbad;
        else
            maxpeak = maxpeakgood;
        end
        
        if count == 1
            subplot(9,1,1);plot(tSteps(toplotstart:toplotend)-100,popRatesbad.exc(1,(toplotstart:toplotend))-popRatesbad.exc(1,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(count,:));set(gca,'xcolor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(1,1))/s,'Xtick',[],'XLim',[x1 x2],'YLim',[0 140]);%ylabel('V1','FontSize', 18)
            hold on;plot(tSteps(round(inpParams.tStart/tParams.dt):round(inpParams.tEnd/tParams.dt))-100,135*ones(size(tSteps(round(inpParams.tStart/tParams.dt):round(inpParams.tEnd/tParams.dt)))),'k');

            if poster == 0
               annotation('textbox', [0.42,0.9,0.1,0.1],'String', 'Weak GBA','FontSize',fsizelab,'LineStyle','None');
               annotation('textbox', [0.23,0.88,0.1,0.1],'String', '250 ms','FontSize',fsizelab,'LineStyle','None');
            end
            
            subplot(9,1,2);plot(tSteps(toplotstart:toplotend)-100,popRatesbad.exc(3,(toplotstart:toplotend))-popRatesbad.exc(3,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(count,:));set(gca,'xcolor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(3,1))/s,'Xtick',[],'XLim',[x1 x2],'YLim',[0 1.2*maxpeak(3,1)]);%ylabel('V4')
            subplot(9,1,3);plot(tSteps(toplotstart:toplotend)-100,popRatesbad.exc(6,(toplotstart:toplotend))-popRatesbad.exc(6,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(count,:));set(gca,'XColor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(6,1))/s, 'Yticklabel',round(s*maxpeak(6,1))/s,'Xtick',[],'XLim',[x1 x2],'YLim',[0 1.2*maxpeak(6,1)]);
            subplot(9,1,4);plot(tSteps(toplotstart:toplotend)-100,popRatesbad.exc(8,(toplotstart:toplotend))-popRatesbad.exc(8,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(count,:));set(gca,'XColor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(8,1))/s,'Xtick',[],'XLim',[x1 x2],'YLim',[0 1.2*maxpeak(8,1)]);%ylabel('8l')
            subplot(9,1,5);plot(tSteps(toplotstart:toplotend)-100,popRatesbad.exc(9,(toplotstart:toplotend))-popRatesbad.exc(9,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(count,:));set(gca,'XColor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(9,1))/s,'Xtick',[],'XLim',[x1 x2],'YLim',[0 1.2*maxpeak(9,1)]);%ylabel('TEO')
            subplot(9,1,6);plot(tSteps(toplotstart:toplotend)-100,popRatesbad.exc(13,(toplotstart:toplotend))-popRatesbad.exc(13,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(count,:));set(gca,'XColor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(13,1))/s,'Xtick',[],'XLim',[x1 x2],'YLim',[0 1.2*maxpeak(13,1)]);%ylabel('7A')
            subplot(9,1,7);plot(tSteps(toplotstart:toplotend)-100,popRatesbad.exc(17,(toplotstart:toplotend))-popRatesbad.exc(17,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(count,:));set(gca,'XColor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(17,1))/s,'Yticklabel',round(s*maxpeak(17,1))/s,'Xtick',[],'XLim',[x1 x2],'YLim',[0 1.2*maxpeak(17,1)]);%ylabel('9/46D')
            subplot(9,1,8);plot(tSteps(toplotstart:toplotend)-100,popRatesbad.exc(19,(toplotstart:toplotend))-popRatesbad.exc(19,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(count,:));set(gca,'XColor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(19,1))/s,'Xtick',[],'XLim',[x1 x2],'YLim',[0 1.2*maxpeak(19,1)]);%ylabel('TEpd')
            subplot(9,1,9);plot(tSteps(toplotstart:toplotend)-100,popRatesbad.exc(29,(toplotstart:toplotend))-popRatesbad.exc(29,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(count,:));set(gca,'XColor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(29,1))/s,'Yticklabel',round(s*maxpeak(29,1))/s,'XLim',[x1 x2],'YLim',[0 1.2*maxpeak(29,1)]);
        else
            
            subplot(9,1,1);plot(tSteps(toplotstart:toplotend)-100,popRatesbad.exc(1,(toplotstart:toplotend))-popRatesbad.exc(1,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(1,:));set(gca,'XColor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(1,1))/s,'Xtick',[],'XLim',[x1 x2],'YLim',[0 140]);%ylabel('V1','FontSize', 18)
              if poster == 0
                 annotation('textbox', [0.42,0.9,0.1,0.1],'String', 'Strong GBA','FontSize',fsizelab,'LineStyle','None');
                 annotation('textbox', [0.21,0.88,0.1,0.1],'String', '250 ms','FontSize',fsizelab,'LineStyle','None');
              end
              hold on;subplot(9,1,1);plot(tSteps(toplotstart:toplotend)-100,popRatesgood.exc(1,(toplotstart:toplotend))-popRatesgood.exc(1,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(count,:));set(gca,'XColor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(1,1))/s,'Xtick',[],'XLim',[x1 x2],'YLim',[0 140]);%ylabel('V1','FontSize', 18)
              hold on;plot(tSteps(round(inpParams.tStart/tParams.dt):round(inpParams.tEnd/tParams.dt))-100,135*ones(size(tSteps(round(inpParams.tStart/tParams.dt):round(inpParams.tEnd/tParams.dt)))),'k');
              
            subplot(9,1,2);plot(tSteps(toplotstart:toplotend)-100,popRatesbad.exc(3,(toplotstart:toplotend))-popRatesbad.exc(3,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(1,:));set(gca,'XColor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(3,1))/s,'Xtick',[],'XLim',[x1 x2],'YLim',[0 1.2*maxpeak(3,1)]);%ylabel('V4')
              hold on;subplot(9,1,2);plot(tSteps(toplotstart:toplotend)-100,popRatesgood.exc(3,(toplotstart:toplotend))-popRatesgood.exc(3,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(count,:));set(gca,'XColor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(3,1))/s,'Xtick',[],'XLim',[x1 x2],'YLim',[0 1.2*maxpeak(3,1)]);%ylabel('V4')
            subplot(9,1,3);plot(tSteps(toplotstart:toplotend)-100,popRatesbad.exc(6,(toplotstart:toplotend))-popRatesbad.exc(6,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(1,:));set(gca,'XColor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(6,1))/s, 'Yticklabel',round(s*maxpeak(6,1))/s,'Xtick',[],'XLim',[x1 x2],'YLim',[0 1.2*maxpeak(6,1)]);
              hold on;subplot(9,1,3);plot(tSteps(toplotstart:toplotend)-100,popRatesgood.exc(6,(toplotstart:toplotend))-popRatesgood.exc(6,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(count,:));set(gca,'XColor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(6,1))/s, 'Yticklabel',round(s*maxpeak(6,1))/s,'Xtick',[],'XLim',[x1 x2],'YLim',[0 1.2*maxpeak(6,1)]);
            subplot(9,1,4);plot(tSteps(toplotstart:toplotend)-100,popRatesbad.exc(8,(toplotstart:toplotend))-popRatesbad.exc(8,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(1,:));set(gca,'XColor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(8,1))/s,'Xtick',[],'XLim',[x1 x2],'YLim',[0 1.2*maxpeak(8,1)]);%ylabel('8l')
              hold on;subplot(9,1,4);plot(tSteps(toplotstart:toplotend)-100,popRatesgood.exc(8,(toplotstart:toplotend))-popRatesgood.exc(8,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(count,:));set(gca,'XColor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(8,1))/s,'Xtick',[],'XLim',[x1 x2],'YLim',[0 1.2*maxpeak(8,1)]);%ylabel('8l')
            subplot(9,1,5);plot(tSteps(toplotstart:toplotend)-100,popRatesbad.exc(9,(toplotstart:toplotend))-popRatesbad.exc(9,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(1,:));set(gca,'XColor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(9,1))/s,'Xtick',[],'XLim',[x1 x2],'YLim',[0 1.2*maxpeak(9,1)]);%ylabel('TEO')
              hold on;subplot(9,1,5);plot(tSteps(toplotstart:toplotend)-100,popRatesgood.exc(9,(toplotstart:toplotend))-popRatesgood.exc(9,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(count,:));set(gca,'XColor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(9,1))/s,'Xtick',[],'XLim',[x1 x2],'YLim',[0 1.2*maxpeak(9,1)]);%ylabel('TEO')
            subplot(9,1,6);plot(tSteps(toplotstart:toplotend)-100,popRatesbad.exc(13,(toplotstart:toplotend))-popRatesbad.exc(13,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(1,:));set(gca,'XColor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(13,1))/s,'Xtick',[],'XLim',[x1 x2],'YLim',[0 1.2*maxpeak(13,1)]);%ylabel('7A')
              hold on;subplot(9,1,6);plot(tSteps(toplotstart:toplotend)-100,popRatesgood.exc(13,(toplotstart:toplotend))-popRatesgood.exc(13,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(count,:));set(gca,'XColor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(13,1))/s,'Xtick',[],'XLim',[x1 x2],'YLim',[0 1.2*maxpeak(13,1)]);%ylabel('7A')
            subplot(9,1,7);plot(tSteps(toplotstart:toplotend)-100,popRatesbad.exc(17,(toplotstart:toplotend))-popRatesbad.exc(17,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(1,:));set(gca,'XColor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(17,1))/s,'Yticklabel',round(s*maxpeak(17,1))/s,'Xtick',[],'XLim',[x1 x2],'YLim',[0 1.2*maxpeak(17,1)]);%ylabel('9/46D')
              hold on;subplot(9,1,7);plot(tSteps(toplotstart:toplotend)-100,popRatesgood.exc(17,(toplotstart:toplotend))-popRatesgood.exc(17,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(count,:));set(gca,'XColor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(17,1))/s,'Yticklabel',round(s*maxpeak(17,1))/s,'Xtick',[],'XLim',[x1 x2],'YLim',[0 1.2*maxpeak(17,1)]);%ylabel('9/46D')
            subplot(9,1,8);plot(tSteps(toplotstart:toplotend)-100,popRatesbad.exc(19,(toplotstart:toplotend))-popRatesbad.exc(19,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(1,:));set(gca,'XColor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(19,1))/s,'Xtick',[],'XLim',[x1 x2],'YLim',[0 1.2*maxpeak(19,1)]);%ylabel('TEpd')
              hold on;subplot(9,1,8);plot(tSteps(toplotstart:toplotend)-100,popRatesgood.exc(19,(toplotstart:toplotend))-popRatesgood.exc(19,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(count,:));set(gca,'XColor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(19,1))/s,'Xtick',[],'XLim',[x1 x2],'YLim',[0 1.2*maxpeak(19,1)]);%ylabel('TEpd')
            subplot(9,1,9);plot(tSteps(toplotstart:toplotend)-100,popRatesbad.exc(29,(toplotstart:toplotend))-popRatesbad.exc(29,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(1,:));set(gca,'XColor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(29,1))/s,'Yticklabel',round(s*maxpeak(29,1))/s,'XLim',[x1 x2],'YLim',[0 1.2*maxpeak(29,1)]);            
              hold on;subplot(9,1,9);plot(tSteps(toplotstart:toplotend)-100,popRatesgood.exc(29,(toplotstart:toplotend))-popRatesgood.exc(29,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(count,:));set(gca,'XColor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(29,1))/s,'Yticklabel',round(s*maxpeak(29,1))/s,'XLim',[x1 x2],'YLim',[0 1.2*maxpeak(29,1)]);            
        end
        
        h = xlabel('Time (s)','FontSize', fsize);set(h, 'Units', 'Normalized');pos = get(h, 'Position');set(h, 'Position', pos + [.01, 0.4, -0.9]);%
        set(gca,'Xtick',-98.:2:-96);set(gca,'XTickLabel',['0';'2']);%timept = [' ';'0';' ';' ';' ';' ';' ';' ';' ';'2'];set(gca,'XTickLabel',timept);
        set(gca,'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(29,1))/s);%ylabel('18c')
        
        if count == 2
            xv = .89;%0.87
            annotation('textbox', [xv,0.83,0.1,0.1],'String', 'V1','FontSize',fsizelab,'LineStyle','None');
            annotation('textbox', [xv,0.735,0.1,0.1],'String', 'V4','FontSize',fsizelab,'LineStyle','None');
            annotation('textbox', [xv,0.645,0.1,0.1],'String', '8m','FontSize',fsizelab,'LineStyle','None');
            annotation('textbox', [xv,0.55,0.1,0.1],'String', '8l','FontSize',fsizelab,'LineStyle','None');
            annotation('textbox', [xv,0.445,0.1,0.1],'String', 'TEO','FontSize',fsizelab,'LineStyle','None');
            annotation('textbox', [xv,0.353,0.1,0.1],'String', '7A','FontSize',fsizelab,'LineStyle','None');
            annotation('textbox', [xv,0.265,0.1,0.1],'String', '9/46d','FontSize',fsizelab,'LineStyle','None');
            annotation('textbox', [xv,0.173,0.1,0.1],'String', 'TEpd','FontSize',fsizelab,'LineStyle','None');
            annotation('textbox', [xv,0.08,0.1,0.1],'String', '24c','FontSize',fsizelab,'LineStyle','None');
        end
        
        if (caseval==0 || caseval==1 || caseval==2 || caseval==-1)
            if (count==1)
             h = ylabel('Change in firing rate (Hz)','FontSize', fsize);set(h, 'Units', 'Normalized');pos = get(h, 'Position');set(h, 'Position', pos + [-.01, 6.2, -0.9]);%bad
            
            end
        end
        if caseval==3
            if (count==1)
             h = ylabel('Change in firing rate (Hz)','FontSize', fsize);set(h, 'Units', 'Normalized');pos = get(h, 'Position');set(h, 'Position', pos + [-.01, 6.3, -0.9]);%bad
            else
             h = ylabel('Change in firing rate (Hz)','FontSize', fsize);set(h, 'Units', 'Normalized');pos = get(h, 'Position');set(h, 'Position', pos + [-.04, 6.3, -0.9]);%good
            end
        end
      
        %%%%%%%%%%%%%%%%%for peak firing rate figures across various areas %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        nodelen=1:29;nodelen=nodelen';
        if (count==1)
            hFig2 = figure();
            set(hFig2,'units', 'centimeters', 'Position', [0 0 8 6],'PaperUnits','centimeters','PaperPosition',[.1 .1 8 6],'PaperSize',[8.2 6.2]) 
       %     set(hFig2,'units', 'centimeters', 'Position', [0 0 8 8],'PaperUnits','centimeters','PaperPosition',[.1 .1 8 8],'PaperSize',[8.2 8.2]) 
            semilogy(nodelen,100*(maxpeakbad)/maxpeakbad(1,1),'.-','LineWidth',lw,'MarkerSize',fsizelab, 'Color',[0.4660, 0.6740, 0.1880]); 
            if caseval==1
            hold on; semilogy(6:2:8,100*(maxpeakbad(6:2:8,1))/maxpeakbad(1,1),':','LineWidth',lw,'MarkerSize',fsizelab,'Color',[0.4660, 0.6740, 0.1880]);
            hold on; semilogy(9:2:11,100*(maxpeakbad(9:2:11,1))/maxpeakbad(1,1),':','LineWidth',lw,'MarkerSize',fsizelab,'Color',[0.4660, 0.6740, 0.1880]);
            set(gca,'Ytick',logspace(-6,2,3),'YTickLabel',['10^{-6}';'10^{-2}';'10^{2} ']);
            else
            set(gca,'Ytick',logspace(-4,2,4),'YTickLabel',['10^{-4}';'10^{-2}';'10^{0} ';'10^{2} ']);
            end
        else
            hFig2 = figure();
            set(hFig2,'units', 'centimeters', 'Position', [0 0 7.5 3.8],'PaperUnits','centimeters','PaperPosition',[.05 .01 7.5 3.8],'PaperSize',[7. 3.8]) ;
            semilogy(nodelen-1,100*(maxpeakbad)/maxpeakbad(1,1),'.-','LineWidth',lw,'MarkerSize',fsizelab,'Color',[0.4660, 0.6740, 0.1880]);hold on;
            semilogy(nodelen-1,100*(maxpeakgood)/maxpeakgood(1,1),'.-','LineWidth',lw,'MarkerSize',fsizelab,'Color',[0.4940,0.1840,0.5560]);        
            if caseval==1   
            hold on; semilogy((6:2:8)-1,100*(maxpeakbad(6:2:8,1))/maxpeakbad(1,1),':','LineWidth',lw,'MarkerSize',fsizelab,'Color',[0.4660, 0.6740, 0.1880]);
            hold on; semilogy((9:2:11)-1,100*(maxpeakbad(9:2:11,1))/maxpeakbad(1,1),':','LineWidth',lw,'MarkerSize',fsizelab,'Color',[0.4660, 0.6740, 0.1880]);
            hold on; semilogy((6:2:8)-1,100*(maxpeakgood(6:2:8,1))/maxpeakgood(1,1),':','LineWidth',lw,'MarkerSize',fsizelab,'Color',[0.4940,0.1840,0.5560]);
            hold on; semilogy((9:2:11)-1,100*(maxpeakgood(9:2:11,1))/maxpeakgood(1,1),':','LineWidth',lw,'MarkerSize',fsizelab,'Color',[0.4940,0.1840,0.5560]);
            set(gca,'Ytick',logspace(-6,2,3),'YTickLabel',['10^{-6}';'10^{-2}';'10^{2} '],'ylim',[1e-8 1e2]);
            else
            set(gca,'Ytick',logspace(-4,2,4),'YTickLabel',['10^{-4}';'10^{-2}';'10^{0} ';'10^{2} ']);
            legend({'weak GBA','strong GBA'},'Position',[0.7 0.87 0.1 0.1],'Box','off','FontSize',fsize);  
            end
        end
        xlabel('Areas','FontSize',fsize);
        set(gca,'FontSize',fsizelab);
        set(gca,'Xtick',0:1:28,'box','off');
        areapt = ['  V1 ';'  V2 ';'  V4 ';'  DP ';'  MT ';'  8m ';'  5  ';'  8l ';' TEO ';'  2  ';'  F1 ';' STPc';'  7A ';' 46d ';'  10 ';'9/46v';'9/46d';'  F5 ';' TEpd';' PBr ';' 7m  ';' 7B  ';'  F2 ';' STPi';' PROm';' F7  ';' 8B  ';' STPr';' 24c ']; %timept = ['';'0';'';'';'1';'';'';'2'];
        set(gca,'XTickLabel',areapt);
        ax = gca;ax.XTickLabelRotation = 90; 
        h = ylabel('Maximum firing rate (Hz)','FontSize',fsize);%set(h, 'Units', 'Normalized');
        pos = get(h, 'Position');set(h, 'Position', pos + [-.02, -.06, -0.9]);%
        
        if count == 2
            if caseval == 0
                maxpeakbad_control = maxpeakbad; maxpeakgood_control = maxpeakgood;
                %save('maxpeakbad_control.mat','maxpeakbad_control'); save('maxpeakgood_control.mat','maxpeakgood_control');
            end
            if caseval == 1
                maxpeakbad_nofb= maxpeakbad; maxpeakgood_nofb = maxpeakgood;
                %save('maxpeakbad_nofb.mat','maxpeakbad_nofb'); save('maxpeakgood_nofb.mat','maxpeakgood_nofb');
            end
            if caseval == 2
                maxpeakbad_arimean = maxpeakbad; maxpeakgood_arimean = maxpeakgood;
                %save('maxpeakbad_arimean.mat','maxpeakbad_arimean'); save('maxpeakgood_arimean.mat','maxpeakgood_arimean');
            end
            if caseval == 3
                maxpeakbad_geomean = maxpeakbad; maxpeakgood_geomean = maxpeakgood;
                %save('maxpeakbad_geomean.mat','maxpeakbad_geomean'); save('maxpeakgood_geomean.mat','maxpeakgood_geomean');
            end
            if caseval == -1
                maxpeakbad_nohier = maxpeakbad; maxpeakgood_nohier = maxpeakgood;
                %save('maxpeakbad_nohier.mat','maxpeakbad_nohier'); save('maxpeakgood_nohier.mat','maxpeakgood_nohier');
            end
   
            if caseval == -2 %new after revision. this and below case -3. 
                maxpeakbad_arimeannohier = maxpeakbad; maxpeakgood_arimeannohier = maxpeakgood;
                %save('maxpeakbad_arimeannohier.mat','maxpeakbad_arimeannohier'); save('maxpeakgood_arimeannohier.mat','maxpeakgood_arimeannohier');
            end
            if caseval == -3
                maxpeakbad_arimeanlochier = maxpeakbad; maxpeakgood_arimeanlochier = maxpeakgood;
                %save('maxpeakbad_arimeanlochier.mat','maxpeakbad_arimeanlochier'); save('maxpeakgood_arimeanlochier.mat','maxpeakgood_arimeanlochier');
            end
         
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if (caseval==0)
            if count == 1
                % saveas(hFig, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/main fig 2E', 'pdf')                

            else
                 %saveas(hFig, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/main fig 2G', 'pdf')
            end
        end
        if (caseval==1)
            if count == 1
             %   saveas(hFig2, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/supp fig 1 nofbbadmaxrate', 'pdf');
            else
              %   saveas(hFig2, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/supp fig 1 nofbgoodmaxrate', 'pdf')
            end
        end
        if (caseval==2)
            if count == 1
              %   saveas(hFig2, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/supp fig 2 symflnarmeanbadmaxrate', 'pdf')
            else
             %   saveas(hFig2, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/supp fig 2 symflnarmeangoodmaxrate', 'pdf')
            end
        end
        if (caseval==3)
            if count == 1
%                 saveas(hFig2, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/supp fig 3 symflngeomeanbadmaxrate', 'pdf')
            else
%                 saveas(hFig2, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/supp fig 3 symflngeomeangoodmaxrate', 'pdf')
            end
        end
  
    end
end

%%%%%%%%%%for figure with threshold and removing weak connections. %%%%%%
if caseval == 4

 binsize=15;%no of divisions for conn strength calculation. 
 thresvec = logspace(-5,-2,binsize)'; 
 frac=zeros(binsize,1);peakratio = zeros(binsize,2);
 inpParams=struct('tStart',2,'tEnd',2.25,'ampl',21.5*1.951);
 areaInput.inp=zeros(tParams.nSteps,1);
 areaInput.inp(round(inpParams.tStart/tParams.dt):round(inpParams.tEnd/tParams.dt))=inpParams.ampl;
for u=1:length(thresvec) %run loop... 
  thres = thresvec(u);
  clear flnMatloop; flnMatloop = flnMat;
 
frac(u,1) = numel(flnMat( flnMat(:)>thresvec(u)  ))/8.12;%--area to area only. on x axis.% figure; plot(thresvec,frac)
 
for flnrow=1:length(flnMat)
  for flncol=1:length(flnMat)
 if(flnMat(flnrow,flncol)<thres)
    flnMatloop(flnrow,flncol) = 0;
 end
  end
end

 for uu = 1:2 %orig vs reg 1.
         
    if (uu==1)
     clear A popRates; muEE = 33.7 ; omegaEI = 19.7 ; 
    else
     clear A popRates; muEE = 51.5; omegaEI = 25.2 ; 
    end

    localParams.bgInh=zeros(netwParams.nNodes,1);%background inhibitory        
    localParams.ee=betaE*omegaEE*(1+netwParams.alpha*netwParams.hier);
    localParams.ie=betaI*omegaIE*(1+netwParams.alpha*netwParams.hier);
    localParams.ei=-betaE*omegaEI; 
    localParams.ii=-betaI*omegaII;         
    ldParams.ee=betaE*muEE*(1+netwParams.alpha*netwParams.hier);
    ldParams.ie=betaI*muIE*(1+netwParams.alpha*netwParams.hier);
    ldConns.ee= bsxfun(@times,flnMatloop,ldParams.ee);
    ldConns.ie= bsxfun(@times,flnMatloop,ldParams.ie);

    wEe=diag(-1+localParams.ee)+ldConns.ee;
    wEi=localParams.ei*eye(nNodes);%these ws would be 29x29 identity matrices. 
    wIe=diag(localParams.ie)+ldConns.ie;
    wIi=(-1+localParams.ii)*eye(nNodes);
    A = [wEe/localParams.tauE wEi/localParams.tauE;...
        wIe/localParams.tauI wIi/localParams.tauI];

    currParams.A=A; 
    B=currParams.A;
    B(1:nNodes,:)=B(1:nNodes,:)*localParams.tauE;
    B((nNodes+1):end,:)=B((nNodes+1):end,:)*localParams.tauI;
    currs=-B*currParams.desiredSs; 

    localParams.bgExc=currs(1:nNodes); localParams.bgInh=currs((nNodes+1):end);
    bgCurr=[localParams.bgExc; localParams.bgInh];%bg = background
    combMat=A;
    combMat(1:nNodes,:)=combMat(1:nNodes,:)*localParams.tauE;
    combMat((nNodes+1):end,:)=combMat((nNodes+1):end,:)*localParams.tauI;
    sstate=-combMat\bgCurr;%which is -bgCurr/combMat
    netwParams.initConds.exc=sstate(1:nNodes); netwParams.initConds.inh=sstate((nNodes+1):end);
 
    popRates=runNetworkAreaInputNEW(tParams,localParams,netwParams,ldConns,areaInput,'threshLinear');

    maxpeak=zeros(nNodes,1);
    for r=1:nNodes
    maxpeak(r,1)=max(popRates.exc(r,(toplotstart:toplotend))-popRates.exc(r,toplotstart));
    end
    peakratio(u,uu) = maxpeak(29,1)/maxpeak(1,1);
    
    
  if u == 12
     %plot -  threshold, connection density 
        lw=1;
        x1 = -98.25; x2 = -96; cv(1,1:3) = [0.4660, 0.6740, 0.1880]; cv(2,1:3) = [0.4940,0.1840,0.5560];

        if uu == 1
            count = uu;
            popRatesbad = popRates;
            hFig = figure();
            s=1000;
            set(hFig,'units', 'centimeters', 'Position', [0 0 5.7 8],'PaperUnits','centimeters','PaperPosition',[.1 -.4 5.7 8],'PaperSize',[6. 7.6]) 
            subplot(9,1,1);plot(tSteps(toplotstart:toplotend)-100,popRatesbad.exc(1,(toplotstart:toplotend))-popRatesbad.exc(1,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(count,:));set(gca,'xcolor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(1,1))/s,'Xtick',[],'XLim',[x1 x2],'YLim',[0 1.2*maxpeak(1,1)]);%ylabel('V1','FontSize', 18)
            hold on;plot(tSteps(round(inpParams.tStart/tParams.dt):round(inpParams.tEnd/tParams.dt))-100,123*ones(size(tSteps(round(inpParams.tStart/tParams.dt):round(inpParams.tEnd/tParams.dt)))),'k');
               annotation('textbox', [0.45,0.9,0.1,0.1],'String', 'weak GBA','FontSize',fsizelab,'LineStyle','None');
               annotation('textbox', [0.23,0.865,0.1,0.1],'String', '250 ms','FontSize',fsizelab,'LineStyle','None');
            subplot(9,1,2);plot(tSteps(toplotstart:toplotend)-100,popRatesbad.exc(3,(toplotstart:toplotend))-popRatesbad.exc(3,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(count,:));set(gca,'xcolor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(3,1))/s,'Xtick',[],'XLim',[x1 x2],'YLim',[0 1.2*maxpeak(3,1)]);%ylabel('V4')
            subplot(9,1,3);plot(tSteps(toplotstart:toplotend)-100,popRatesbad.exc(6,(toplotstart:toplotend))-popRatesbad.exc(6,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(count,:));set(gca,'XColor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(6,1))/s, 'Yticklabel',round(s*maxpeak(6,1))/s,'Xtick',[],'XLim',[x1 x2],'YLim',[0 1.2*maxpeak(6,1)]);
            subplot(9,1,4);plot(tSteps(toplotstart:toplotend)-100,popRatesbad.exc(8,(toplotstart:toplotend))-popRatesbad.exc(8,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(count,:));set(gca,'XColor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(8,1))/s,'Xtick',[],'XLim',[x1 x2],'YLim',[0 1.2*maxpeak(8,1)]);%ylabel('8l')
            subplot(9,1,5);plot(tSteps(toplotstart:toplotend)-100,popRatesbad.exc(9,(toplotstart:toplotend))-popRatesbad.exc(9,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(count,:));set(gca,'XColor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(9,1))/s,'Xtick',[],'XLim',[x1 x2],'YLim',[0 1.2*maxpeak(9,1)]);%ylabel('TEO')
            subplot(9,1,6);plot(tSteps(toplotstart:toplotend)-100,popRatesbad.exc(13,(toplotstart:toplotend))-popRatesbad.exc(13,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(count,:));set(gca,'XColor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(13,1))/s,'Xtick',[],'XLim',[x1 x2],'YLim',[0 1.2*maxpeak(13,1)]);%ylabel('7A')
            subplot(9,1,7);plot(tSteps(toplotstart:toplotend)-100,popRatesbad.exc(17,(toplotstart:toplotend))-popRatesbad.exc(17,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(count,:));set(gca,'XColor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(17,1))/s,'Yticklabel',round(s*maxpeak(17,1))/s,'Xtick',[],'XLim',[x1 x2],'YLim',[0 1.2*maxpeak(17,1)]);%ylabel('9/46D')
            subplot(9,1,8);plot(tSteps(toplotstart:toplotend)-100,popRatesbad.exc(19,(toplotstart:toplotend))-popRatesbad.exc(19,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(count,:));set(gca,'XColor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(19,1))/s,'Xtick',[],'XLim',[x1 x2],'YLim',[0 1.2*maxpeak(19,1)]);%ylabel('TEpd')
            subplot(9,1,9);plot(tSteps(toplotstart:toplotend)-100,popRatesbad.exc(29,(toplotstart:toplotend))-popRatesbad.exc(29,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(count,:));set(gca,'XColor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(29,1))/s,'Yticklabel',round(s*maxpeak(29,1))/s,'XLim',[x1 x2],'YLim',[0 1.2*maxpeak(29,1)]);
    
        else
             s=100;
             count = uu;
             popRatesgood = popRates;
             hFig = figure();
             set(hFig,'units', 'centimeters', 'Position', [0 0 5.7 8],'PaperUnits','centimeters','PaperPosition',[.1 -.4 5.7 8],'PaperSize',[6. 7.6])  
             subplot(9,1,1);plot(tSteps(toplotstart:toplotend)-100,popRatesgood.exc(1,(toplotstart:toplotend))-popRatesgood.exc(1,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(count,:));set(gca,'XColor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(1,1))/s,'Xtick',[],'XLim',[x1 x2],'YLim',[0 1.2*maxpeak(1,1)]);%ylabel('V1','FontSize', 18)
                 annotation('textbox', [0.45,0.9,0.1,0.1],'String', 'strong GBA','FontSize',fsizelab,'LineStyle','None');
                 annotation('textbox', [0.21,0.88,0.1,0.1],'String', '250 ms','FontSize',fsizelab,'LineStyle','None');
              hold on;subplot(9,1,2);plot(tSteps(toplotstart:toplotend)-100,popRatesgood.exc(3,(toplotstart:toplotend))-popRatesgood.exc(3,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(count,:));set(gca,'XColor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(3,1))/s,'Xtick',[],'XLim',[x1 x2],'YLim',[0 1.2*maxpeak(3,1)]);%ylabel('V4')
              hold on;subplot(9,1,3);plot(tSteps(toplotstart:toplotend)-100,popRatesgood.exc(6,(toplotstart:toplotend))-popRatesgood.exc(6,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(count,:));set(gca,'XColor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(6,1))/s, 'Yticklabel',round(s*maxpeak(6,1))/s,'Xtick',[],'XLim',[x1 x2],'YLim',[0 1.2*maxpeak(6,1)]);
              hold on;subplot(9,1,4);plot(tSteps(toplotstart:toplotend)-100,popRatesgood.exc(8,(toplotstart:toplotend))-popRatesgood.exc(8,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(count,:));set(gca,'XColor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(8,1))/s,'Xtick',[],'XLim',[x1 x2],'YLim',[0 1.2*maxpeak(8,1)]);%ylabel('8l')
              hold on;subplot(9,1,5);plot(tSteps(toplotstart:toplotend)-100,popRatesgood.exc(9,(toplotstart:toplotend))-popRatesgood.exc(9,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(count,:));set(gca,'XColor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(9,1))/s,'Xtick',[],'XLim',[x1 x2],'YLim',[0 1.2*maxpeak(9,1)]);%ylabel('TEO')
              hold on;subplot(9,1,6);plot(tSteps(toplotstart:toplotend)-100,popRatesgood.exc(13,(toplotstart:toplotend))-popRatesgood.exc(13,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(count,:));set(gca,'XColor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(13,1))/s,'Xtick',[],'XLim',[x1 x2],'YLim',[0 1.2*maxpeak(13,1)]);%ylabel('7A')
              hold on;subplot(9,1,7);plot(tSteps(toplotstart:toplotend)-100,popRatesgood.exc(17,(toplotstart:toplotend))-popRatesgood.exc(17,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(count,:));set(gca,'XColor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(17,1))/s,'Yticklabel',round(s*maxpeak(17,1))/s,'Xtick',[],'XLim',[x1 x2],'YLim',[0 1.2*maxpeak(17,1)]);%ylabel('9/46D')
              hold on;subplot(9,1,8);plot(tSteps(toplotstart:toplotend)-100,popRatesgood.exc(19,(toplotstart:toplotend))-popRatesgood.exc(19,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(count,:));set(gca,'XColor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(19,1))/s,'Xtick',[],'XLim',[x1 x2],'YLim',[0 1.2*maxpeak(19,1)]);%ylabel('TEpd')
              hold on;subplot(9,1,9);plot(tSteps(toplotstart:toplotend)-100,popRatesgood.exc(29,(toplotstart:toplotend))-popRatesgood.exc(29,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(count,:));set(gca,'XColor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(29,1))/s,'Yticklabel',round(s*maxpeak(29,1))/s,'XLim',[x1 x2],'YLim',[0 1.2*maxpeak(29,1)]);            
        end     
        h = xlabel('Time (s)','FontSize', fsize);set(h, 'Units', 'Normalized');pos = get(h, 'Position');set(h, 'Position', pos + [.01, 0.4, -0.9]);%
        set(gca,'Xtick',-98.:2:-96);set(gca,'XTickLabel',['0';'2']);%timept = [' ';'0';' ';' ';' ';' ';' ';' ';' ';'2'];set(gca,'XTickLabel',timept);
        set(gca,'FontSize',fsizelab,'box','off','Ytick',round(s*maxpeak(29,1))/s);%ylabel('18c')     
        if uu == 2
          %  annotation('textbox', [0.89,0.83,0.1,0.1],'String', 'Input','FontSize',fsizelab,'LineStyle','None');
            annotation('textbox', [0.87,0.83,0.1,0.1],'String', 'V1','FontSize',fsizelab,'LineStyle','None');
            annotation('textbox', [0.87,0.725,0.1,0.1],'String', 'V4','FontSize',fsizelab,'LineStyle','None');
            annotation('textbox', [0.87,0.645,0.1,0.1],'String', '8m','FontSize',fsizelab,'LineStyle','None');
            annotation('textbox', [0.87,0.54,0.1,0.1],'String', '8l','FontSize',fsizelab,'LineStyle','None');
            annotation('textbox', [0.87,0.445,0.1,0.1],'String', 'TEO','FontSize',fsizelab,'LineStyle','None');
            annotation('textbox', [0.87,0.353,0.1,0.1],'String', '7A','FontSize',fsizelab,'LineStyle','None');
            annotation('textbox', [0.87,0.265,0.1,0.1],'String', '9/46d','FontSize',fsizelab,'LineStyle','None');
            annotation('textbox', [0.87,0.163,0.1,0.1],'String', 'TEpd','FontSize',fsizelab,'LineStyle','None');
            annotation('textbox', [0.87,0.08,0.1,0.1],'String', '24c','FontSize',fsizelab,'LineStyle','None');
        end
        if uu == 1
        h = ylabel('Change in firing rate (Hz)','FontSize', fsize);set(h, 'Units', 'Normalized');pos = get(h, 'Position');set(h, 'Position', pos + [-.01, 6.2, -0.9]);%bad
        end
%        if uu == 1
%                 saveas(hFig, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/supp fig 4 conndens 26percent bad', 'pdf');
%        else
%                 saveas(hFig, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/supp fig 4 conndens 26percent good', 'pdf');
%        end

  end  
    
    
    
  end
end  

hFig = figure();
    set(hFig,'units', 'centimeters', 'Position', [0 0 4.3 4],'PaperUnits','centimeters','PaperPosition',[.05 .2 4.3 4],'PaperSize',[4.8 4.5])  

semilogy(frac,peakratio(:,1),'.-','LineWidth',lw,'MarkerSize',msize,'Color',cv(1,:));hold on; 
semilogy(frac,peakratio(:,2),'.-','LineWidth',lw,'MarkerSize',msize,'Color',cv(2,:));%legend('Original','New','FontSize', 24,'fontweight','bold');set(gca,'FontSize',fsize)
%hold on; semilogy(frac(12),peakratio(12,2),'vk','LineWidth',lw,'MarkerSize',msize-2,'Color','k');%ADDED
%hold on; semilogy(frac(12),peakratio(12,2),'vk','LineWidth',lw,'MarkerSize',msize-2,'Color','k');%ADDED
set(gca,'FontSize',fsizelab,'box','off');
set(gca,'Xtick',15:15:60,'XTickLabel',['15';'30';'45';'60'],'xlim',[15 65]);
set(gca,'ylim',[.99999*1e-5 2*1e-2],'Ytick',logspace(-5,-2,4),'YTickLabel',['10^{-5}';'10^{-4}';'10^{-3}';'10^{-2}'],'YMinorTick','off');
h = xlabel('Connection density','FontSize', fsize);%set(h, 'Units', 'Normalized');pos = get(h, 'Position');set(h, 'Position', pos + [-.0, -0.01, -0.9]);
h = ylabel('Propagation ratio','FontSize', fsize);%set(h, 'Units', 'Normalized');pos = get(h, 'Position');set(h, 'Position', pos + [-.01, 0., -0.9]);
%legend('control','strong GBA')
    legend({'weak GBA','strong GBA'},'Position',[0.75 0.21 0.1 0.1],'Box','off','FontSize',fsize);         

hFig2 = figure(); 
set(hFig2,'units', 'centimeters', 'Position', [0 0 4 4],'PaperUnits','centimeters','PaperPosition',[.1 .1 4 4],'PaperSize',[4.2 4.2])   
semilogx(thresvec,frac,'.-','LineWidth',lw,'MarkerSize',fsizelab);
%hold on;semilogx(thresvec(12),frac(12),'v','LineWidth',lw,'MarkerSize',fsizelab-4);%ADDED
xlabel('Threshold','FontSize',fsize);
set(gca,'FontSize',fsizelab,'box','off');
set(gca,'Xtick',logspace(-5,-2,4),'XTickLabel',['10^{-5}';'10^{-4}';'10^{-3}';'10^{-2}'],'xlim',[.99999*1e-5 2*1e-2],'XMinorTick','off');
set(gca,'Ytick',15:15:65,'YTickLabel',['15';'30';'45';'60'],'ylim',[15 65]);
h = ylabel('Connection density','FontSize', fsize);set(h, 'Units', 'Normalized');%pos = get(h, 'Position');set(h, 'Position', pos + [-.03, 0., -0.9]);


%saveas(hFig2, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/supp fig 4B', 'pdf');

end

%%%%%%%%%%%%%%%%change muEE - show bad prop or blowup %%%%%%%%%%%%%%%%%%%%%

if caseval == 5
    inpParams=struct('tStart',2,'tEnd',2.25,'ampl',22.05*1.9);
    areaInput.inp=zeros(tParams.nSteps,1);
    areaInput.inp(round(inpParams.tStart/tParams.dt):round(inpParams.tEnd/tParams.dt))=inpParams.ampl;
    %muEErange = 21:2:51;
    muEErange = 20:2:50;
    muEErange = 28:2:42;omegaEI = 19.7; %final
    
    muEErange = 28:2:42;omegaEI = 19.7;%testing only. 
    
    prop24c = zeros(length(muEErange),2);
    condA2 = zeros(length(muEErange),1); propV1 = zeros(length(muEErange),1); condA1 = zeros(length(muEErange),1); condA3 = zeros(length(muEErange),1);maxeig = zeros(length(muEErange),1);
    
    for uu = 1:2 %orig vs regime.

     for muval=1:length(muEErange)
        muEE = muEErange(muval); clear A popRates;
        if (uu==2) 
          omegaEI = 19.7 + (muEE-33.7)*55/178; %%%%%%now changing omegaEI too. 
        end   
        
        localParams.bgInh=zeros(netwParams.nNodes,1);%background inhibitory
        localParams.ee=betaE*omegaEE*(1+netwParams.alpha*netwParams.hier);%rsc*2.29119*(1+netwParams.alpha*netwParams.hier);
        localParams.ie=betaI*omegaIE*(1+netwParams.alpha*netwParams.hier);%rsc*6.09552*(1+netwParams.alpha*netwParams.hier);
        localParams.ei=-betaE*omegaEI; %-1.29954;
        localParams.ii=-betaI*omegaII; %-4.40204;        
        ldParams.ee=betaE*muEE*(1+netwParams.alpha*netwParams.hier);%rsc*2.29119*(1+netwParams.alpha*netwParams.hier);
        ldParams.ie=betaI*muIE*(1+netwParams.alpha*netwParams.hier);%rsc*6.09552*(1+netwParams.alpha*netwParams.hier);
        ldConns.ee= bsxfun(@times,flnMat,ldParams.ee);
        ldConns.ie= bsxfun(@times,flnMat,ldParams.ie);
        wEe=diag(-1+localParams.ee)+ldConns.ee;
        wEi=localParams.ei*eye(nNodes);%these ws would be 29x29 identity matrices. 
        wIe=diag(localParams.ie)+ldConns.ie;
        wIi=(-1+localParams.ii)*eye(nNodes);%compare this to eqn 4 - where does -1 come from??
        A = [wEe/localParams.tauE wEi/localParams.tauE;...
            wIe/localParams.tauI wIi/localParams.tauI];
        currParams.A=A; 
        B=currParams.A;%B=params.A;
        B(1:nNodes,:)=B(1:nNodes,:)*localParams.tauE;
        B((nNodes+1):end,:)=B((nNodes+1):end,:)*localParams.tauI;
        currs=-B*currParams.desiredSs; %currs=-B*params.desiredSs;
        localParams.bgExc=currs(1:nNodes); localParams.bgInh=currs((nNodes+1):end);
        bgCurr=[localParams.bgExc; localParams.bgInh];%bg = background
        combMat=A;
        combMat(1:nNodes,:)=combMat(1:nNodes,:)*localParams.tauE;
        combMat((nNodes+1):end,:)=combMat((nNodes+1):end,:)*localParams.tauI;
        sstate=-combMat\bgCurr;%which is -bgCurr/combMat%this is steady state. 
        netwParams.initConds.exc=sstate(1:nNodes); netwParams.initConds.inh=sstate((nNodes+1):end);
        popRates=runNetworkAreaInputNEW(tParams,localParams,netwParams,ldConns,areaInput,'threshLinear'); 
        maxpeak=zeros(nNodes,1);
        for r=1:nNodes
        maxpeak(r,1)=max(popRates.exc(r,(toplotstart:toplotend))-popRates.exc(r,toplotstart));
        end
        nodelen=1:29;nodelen=nodelen';
        prop24c(muval,uu) = maxpeak(nNodes,1);
        if (prop24c(muval,uu)>500)
        prop24c(muval,uu) = 500;
        end
        
        if uu==1
            [vcond,wcond]=sortEig(A);maxeig(muval,1)=max(real(wcond));
            disp(max(real(wcond)));
        else    
            condA2(muval,1) = sqrt( sum(svd(A).^2) - sum(abs(eig(A)).^2) ); propV1(muval,1) = maxpeak(1,1);
            [vcond,wcond]=sortEig(A);condA1(muval,1) = cond(vcond); condA3(muval,1) = norm(A*A' - A'*A,'fro');
        end
        %propv1(muval,uu) = maxpeak(1,1);if (propv1(muval,uu)>500)propv1(muval,uu) = 500;end
        disp([muEE,uu]);
     end        
    end
    cv(1,1:3) = [0.4660, 0.6740, 0.1880]; cv(2,1:3) = [0.4940,0.1840,0.5560];
    hFig = figure(); 
    if nofb == 0 
        set(hFig,'units', 'centimeters', 'Position', [0 0 3.6 4],'PaperUnits','centimeters','PaperPosition',[.1 .1 3.2 4],'PaperSize',[3.6 4.]) ;
        %set(hFig,'units', 'centimeters', 'Position', [0 0 4 3.5],'PaperUnits','centimeters','PaperPosition',[.1 .1 4 3.5],'PaperSize',[3.8 3.5]);%for xjtalk             
    else
        set(hFig,'units', 'centimeters', 'Position', [0 0 3. 4],'PaperUnits','centimeters','PaperPosition',[-.2 .1 2.6 4],'PaperSize',[2.7 4]) ;
    end
    fsize = 7; msize = 7; 
    semilogy(muEErange,prop24c(:,1),'.-','LineWidth',lw,'MarkerSize',msize, 'Color',cv(1,:));set(gca,'FontSize',fsizelab,'box','off');
    h = xlabel('Global E to E coupling','FontSize', fsize);%set(h, 'Units', 'Normalized');%pos = get(h, 'Position');set(h, 'Position', pos + [-.0, -0.01, -0.9]);
    %hold on; semilogy(muEErange(4),prop24c(4,1),'x','LineWidth',lw,'MarkerSize',msize, 'Color',cv(1,:));
    %hold on; semilogy(muEErange(5),prop24c(5,1),'o','LineWidth',lw,'MarkerSize',msize, 'Color',cv(1,:));
    
    %xlabel({'line 1';'line2'});
    %xlabel({'Global coupling';'Global coupling'},'FontSize', fsize);
  
    %set(gca,'Xtick',30:5:40,'XTickLabel',['30';'35';'40'],'xlim',[28. 42]);
    %set(gca,'Ytick',logspace(-6,3,4),'YTickLabel',['10^{-6}';'10^{-3}';'10^{0} ';'10^{3} '],'ylim',[1e-6, 1e3]);
    set(gca,'Ytick',logspace(-8,4,4),'YTickLabel',['10^{-8}';'10^{-4}';'10^{0} ';'10^{4} '],'ylim',[1e-8, 1e4]);
    set(gca,'Xtick',20:10:50,'XTickLabel',['20';'30';'40';'50'],'xlim',[20. 50]);
    
    if nofb==0
        h = ylabel('Maximum rate in 24c (Hz)','FontSize', fsize);%set(h, 'Units', 'Normalized');%pos = get(h, 'Position');set(h, 'Position', pos + [-.02, 0., -0.9]);
    else
        set(gca,'ycolor',get(gcf,'color') ) ;     
    end
    %saveas(hFig, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/main fig 1 24cblowup', 'pdf')
    %saveas(hFig, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/main fig 1C ', 'pdf')
    
    %saveas(hFig, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/main fig 1 24c no fb slowblowup', 'pdf')
    %saveas(hFig, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/main fig 1C nofb', 'pdf')
    
    %saveas(hFig, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/suppfig8_nohier', 'pdf')
    %saveas(hFig, '/Users/maddy/Dropbox/rishicode_maddy070115/xj columbia talk/main fig 2I only green', 'pdf');

    hFig = figure();  fsize = 7; msize = 7;
        set(hFig,'units', 'centimeters', 'Position', [0 0 4 3.5],'PaperUnits','centimeters','PaperPosition',[.1 .1 4 3.5],'PaperSize',[3.8 3.5])
    semilogy(muEErange,prop24c(:,1),'.-','LineWidth',lw,'MarkerSize',msize,'Color',cv(1,:));set(gca,'FontSize',fsizelab,'box','off');
    hold on; semilogy(muEErange,prop24c(:,2),'.-','LineWidth',lw,'MarkerSize',msize,'Color',cv(2,:));set(gca,'box','off');
    h = xlabel('Global E to E coupling','FontSize', fsize);set(h, 'Units', 'Normalized');%pos = get(h, 'Position');set(h, 'Position', pos + [-.0, -0.01, -0.9]);
    h = ylabel('Maximum rate in 24c (Hz)','FontSize', fsize);set(h, 'Units', 'Normalized');%pos = get(h, 'Position');set(h, 'Position', pos + [-.02, 0., -0.9]);
    set(gca,'Xtick',20:10:50,'XTickLabel',['20';'30';'40';'50'],'xlim',[20 52]);
    %legend({'increase \mu_{EE}', 'increase \mu_{EE}, \omega_{EI}'},'Position',[0.58 0.2 0.1 0.1],'Box','off','FontSize',7);
    set(gca,'Ytick',logspace(-8,4,4),'YTickLabel',['10^{-8}';'10^{-4}';'10^{0} ';'10^{4} '],'ylim',[1e-8, 1e4]);
    %saveas(hFig, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/main fig 3 24cgoodandbad', 'pdf');
    %saveas(hFig, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/main fig 2I', 'pdf');

    
    hFig = figure();set(hFig,'units', 'centimeters', 'Position', [0 0 4 4],'PaperUnits','centimeters','PaperPosition',[.1 .1 4 4],'PaperSize',[4. 4.])
    semilogx(prop24c(:,2),condA2,'.-','LineWidth',lw,'MarkerSize',msize,'Color',cv(2,:));set(gca,'FontSize',fsizelab,'box','off');
    h = xlabel('Maximum rate in 24c (Hz)', 'FontSize',fsize);set(h, 'Units', 'Normalized');
    h = ylabel('Nonnormality measure','FontSize', fsize);set(h, 'Units', 'Normalized');
    set(gca,'Xtick',logspace(-7,1,3),'XTickLabel',['10^{-7}';'10^{-3}';'10^{1} '],'xlim',[1e-7 1e1]);
    set(gca,'Ytick',4500:150:4650,'YTickLabel',['4500';'4650'],'ylim',[4500, 4650]);
    %saveas(hFig, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/supp fig nonnormality', 'pdf');

    
    %%%%%%%%%%% brain figure %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    load('2012_12_10_data_sorted_coordinates.mat');
    fsize = 6; msize = 6; 
    crds=coord.sortCrdList;
    %hFig2 = figure(); set(hFig2,'units', 'centimeters', 'Position', [0 0 8.7 6.4],'PaperUnits','centimeters','PaperPosition',[.0 .0 8.7 6.4],'PaperSize',[8.7 6.4]) 
    hFig2 = figure(); set(hFig2,'units', 'centimeters', 'Position', [0 0 6.4 4.6],'PaperUnits','centimeters','PaperPosition',[.0 .0 6.4 4.6],'PaperSize',[6.4 4.6]) 
    
    axes1 = axes('Parent',hFig2); hold on;
    plot3(crds(:,1),crds(:,2),crds(:,3),'.','MarkerSize',msize,'Color',[0 0 0]);
    
    % Now start to fill in the strong connections Symmetrize, for now
    symmConns=(flnMat+flnMat')/2; 
    thresh=5e-3;
    lnWt=@(x) log10(x/thresh)/log10(1/thresh);
    % Can set various thresholds for what gets plotted and scale the weights in various ways
    for i=1:nNodes
        for j=1:(i-1)
            if(symmConns(i,j)>thresh)
                lineToPlot=[crds(i,:); crds(j,:)];
                plot3(lineToPlot(:,1),lineToPlot(:,2),lineToPlot(:,3),'LineWidth',(.1*lnWt(symmConns(i,j))),'Color', [0.9290    0.6940    0.1250]);
            end
        end
    end
    view(axes1,[106 18]); 
    labels=areaList;
    xc=crds(:,1); yc=crds(:,2); zc=crds(:,3); text(xc,yc,zc,labels,'FontSize',fsize);
    plot3(crds(:,1),crds(:,2),crds(:,3),'.','MarkerSize',msize,'Color',[0 0 0]);
    set(gca,'xlim',[min(crds(:,1)) max(crds(:,1))],'ylim',[min(crds(:,2)) max(crds(:,2))],'zlim',[min(crds(:,3)) max(crds(:,3))]);axis off;
 %   saveas(hFig2, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/main fig 3 brain', 'pdf')
   % saveas(hFig2, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/main fig 1A', 'pdf')

    hFig2 = figure(); set(hFig2,'units', 'centimeters', 'Position', [0 0 8.7 6.4],'PaperUnits','centimeters','PaperPosition',[.0 .0 8.7 6.4],'PaperSize',[8.7 6.4]) 
    axes1 = axes('Parent',hFig2); hold on;
    plot3(crds(:,1),crds(:,2),crds(:,3),'.','MarkerSize',msize,'Color',[0 0 0]);
    for i=1:nNodes
        for j=1:(i-1)
            if(symmConns(i,j)>thresh)
                lineToPlot=[crds(i,:); crds(j,:)];
                plot3(lineToPlot(:,1),lineToPlot(:,2),lineToPlot(:,3),'LineWidth',(.1*lnWt(symmConns(i,j))),'Color', [0.9290    0.6940    0.1250]);
            end
        end
    end
    view(axes1,[106 18]); 
    xc=crds(:,1); yc=crds(:,2); zc=crds(:,3);% text(xc,yc,zc,labels,'FontSize',fsize);
    plot3(crds(:,1),crds(:,2),crds(:,3),'.','MarkerSize',msize,'Color',[0 0 0]);
    set(gca,'xlim',[min(crds(:,1)) max(crds(:,1))],'ylim',[min(crds(:,2)) max(crds(:,2))],'zlim',[min(crds(:,3)) max(crds(:,3))]);axis off;
%    saveas(hFig2, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/main fig 3 brainsmall', 'pdf')

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
end

if caseval == 6
    fsize=8;fsizelab=8;msize=8;
    
    ventarea = [1:3,9,19];flnMatpos=zeros(500,1);
    count=0;
    for i=1:29 
        for j=1:29 
            if(flnMat(i,j)>0) 
                count=count+1;flnMatpos(count,1) = flnMat(i,j);
            end; 
        end;
    end;
    flnMatpos=flnMatpos(1:count,1);
    
    count=0;flnMatvisven=zeros(500,1);
    for i = 1:length(ventarea) 
        for j = 1:length(ventarea) 
            if(flnMat(ventarea(i),ventarea(j))>0) 
                count=count+1;
                flnMatvisven(count,1)=(flnMat(ventarea(i),ventarea(j)));
                disp([i, j, flnMatvisven(count,1)]);
            end; 
        end;
    end;
    flnMatvisven = flnMatvisven(1:count,1);
    
hFig = figure(); set(hFig,'units', 'centimeters', 'Position', [0 0 4 3],'PaperUnits','centimeters','PaperPosition',[.1 .1 4 3],'PaperSize',[4.1 3.1]) 
hist(flnMatpos,logspace(-6,0,13));set(gca,'xscale','log');h = findobj(gca,'Type','patch');
set(h,'FaceColor',[0.4940    0.1840    0.5560],'EdgeColor','w','FaceAlpha',0.35,'LineWidth',0.2);
%h = findobj(gca,'Type','line');set(h,'Marker','none'); 
hold on;
hist(flnMatvisven,logspace(-6,0,13));
h = findobj(gca,'Type','line');set(h,'Marker','none'); 
%h = findobj(gca,'Type','patch');%set(h,'EdgeAlpha',0.8);
set(gca,'FontSize',fsizelab,'box','off');
set(gca,'xlim',[1e-6 3],'Xtick',logspace(-6,0,4),'XTickLabel',['10^{-6}';'10^{-4}';'10^{-2}';'10^{0} ']);
set(gca,'ylim',[0 70],'Ytick',0:20:60,'YTickLabel',['0 ';'20';'40';'60']);
ylabel('FLN counts','FontSize', fsize);%set(h, 'Units', 'Normalized');pos = get(h, 'Position');set(h, 'Position', pos + [-.01, -0.0, -0.9]);
[counts,~]=hist(flnMatpos,logspace(-6,0,13));[countsvv,~]=hist(flnMatvisven,logspace(-6,0,13));
disp(sum(countsvv(10:13))/sum(counts(10:13)));

%saveas(hFig, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/main fig 5 histfln', 'pdf')

    hFig = figure(); 
    %set(hFig,'units', 'centimeters', 'Position', [0 0 4 2.5],'PaperUnits','centimeters','PaperPosition',[.1 .1 4 2.5],'PaperSize',[4.1 2.6])
set(hFig,'units', 'centimeters', 'Position', [0 0 4.2 2.5],'PaperUnits','centimeters','PaperPosition',[.1 .1 4.2 2.5],'PaperSize',[4.3 2.6])

hist(flnMatpos,logspace(-6,1,15));set(gca,'xscale','log');h = findobj(gca,'Type','patch');
%set(h,'FaceColor',[0.4940    0.1840    0.5560],'EdgeColor','w','FaceAlpha',0.35,'LineWidth',0.2);
set(h,'FaceColor',[0.4940    0.1840    0.5560],'FaceAlpha',0.35,'LineWidth',0.2);
h = findobj(gca,'Type','line');set(h,'Marker','none'); h = findobj(gca,'Type','patch');%set(h,'EdgeAlpha',0.8);
set(gca,'FontSize',fsizelab,'box','off');
set(gca,'xlim',[1e-6 3],'Xtick',logspace(-6,0,4),'XTickLabel',['10^{-6}';'10^{-4}';'10^{-2}';'10^{0} ']);
set(gca,'ylim',[0 70],'Ytick',0:20:60,'YTickLabel',['0 ';'20';'40';'60']);
ylabel('FLN counts','FontSize', fsize);%set(h, 'Units', 'Normalized');pos = get(h, 'Position');set(h, 'Position', pos + [-.01, -0.0, -0.9]);
[counts,~]=hist(flnMatpos,logspace(-6,0,13));
disp(sum(~(10:13))/sum(counts(10:13)));
%saveas(hFig, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/main fig 5 histfln1', 'pdf')
saveas(hFig, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/main fig 4F', 'pdf')

    hFig = figure(); 
    %set(hFig,'units', 'centimeters', 'Position', [0 0 4 2.5],'PaperUnits','centimeters','PaperPosition',[.1 .1 4 2.5],'PaperSize',[4.1 2.9]) 
    set(hFig,'units', 'centimeters', 'Position', [0 0 4.2 2.5],'PaperUnits','centimeters','PaperPosition',[.1 .1 4.2 2.5],'PaperSize',[4.3 2.9]) 
    %hist(flnMatvisven,logspace(-4,1,11));
hist(flnMatvisven,logspace(-6,1,15));
set(gca,'xscale','log');h = findobj(gca,'Type','patch');
set(h,'FaceColor',[0.4940    0.1840    0.5560],'FaceAlpha',0.75,'LineWidth',0.2);
h = findobj(gca,'Type','line');set(h,'Marker','none'); 
%h = findobj(gca,'Type','patch');%set(h,'EdgeAlpha',0.8);
set(gca,'FontSize',fsizelab,'box','off');
    set(gca,'xlim',[1e-6 3],'Xtick',logspace(-6,0,4),'XTickLabel',['10^{-6}';'10^{-4}';'10^{-2}';'10^{0} ']);%['10^{-6}';'10^{-4}';'10^{-2}';'10^{0} ']);'XTickLabel',[]
    set(gca,'ylim',[0 5],'Ytick',0:5:5,'YTickLabel',['0 ';'5 ']);
h = ylabel('FLNs: ventral stream','FontSize', fsize);%set(h, 'Units', 'Normalized');pos = get(h, 'Position');set(h, 'Position', pos + [-.01, -0.0, -0.9]);

[countsvv,centersvv]=hist(flnMatvisven,logspace(-4,1,11));
%disp(sum(countsvv(10:13))/sum(counts(10:13)));
%saveas(hFig, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/main fig 5 histfln2', 'pdf')
saveas(hFig, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/main fig 4G', 'pdf')

%all vs feedback
flnMatfb=zeros(500,1);
    count=0;
    for i=1:29 
        for j=i:29 
            if(flnMat(i,j)>0) 
                count=count+1;flnMatfb(count,1) = flnMat(i,j);
            end; 
        end;
    end;
flnMatfb=flnMatfb(1:count,1);
hFig = figure(); set(hFig,'units', 'centimeters', 'Position', [0 0 4 3],'PaperUnits','centimeters','PaperPosition',[.1 .1 4 3],'PaperSize',[4.1 3.1]) 
hist(flnMatpos,logspace(-6,1,15));set(gca,'xscale','log');h = findobj(gca,'Type','patch');
set(h,'FaceColor',[0.4940    0.1840    0.5560],'EdgeColor','w','FaceAlpha',0.35,'LineWidth',0.2);
%h = findobj(gca,'Type','line');set(h,'Marker','none'); 
hold on;
hist(flnMatfb,logspace(-6,1,15));
h = findobj(gca,'Type','line');set(h,'Marker','none'); 
%h = findobj(gca,'Type','patch');%set(h,'EdgeAlpha',0.8);
set(gca,'FontSize',fsizelab,'box','off');
set(gca,'xlim',[1e-6 3],'Xtick',logspace(-6,0,4),'XTickLabel',['10^{-6}';'10^{-4}';'10^{-2}';'10^{0} ']);
set(gca,'ylim',[0 70],'Ytick',0:20:60,'YTickLabel',['0 ';'20';'40';'60']);
ylabel('FLN counts','FontSize', fsize);%set(h, 'Units', 'Normalized');pos = get(h, 'Position');set(h, 'Position', pos + [-.01, -0.0, -0.9]);
[counts,~]=hist(flnMatpos,logspace(-6,0,13));[countsvv,~]=hist(flnMatfb,logspace(-6,0,13));
disp(sum(countsvv)/sum(counts)); %.5093
%saveas(hFig, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/main fig fb fln', 'pdf')
        

%one-directional. 
count=0;flnMatonedir=zeros(500,1);
    for i = 1:29 
        for j = 1:i 
            if( (flnMat(i,j)*flnMat(j,i)==0) && (flnMat(i,j) + flnMat(j,i) > 0) )
                count=count+1;
                flnMatonedir(count,1)= max( flnMat(i,j) , flnMat(j,i) );
            end; 
        end;
    end;
flnMatonedir = flnMatonedir(1:count,1);

hFig = figure(); set(hFig,'units', 'centimeters', 'Position', [0 0 4 3],'PaperUnits','centimeters','PaperPosition',[.1 .1 4 3],'PaperSize',[4.1 3.1]) 
hist(flnMatpos,logspace(-6,1,15));set(gca,'xscale','log');h = findobj(gca,'Type','patch');
set(h,'FaceColor',[0.4940    0.1840    0.5560],'EdgeColor','w','FaceAlpha',0.35,'LineWidth',0.2);
%h = findobj(gca,'Type','line');set(h,'Marker','none'); 
hold on;
hist(flnMatonedir,logspace(-6,1,15));
h = findobj(gca,'Type','line');set(h,'Marker','none'); 
%h = findobj(gca,'Type','patch');%set(h,'EdgeAlpha',0.8);
set(gca,'FontSize',fsizelab,'box','off');
set(gca,'xlim',[1e-6 3],'Xtick',logspace(-6,0,4),'XTickLabel',['10^{-6}';'10^{-4}';'10^{-2}';'10^{0} ']);
set(gca,'ylim',[0 70],'Ytick',0:20:60,'YTickLabel',['0 ';'20';'40';'60']);
ylabel('FLN counts','FontSize', fsize);%set(h, 'Units', 'Normalized');pos = get(h, 'Position');set(h, 'Position', pos + [-.01, -0.0, -0.9]);
[counts,~]=hist(flnMatpos,logspace(-6,0,13));[countsvv,~]=hist(flnMatonedir,logspace(-6,0,13));
disp(sum(countsvv)/sum(counts)); %.2015
%saveas(hFig, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/main fig fln one dir', 'pdf')

flnMatthres=zeros(500,1); thres = .0023;
    count=0;
    for i=1:29 
        for j=1:29 
            if(flnMat(i,j)>thres) 
                count=count+1;flnMatthres(count,1) = flnMat(i,j);
                if (i==2 && j == 1)
                    disp('here')
                end
            end; 
        end;
    end;
flnMatthres=flnMatthres(1:count,1);
hFig = figure(); set(hFig,'units', 'centimeters', 'Position', [0 0 4 3],'PaperUnits','centimeters','PaperPosition',[.1 .1 4 3],'PaperSize',[4.1 3.1]) 
hist(flnMatpos,logspace(-6,1,15));set(gca,'xscale','log');h = findobj(gca,'Type','patch');
set(h,'FaceColor',[0.4940    0.1840    0.5560],'EdgeColor','w','FaceAlpha',0.35,'LineWidth',0.2);
%h = findobj(gca,'Type','line');set(h,'Marker','none'); 
hold on;
hist(flnMatthres,logspace(-6,1,15));
h = findobj(gca,'Type','line');set(h,'Marker','none'); 
%h = findobj(gca,'Type','patch');%set(h,'EdgeAlpha',0.8);
set(gca,'FontSize',fsizelab,'box','off');
set(gca,'xlim',[1e-6 3],'Xtick',logspace(-6,0,4),'XTickLabel',['10^{-6}';'10^{-4}';'10^{-2}';'10^{0} ']);
set(gca,'ylim',[0 70],'Ytick',0:20:60,'YTickLabel',['0 ';'20';'40';'60']);
ylabel('FLN counts','FontSize', fsize);%set(h, 'Units', 'Normalized');pos = get(h, 'Position');set(h, 'Position', pos + [-.01, -0.0, -0.9]);
[counts,~]=hist(flnMatpos,logspace(-6,0,13));[countsvv,~]=hist(flnMatthres,logspace(-6,0,13));
disp(sum(countsvv)/sum(counts)); %
%saveas(hFig, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/main fig thres fln', 'pdf')

end

if caseval == 7
    
    wmillerEErange = 4.:.025:6.5; wmillerEIrange = 4.5:.025:7.;%for fig. 


    wmillerorig=4 + 2/7; kmillerorig=1.1;
    wmillerorigtoplot=4.45; kmillerorigtoplot=4.7/wmillerorigtoplot;

    taue=.02; 
    taumatrix = [1/taue;1/taue];  
    tmpzero = zeros(2,1);
    maxamplocal = zeros(length(wmillerEIrange),length(wmillerEErange));
    eigA = zeros(length(wmillerEIrange),length(wmillerEErange));

    tstep = .0001; tmax=2;
    trange = 0:tstep:tmax;
    init = [1;0];
    state_m=zeros(2,length(trange));state_m(1:2,1) = init;

    wexctoinh = wmillerorig; winhself = wmillerorig*kmillerorig;

%     for s=1:length(wmillerEErange)   %comment to make run faster - save
%     values
%     %data for contour. 
%          for ss=1:length(wmillerEIrange)
%             disp([s, ss]);
% 
%       wmiller = wmillerEErange(s); wkmiller=wmillerEIrange(ss); 
% 
%       wexcself = wmiller; 
%       winhtoexc = wkmiller; 
% 
%       A = [(wexcself-1) -winhtoexc; wexctoinh (-winhself-1)];
%       A = bsxfun(@times, A,taumatrix);
% 
%     for time=1:length(trange)-1
%     state_m(1:2,time+1) = max(state_m(1:2,time)+ tstep*A*state_m(1:2,time), tmpzero);
%     end
%     [~,wcond]=eig(A);
%     eigA(ss,s) = max(real(diag(wcond)));
% 
%     if (eigA(ss,s)<0)
%     maxamplocal(ss,s) = max(state_m(1,:)); 
%     end;
%         end
%     end 

    
    hFig = figure(); fsizelab = 7; fsize=7;mksize = 4;
    %set(hFig,'units', 'centimeters', 'Position', [0 0 7. 4.],'PaperUnits','centimeters','PaperPosition',[.1 .1 7. 4.],'PaperSize',[6.7 4.]) 
    %save('maxampsaved','maxampsaved')   
    set(hFig,'units', 'centimeters', 'Position', [0 0 7. 4.],'PaperUnits','centimeters','PaperPosition',[.1 .1 7. 4.],'PaperSize',[7 4.3]) 
    load maxampsaved; maxamplocal = maxampsaved; 
    for i = 1:101; for j = 1:101; if maxamplocal(i,j) ==0;  maxamplocal(i,j)=-5; end;end;end; %revision added.    
    ax=gca; pos=get(gca,'pos');
    [C,h] = contourf(wmillerEErange,wmillerEIrange,maxamplocal,20);%,'Position',[pos(1)+pos(3)+.1 pos(2) .1 pos(4)]);    
    xlabel('Local E to E coupling','FontSize', fsize); ylabel('Local I to E coupling','FontSize', fsize);
    annotation('textbox', [0.4,0.25,0.1,0.1],'Color','w','String', 'unstable','FontSize',fsizelab,'LineStyle','None');
    shading flat;set(h,'LineColor','none');h = colorbar;ylabel(h, 'Maximum firing rate in E (Hz)','FontSize',fsizelab);
    set(h,'Ticks',[0,4,8]);%colorbar('Ticks',[0,1,2,3]);  
    x1=get(gca,'position');x=get(h,'Position');x(3)=0.03;set(h,'Position',x);set(gca,'position',x1);
    set(gca,'FontSize',fsizelab);
    set(gca,'Xtick',4:1:6,'XTickLabel',['4';'5';'6']); set(gca,'Ytick',4.5:1:6.5,'YTickLabel',['4.5';'5.5';'6.5']);
    %h = xlabel('Local E to E coupling','FontSize', fsize);set(h, 'Units', 'Normalized');
    %h = ylabel('Local I to E coupling','FontSize', fsize);set(h, 'Units', 'Normalized');%pos = get(h, 'Position');set(h, 'Position', pos + [-.01, -0.0, -0.9]);
    wkmillerorigtoplot = wmillerorigtoplot*kmillerorigtoplot; wmillernew = 6; wkmillernew = 6.7; 
    %hold on; plot(wmillerorigtoplot, wkmillerorigtoplot, 'v','LineWidth',.1,'MarkerEdgeColor',cv(1,:),'MarkerFaceColor',cv(1,:),'Markersize',mksize);
    %hold on;plot(wmillernew,wkmillernew,'v','LineWidth',0.1,'MarkerEdgeColor',cv(2,:),'MarkerFaceColor',cv(2,:),'Markersize',mksize);
    hold on; plot(wmillerorigtoplot, wkmillerorigtoplot, 'x','LineWidth',.1,'MarkerEdgeColor',cv(1,:),'MarkerFaceColor',cv(1,:),'Markersize',mksize);
    hold on;plot(wmillernew,wkmillernew,'x','LineWidth',.1,'MarkerEdgeColor',cv(2,:),'MarkerFaceColor',cv(2,:),'Markersize',mksize);
    %caxis([0.5 9.3]); set colorvar limit manually 
    colormap(hot); %revision    
    %saveas(hFig, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/main fig 2 contour', 'pdf');
    %saveas(hFig, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/main fig 2C', 'pdf');

    tmax=.6;trange=0:tstep:tmax;
    for count=1:2
        if count==1
          
          wmiller =  wmillerorigtoplot; wkmiller = wmillerorigtoplot*kmillerorigtoplot;wkmillerboundary=4.62;
        else      
          wmiller = 6; wkmiller = 6.7; wkmillerboundary=6.668; 
        end
        
        wexcself = wmiller; 
        winhtoexc = wkmiller;
        winhtoexc_bound = wkmillerboundary;

        A = [(wexcself-1) -winhtoexc; wexctoinh (-winhself-1)];
        A = bsxfun(@times, A,taumatrix);

        Abound = [(wexcself-1) -winhtoexc_bound; wexctoinh (-winhself-1)];
        Abound = bsxfun(@times, Abound,taumatrix);
        
        state_m=zeros(2,length(trange));state_m(1:2,1) = init;
        state_mbound=zeros(2,length(trange));state_mbound(1:2,1) = init;
        
        for time=1:length(trange)-1
        state_m(1:2,time+1) = max ( state_m(1:2,time) + tstep*A*state_m(1:2,time) , tmpzero);
        state_mbound(1:2,time+1) = max ( state_mbound(1:2,time) + tstep*Abound*state_mbound(1:2,time) , tmpzero);
        end
        
        if count==1
            lw = 1; fsizelab = 7; fsize=7;msize=7;
            state_m_old = state_m; state_mbound_old = state_mbound;
            hFig = figure();set(hFig,'units', 'centimeters', 'Position', [0 0 3.6 2.5],'PaperUnits','centimeters','PaperPosition',[.1 .1 3.6 2.5],'PaperSize',[3.8 2.8]);
            plot(trange,state_m(1,:),'g','LineWidth',lw,'MarkerSize',msize,'Color',cv(1,:));% ,'Color',[0.4660    0.6740    0.1880]);,'Color',[0.4940    0.1840    0.5560]
        else
        hold on;plot(trange,state_m(1,:),'m','LineWidth',lw,'MarkerSize',msize,'Color',cv(2,:)); grid off;set(gca,'FontSize',fsizelab,'box','off');            
        h = xlabel('Time (s)','FontSize', fsize);set(h, 'Units', 'Normalized');%pos = get(h, 'Position');set(h, 'Position', pos + [-.0, -0.01, -0.9]);
        h = ylabel('Excitatory rate (Hz)','FontSize', fsize);set(h, 'Units', 'Normalized');%pos = get(h, 'Position');set(h, 'Position', pos + [-.02, 0., -0.9]);
        set(gca,'Xtick',0:.2:.6,'XTickLabel',['0  ';'0.2';'0.4';'0.6']);%,'xlim',[20 52]);
        set(gca,'Ytick',0:2:6,'YTickLabel',['0';'2';'4';'6']);
        legend({'weak LBA','strong LBA'},'Position',[0.61 0.89 0.1 0.1],'Box','off','FontSize',fsize);         
%        saveas(hFig, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/main fig 2 BA', 'pdf')
 %       saveas(hFig, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/main fig 2B', 'pdf')

        hFig = figure();set(hFig,'units', 'centimeters', 'Position', [0 0 4 4],'PaperUnits','centimeters','PaperPosition',[.2 .2 4 4],'PaperSize',[4.2 4.2]);
        plot(trange,state_m_old(1,:),'Color',cv(1,:),'LineWidth',lw,'MarkerSize',msize); hold on; 
        plot(trange,state_m(1,:),'Color',cv(2,:),'LineWidth',lw,'MarkerSize',msize); hold on; 
        plot(trange,state_mbound_old(1,:),'-.','Color',cv(1,:),'LineWidth',lw,'MarkerSize',msize);hold on;                 
        plot(trange,state_mbound(1,:),'-.','Color',cv(2,:),'LineWidth',lw,'MarkerSize',msize);
        grid off;set(gca,'FontSize',fsizelab,'box','off');            
        h = xlabel('Time (s)','FontSize', fsize);set(h, 'Units', 'Normalized');%pos = get(h, 'Position');set(h, 'Position', pos + [-.0, -0.01, -0.9]);
        h = ylabel('Excitatory rate (Hz)','FontSize', fsize);set(h, 'Units', 'Normalized');%pos = get(h, 'Position');set(h, 'Position', pos + [-.02, 0., -0.9]);
        %set(gca,'Xtick',0:.1:.2,'XTickLabel',['0  ';'0.1';'0.2']);%,'xlim',[20 52]);
        set(gca,'Xtick',0:.2:.6,'XTickLabel',['0  ';'0.2';'0.4';'0.6']);%,'xlim',[20 52]);
        set(gca,'Ytick',0:2:6,'YTickLabel',['0';'2';'4';'6']);
        %saveas(hFig, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/main fig 2 BAnearedge good', 'pdf')
%        saveas(hFig, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/supp fig 5 BAnearedge', 'pdf');
        end
    end
end

if caseval == 8
   
    inpParams=struct('tStart',2,'tEnd',2.25,'ampl',21.8*1.9);%,22.05*1.9);
    areaInput.inp=zeros(tParams.nSteps,1);
    areaInput.inp(round(inpParams.tStart/tParams.dt):round(inpParams.tEnd/tParams.dt))=inpParams.ampl;
    muEErange = 34:2:36;omegaEI = 19.7;
    prop24c = zeros(length(muEErange),2);

     for muval=1:length(muEErange)
        muEE = muEErange(muval); clear A popRates;
 
        localParams.bgInh=zeros(netwParams.nNodes,1);%background inhibitory
        localParams.ee=betaE*omegaEE*(1+netwParams.alpha*netwParams.hier);
        localParams.ie=betaI*omegaIE*(1+netwParams.alpha*netwParams.hier);
        localParams.ei=-betaE*omegaEI; 
        localParams.ii=-betaI*omegaII;         
        ldParams.ee=betaE*muEE*(1+netwParams.alpha*netwParams.hier);
        ldParams.ie=betaI*muIE*(1+netwParams.alpha*netwParams.hier);
        ldConns.ee= bsxfun(@times,flnMat,ldParams.ee);
        ldConns.ie= bsxfun(@times,flnMat,ldParams.ie);
        wEe=diag(-1+localParams.ee)+ldConns.ee;
        wEi=localParams.ei*eye(nNodes);%these ws would be 29x29 identity matrices. 
        wIe=diag(localParams.ie)+ldConns.ie;
        wIi=(-1+localParams.ii)*eye(nNodes);
        A = [wEe/localParams.tauE wEi/localParams.tauE;...
            wIe/localParams.tauI wIi/localParams.tauI];
        currParams.A=A; 
        B=currParams.A;%B=params.A;
        B(1:nNodes,:)=B(1:nNodes,:)*localParams.tauE;
        B((nNodes+1):end,:)=B((nNodes+1):end,:)*localParams.tauI;
        currs=-B*currParams.desiredSs; 
        localParams.bgExc=currs(1:nNodes); localParams.bgInh=currs((nNodes+1):end);
        bgCurr=[localParams.bgExc; localParams.bgInh];%bg = background
        combMat=A;
        combMat(1:nNodes,:)=combMat(1:nNodes,:)*localParams.tauE;
        combMat((nNodes+1):end,:)=combMat((nNodes+1):end,:)*localParams.tauI;
        sstate=-combMat\bgCurr;%which is -bgCurr/combMat%this is steady state. 
        netwParams.initConds.exc=sstate(1:nNodes); netwParams.initConds.inh=sstate((nNodes+1):end);
        popRates=runNetworkAreaInputNEW(tParams,localParams,netwParams,ldConns,areaInput,'threshLinear'); 
        maxpeak=zeros(nNodes,1);
        for r=1:nNodes
        maxpeak(r,1)=max(popRates.exc(r,(toplotstart:toplotend))-popRates.exc(r,toplotstart));
        end
        nodelen=1:29;nodelen=nodelen';disp(muEE);
        
        x1 = -98.25; 
        x2 = -95.5;
        s=1000;
        a = max(popRates.exc(29,(toplotstart:toplotstart+16000))-popRates.exc(29,toplotstart));
        b = max(popRates.exc(1,(toplotstart:toplotstart+16000))-popRates.exc(1,toplotstart));
        if muval==1
            hFig = figure();set(hFig,'units', 'centimeters', 'Position', [0 0 4 4],'PaperUnits','centimeters','PaperPosition',[.1 .1 3.5 4],'PaperSize',[3.7 4.1]);
        else
             hFig = figure();set(hFig,'units', 'centimeters', 'Position', [0 0 3.4 4],'PaperUnits','centimeters','PaperPosition',[-.2 .1 3.1 4],'PaperSize',[3. 4.1]);

        end
 %       plot(tSteps(toplotstart:toplotend)-100,popRates.exc(1,(toplotstart:toplotend))-popRates.exc(1,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab);set(gca,'FontSize',fsizelab,'box','off','Xtick',[],'XLim',[x1 x2]);%ylabel('V1','FontSize', 18)
 %       hold on;plot(tSteps(toplotstart:toplotend)-100,popRates.exc(29,(toplotstart:toplotend))-popRates.exc(29,toplotstart),'LineWidth',lw,'MarkerSize',fsizelab);set(gca,'FontSize',fsizelab,'box','off','XLim',[x1 x2]);
        popratesv1notzero = max(1e-2,popRates.exc(1,(toplotstart:toplotend))-popRates.exc(1,toplotstart));
        poprates24cnotzero = max(1e-2,popRates.exc(29,(toplotstart:toplotend))-popRates.exc(29,toplotstart));
        semilogy(tSteps(toplotstart:toplotend)-100,popratesv1notzero,'LineWidth',lw,'MarkerSize',fsizelab,'color',[65 182 196]/255);%cv(1,:)*.8);
        set(gca,'FontSize',fsizelab,'box','off','Xtick',[],'XLim',[x1 x2]);%ylabel('V1','FontSize', 18)
        hold on;semilogy(tSteps(toplotstart:toplotend)-100,poprates24cnotzero,'LineWidth',lw,'MarkerSize',fsizelab,'color',[0 109 44]/255);%cv(1,:)*(1/.8));
        set(gca,'FontSize',fsizelab,'box','off','XLim',[x1 x2]);

        h = xlabel('Time (s)','FontSize', fsize);
        set(gca,'Xtick',-98.:2:-96);set(gca,'XTickLabel',['0';'2']);
        if muval==1
            c = 1.2*max(a,b); set(gca,'Ytick',[round(s*maxpeak(29,1))/s,round(s*maxpeak(1,1))/s]);set(gca,'YTickLabel',[round(s*maxpeak(29,1))/s,round(s*maxpeak(1,1))/s],'YLim',[1e-2 5*c]);
            ylim1 = 5*c;
            set(gca,'Ytick',logspace(-2,2,3),'YTickLabel',['10^{-2}';'10^{0} ';'10^{2} ']);
            hold on; plot(tSteps(round(inpParams.tStart/tParams.dt):round(inpParams.tEnd/tParams.dt))-100,1.6*c*ones(size(tSteps(round(inpParams.tStart/tParams.dt):round(inpParams.tEnd/tParams.dt)))),'k');
            %%legend('V1','24c');
            
            annotation('textbox', [0.38,0.55,0.1,0.1],'String', 'V1','FontSize',fsizelab,'LineStyle','None');
            annotation('textbox', [0.47,0.22,0.1,0.1],'String', '24c','FontSize',fsizelab,'LineStyle','None');
            h = ylabel('Change in firing rate (Hz)','FontSize', fsize);%set(h, 'Units', 'Normalized');pos = get(h, 'Position');set(h, 'Position', pos + [-.01, 6.2, -0.9]);%bad
            set(gca,'FontSize',fsizelab,'box','off');annotation('textbox', [0.25,0.85,0.1,0.1],'String', '250 ms','FontSize',fsizelab,'LineStyle','None');
        else
            c = 300; %set(gca,'Ytick',[round(s*maxpeak(1,1))/s,round(s*maxpeak(29,1))/s]);
            %set(gca,'YTickLabel',[round(s*maxpeak(1,1))/s,round(s*maxpeak(29,1))/s],'YLim',[1e-2 800]);%300]);
            set(gca,'YTickLabel',[],'YLim',[1e-2 ylim1]);%800]);
            %set(gca,'Ytick',logspace(-2,2,3),'YTickLabel',['10^{-2}';'10^{0} ';'10^{2} ']);
            hold on; plot(tSteps(round(inpParams.tStart/tParams.dt):round(inpParams.tEnd/tParams.dt))-100,.68*c*ones(size(tSteps(round(inpParams.tStart/tParams.dt):round(inpParams.tEnd/tParams.dt)))),'k');
            %legend('V1','24c');
            annotation('textbox', [0.54,0.505,0.1,0.1],'String', 'V1','FontSize',fsizelab,'LineStyle','None');
            annotation('textbox', [0.55,0.85,0.1,0.1],'String', '24c','FontSize',fsizelab,'LineStyle','None');
            %h = ylabel('Change in firing rate (Hz)','FontSize', fsize);%set(h, 'Units', 'Normalized');pos = get(h, 'Position');set(h, 'Position', pos + [-.01, 6.2, -0.9]);%bad
            set(gca,'FontSize',fsizelab,'box','off');annotation('textbox', [0.13,0.85,0.1,0.1],'String', '250 ms','FontSize',fsizelab,'LineStyle','None');
            set(gca,'ycolor',get(gcf,'color') ) ;     
        end
        %h = ylabel('Change in firing rate (Hz)','FontSize', fsize);%set(h, 'Units', 'Normalized');pos = get(h, 'Position');set(h, 'Position', pos + [-.01, 6.2, -0.9]);%bad
         if muval==1
%             saveas(hFig, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/main fig 1 muEE weak', 'pdf')
         else
%             saveas(hFig, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/main fig 1 muEE blowup', 'pdf')
      %       saveas(hFig, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/main fig 1B', 'pdf')

         end
     end    
end

if caseval == 9
    load propagation.mat
    hFig2 = figure();

    set(hFig2,'units', 'centimeters', 'Position', [0 0 7.1 1.6],'PaperUnits','centimeters','PaperPosition',[0. .15 7.1 1.6],'PaperSize',[6.6 2.])     

    fsize = 6; fsizelab = 6; 
    
    semilogy(1:29,100*ratios0/ratios0(1),'.-','LineWidth',lw,'MarkerSize',fsizelab,'Color',[0.4660, 0.6740, 0.1880]);hold on;
    semilogy(1:29,100*ratios1/ratios1(1),'.-','LineWidth',lw,'MarkerSize',fsizelab,'Color',[0.4940,0.1840,0.5560]);  
    set(gca,'Ytick',logspace(-2,2,3),'YTickLabel',['10^{-2}';'10^{0} ';'10^{2} ']);
    legend({'weak GBA','strong GBA'},'Position',[0.69 0.85 0.1 0.1],'Box','off','FontSize',fsize);  %  'Position',[0.71 0.84 0.1 0.1]     
    %xlabel('Areas','FontSize',fsize);
    set(gca,'FontSize',fsizelab);
    set(gca,'Xtick',1:1:29,'box','off');
    areapt = ['  V1 ';'  V2 ';'  V4 ';'  DP ';'  MT ';'  8m ';'  5  ';'  8l ';' TEO ';'  2  ';'  F1 ';' STPc';'  7A ';' 46d ';'  10 ';'9/46v';'9/46d';'  F5 ';' TEpd';' PBr ';' 7m  ';' 7B  ';'  F2 ';' STPi';' PROm';' F7  ';' 8B  ';' STPr';' 24c ']; %timept = ['';'0';'';'';'1';'';'';'2'];
    %set(gca,'XTickLabel',areapt);
    set(gca,'XTickLabel',[])
    ax = gca;ax.XTickLabelRotation = 90; 
    h = ylabel(sprintf('Peak firing rate (Hz)'),'FontSize',fsize);
    pos = get(h, 'Position');set(h, 'Position', pos + [-.02, .26, -0.9]);%
     
    %pos = get(h, 'Position');set(h, 'Position', pos + [-.02, -.06, -0.9]);%
      
 %   saveas(hFig2, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/main fig 4 propgoodmaxrate', 'pdf');    
     saveas(hFig2, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/main fig 3B', 'pdf');    

end

if caseval == 10
    load prop10.mat; for10mean = meangamma; for10sig = siggamma; 
    load prop20.mat; for20mean = meangamma; for20sig = siggamma; 
    load prop30.mat; for30mean = meangamma; for30sig = siggamma; 
    load prop40.mat; for40mean = meangamma; for40sig = siggamma; 
  
    hFig2 = figure();

    set(hFig2,'units', 'centimeters', 'Position', [0 0 7.1 2.3],'PaperUnits','centimeters','PaperPosition',[0. .01 7.1 2.3],'PaperSize',[6.6 2.3])     
    
    fsize = 6; fsizelab = 6; 
    
    errorbar(1:29,for10mean,for10sig,'.-','LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(1,:));hold on;

    errorbar(1:29,for40mean,for40sig,'.-','LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(2,:));
    xlabel('Areas','FontSize',fsize);
    set(gca,'FontSize',fsizelab);
    set(gca,'Xtick',1:1:29,'box','off');%set(gca,'Xtick',0:1:28,'box','off');
    areapt = ['  V1 ';'  V2 ';'  V4 ';'  DP ';'  MT ';'  8m ';'  5  ';'  8l ';' TEO ';'  2  ';'  F1 ';' STPc';'  7A ';' 46d ';'  10 ';'9/46v';'9/46d';'  F5 ';' TEpd';' PBr ';' 7m  ';' 7B  ';'  F2 ';' STPi';' PROm';' F7  ';' 8B  ';' STPr';' 24c ']; %timept = ['';'0';'';'';'1';'';'';'2'];
    set(gca,'XTickLabel',areapt);
    set(gca,'ylim',[.0012 .005],'Ytick',0.001:.002:.005);%,'YTickLabel',['0.001';'0.003';'0.005']);
    ax = gca;ax.XTickLabelRotation = 90; 
    h = ylabel(sprintf('Gamma power'),'FontSize',fsize);%set(h, 'Units', 'Normalized');pos = get(h, 'Position');set(h, 'Position', pos + [-.02, 0., -0.9]);%
    %saveas(hFig2, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/main fig 4 gamma', 'pdf');    
     saveas(hFig2, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/main fig 3C', 'pdf');    

end

if caseval == 11
    load 'maxpeakbad_control.mat'; load 'maxpeakbad_nofb.mat'; load 'maxpeakbad_arimean.mat'; load 'maxpeakbad_geomean.mat'; load 'maxpeakbad_nohier.mat';
    load 'maxpeakgood_control.mat'; load 'maxpeakgood_nofb.mat'; load 'maxpeakgood_arimean.mat'; load 'maxpeakgood_geomean.mat';load 'maxpeakgood_nohier.mat';
    nodelen=1:29;nodelen=nodelen';
    cy = [237 177 32]/255;
    hFig = figure();
    set(hFig,'units', 'centimeters', 'Position', [0 0 7.5 4.5],'PaperUnits','centimeters','PaperPosition',[.05 .05 7.5 4.5],'PaperSize',[8.6 4.4]) ;
    semilogy(nodelen-1,100*(maxpeakbad_control)/maxpeakbad_control(1,1),'.-','LineWidth',lw,'MarkerSize',fsizelab,'Color','k');hold on;
    semilogy(nodelen-1,100*(maxpeakbad_nofb)/maxpeakbad_nofb(1,1),'.-','LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(1,:));hold on;
    semilogy(nodelen-1,100*(maxpeakbad_geomean)/maxpeakbad_geomean(1,1),'.-','LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(2,:));hold on;
%     semilogy(nodelen-1,100*(maxpeakbad_arimean)/maxpeakbad_arimean(1,1),'.-','LineWidth',lw,'MarkerSize',fsizelab,'Color',cy);hold on;

    %if caseval==1   
        hold on; semilogy((6:2:8)-1,100*(maxpeakbad_nofb(6:2:8,1))/maxpeakbad_nofb(1,1),':','LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(1,:));
        hold on; semilogy((9:2:11)-1,100*(maxpeakbad_nofb(9:2:11,1))/maxpeakbad_nofb(1,1),':','LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(1,:));
        set(gca,'Ytick',logspace(-6,2,3),'YTickLabel',['10^{-6}';'10^{-2}';'10^{2} '],'ylim',[1e-8 1e2]);
    %else
    %    set(gca,'Ytick',logspace(-4,2,4),'YTickLabel',['10^{-4}';'10^{-2}';'10^{0} ';'10^{2} ']);
        %legend({'weak GBA','no fb','geom. mean'},'Position',[0.68 0.35 0.1 0.1],'Box','off','FontSize',fsize);  
%         legend({'weak GBA','no fb','geom. mean','ari. mean'},'Position',[0.9 0.4 0.1 0.1],'Box','off','FontSize',fsize);  

    %end
    
    xlabel('Areas','FontSize',fsize);
    set(gca,'FontSize',fsizelab);
    set(gca,'Xtick',0:1:28,'box','off');
    areapt = ['  V1 ';'  V2 ';'  V4 ';'  DP ';'  MT ';'  8m ';'  5  ';'  8l ';' TEO ';'  2  ';'  F1 ';' STPc';'  7A ';' 46d ';'  10 ';'9/46v';'9/46d';'  F5 ';' TEpd';' PBr ';' 7m  ';' 7B  ';'  F2 ';' STPi';' PROm';' F7  ';' 8B  ';' STPr';' 24c ']; %timept = ['';'0';'';'';'1';'';'';'2'];
    set(gca,'XTickLabel',areapt);
    ax = gca;ax.XTickLabelRotation = 90; 
    h = ylabel('Maximum firing rate (Hz)','FontSize',fsize);%set(h, 'Units', 'Normalized');pos = get(h, 'Position');set(h, 'Position', pos + [-.02, 0., -0.9]);%
    %title('Weak case');
    
    hFig2 = figure();
    set(hFig2,'units', 'centimeters', 'Position', [0 0 7.5 4.5],'PaperUnits','centimeters','PaperPosition',[.05 .05 7.5 4.5],'PaperSize',[7. 4.4]) ;
    semilogy(nodelen-1,100*(maxpeakgood_control)/maxpeakgood_control(1,1),'.-','LineWidth',lw,'MarkerSize',fsizelab,'Color','k');hold on;
    semilogy(nodelen-1,100*(maxpeakgood_nofb)/maxpeakgood_nofb(1,1),'.-','LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(1,:));hold on;
    semilogy(nodelen-1,100*(maxpeakgood_geomean)/maxpeakgood_geomean(1,1),'.-','LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(2,:));hold on;
%     semilogy(nodelen-1,100*(maxpeakgood_arimean)/maxpeakgood_arimean(1,1),'.-','LineWidth',lw,'MarkerSize',fsizelab,'Color',cy);hold on;

    %if caseval==1   
        hold on; semilogy((6:2:8)-1,100*(maxpeakgood_nofb(6:2:8,1))/maxpeakgood_nofb(1,1),':','LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(1,:));
        hold on; semilogy((9:2:11)-1,100*(maxpeakgood_nofb(9:2:11,1))/maxpeakgood_nofb(1,1),':','LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(1,:));
        set(gca,'Ytick',logspace(-6,2,3),'YTickLabel',['10^{-6}';'10^{-2}';'10^{2} '],'ylim',[1e-8 1e2]);
    %else
    %    set(gca,'Ytick',logspace(-4,2,4),'YTickLabel',['10^{-4}';'10^{-2}';'10^{0} ';'10^{2} ']);
        legend({'weak GBA','no fb','geom. mean'},'Position',[0.68 0.35 0.1 0.1],'Box','off','FontSize',fsize); 
%         legend({'weak GBA','no fb','geom. mean','ari. mean'},'Position',[0.65 0.4 0.1 0.1],'Box','off','FontSize',fsize); 

      %  title('strong GBA case')
    %end
    
    xlabel('Areas','FontSize',fsize);
    set(gca,'FontSize',fsizelab);
    set(gca,'Xtick',0:1:28,'box','off');
    areapt = ['  V1 ';'  V2 ';'  V4 ';'  DP ';'  MT ';'  8m ';'  5  ';'  8l ';' TEO ';'  2  ';'  F1 ';' STPc';'  7A ';' 46d ';'  10 ';'9/46v';'9/46d';'  F5 ';' TEpd';' PBr ';' 7m  ';' 7B  ';'  F2 ';' STPi';' PROm';' F7  ';' 8B  ';' STPr';' 24c ']; %timept = ['';'0';'';'';'1';'';'';'2'];
    set(gca,'XTickLabel',areapt);
    ax = gca;ax.XTickLabelRotation = 90; 
    h = ylabel('Maximum firing rate (Hz)','FontSize',fsize);%set(h, 'Units', 'Normalized');pos = get(h, 'Position');set(h, 'Position', pos + [-.02, 0., -0.9]);%

%     saveas(hFig,'/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/supp fig bad nofb geo mean'); 
%     saveas(hFig2,'/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/supp fig good no fb geo mean'); 
%     
%     saveas(hFig, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/supp fig bad nofb ari geo mean', 'pdf');
%     saveas(hFig2, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/supp fig good no fb ari geo mean', 'pdf');

    hFig = figure(); %control and no hierarchy. 
    set(hFig,'units', 'centimeters', 'Position', [0 0 7.5 4.5],'PaperUnits','centimeters','PaperPosition',[.05 .05 7.5 4.5],'PaperSize',[8.6 4.4]) ;
    semilogy(nodelen-1,100*(maxpeakbad_nohier)/maxpeakbad_nohier(1,1),'.-','LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(1,:));hold on;
    semilogy(nodelen-1,100*(maxpeakgood_nohier)/maxpeakgood_nohier(1,1),'.-','LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(2,:));hold on;

    h2a = semilogy(nodelen-1,100*(maxpeakbad_control)/maxpeakbad_control(1,1),'-','LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(1,:));hold on;
    h2b = semilogy(nodelen-1,100*(maxpeakgood_control)/maxpeakgood_control(1,1),'-','LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(2,:));hold on;
    h2a.Color(4)=0.4; h2b.Color(4)=0.4; 
           
    set(gca,'Ytick',logspace(-6,2,3),'YTickLabel',['10^{-6}';'10^{-2}';'10^{2} '],'ylim',[1e-6 1e2]);
    %legend({'control: weak GBA','control: strong GBA,','no hier.: weak GBA','no hier.: strong GBA'},'Position',[0.68 0.35 0.1 0.1],'Box','off','FontSize',fsize);  
    legend({'weak GBA','strong GBA'},'Position',[0.78 0.85 0.1 0.1],'Box','off','FontSize',fsize);  
    xlabel('Areas','FontSize',fsize);
    set(gca,'FontSize',fsizelab);
    set(gca,'Xtick',0:1:28,'box','off');
    areapt = ['  V1 ';'  V2 ';'  V4 ';'  DP ';'  MT ';'  8m ';'  5  ';'  8l ';' TEO ';'  2  ';'  F1 ';' STPc';'  7A ';' 46d ';'  10 ';'9/46v';'9/46d';'  F5 ';' TEpd';' PBr ';' 7m  ';' 7B  ';'  F2 ';' STPi';' PROm';' F7  ';' 8B  ';' STPr';' 24c ']; %timept = ['';'0';'';'';'1';'';'';'2'];
    set(gca,'XTickLabel',areapt);
    ax = gca;ax.XTickLabelRotation = 90; 
    h = ylabel('Maximum firing rate (Hz)','FontSize',fsize);%set(h, 'Units', 'Normalized');pos = get(h, 'Position');set(h, 'Position', pos + [-.02, 0., -0.9]);%
   %  saveas(hFig, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/supp fig control no hier', 'pdf');


    hFig = figure(); %control and no feedback. 
    set(hFig,'units', 'centimeters', 'Position', [0 0 7.5 4.5],'PaperUnits','centimeters','PaperPosition',[.05 .05 7.5 4.5],'PaperSize',[8.6 4.4]) ;
    h2a = semilogy(nodelen-1,100*(maxpeakbad_control)/maxpeakbad_control(1,1),'-','LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(1,:));hold on;
    h2b = semilogy(nodelen-1,100*(maxpeakgood_control)/maxpeakgood_control(1,1),'-','LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(2,:));hold on;
    h2a.Color(4)=0.4; h2b.Color(4)=0.4; 
   
    semilogy(nodelen-1,100*(maxpeakbad_nofb)/maxpeakbad_nofb(1,1),'.-','LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(1,:));hold on;
        hold on; semilogy((6:2:8)-1,100*(maxpeakbad_nofb(6:2:8,1))/maxpeakbad_nofb(1,1),':','LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(1,:));
        hold on; semilogy((9:2:11)-1,100*(maxpeakbad_nofb(9:2:11,1))/maxpeakbad_nofb(1,1),':','LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(1,:));
    semilogy(nodelen-1,100*(maxpeakgood_nofb)/maxpeakgood_nofb(1,1),'.-','LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(2,:));hold on;
        hold on; semilogy((6:2:8)-1,100*(maxpeakgood_nofb(6:2:8,1))/maxpeakgood_nofb(1,1),':','LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(2,:));
        hold on; semilogy((9:2:11)-1,100*(maxpeakgood_nofb(9:2:11,1))/maxpeakgood_nofb(1,1),':','LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(2,:));
        
    set(gca,'Ytick',logspace(-6,2,3),'YTickLabel',['10^{-6}';'10^{-2}';'10^{2} '],'ylim',[1e-8 1e2]);
    %legend({'control: weak GBA','control: strong GBA,','no fb: weak GBA','no fb: strong GBA'},'Position',[0.68 0.35 0.1 0.1],'Box','off','FontSize',fsize);  
    xlabel('Areas','FontSize',fsize);
    set(gca,'FontSize',fsizelab);
    set(gca,'Xtick',0:1:28,'box','off');
    areapt = ['  V1 ';'  V2 ';'  V4 ';'  DP ';'  MT ';'  8m ';'  5  ';'  8l ';' TEO ';'  2  ';'  F1 ';' STPc';'  7A ';' 46d ';'  10 ';'9/46v';'9/46d';'  F5 ';' TEpd';' PBr ';' 7m  ';' 7B  ';'  F2 ';' STPi';' PROm';' F7  ';' 8B  ';' STPr';' 24c ']; %timept = ['';'0';'';'';'1';'';'';'2'];
    set(gca,'XTickLabel',areapt);
    ax = gca;ax.XTickLabelRotation = 90; 
    h = ylabel('Maximum firing rate (Hz)','FontSize',fsize);%set(h, 'Units', 'Normalized');pos = get(h, 'Position');set(h, 'Position', pos + [-.02, 0., -0.9]);%
  %   saveas(hFig, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/supp fig control nofb', 'pdf');
    

    hFig = figure(); %control and geom mean. 
    set(hFig,'units', 'centimeters', 'Position', [0 0 7.5 4.5],'PaperUnits','centimeters','PaperPosition',[.05 .05 7.5 4.5],'PaperSize',[8.6 4.4]) ;
    h2a = semilogy(nodelen-1,100*(maxpeakbad_control)/maxpeakbad_control(1,1),'-','LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(1,:));hold on;
    h2b = semilogy(nodelen-1,100*(maxpeakgood_control)/maxpeakgood_control(1,1),'-','LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(2,:));hold on;
    h2a.Color(4)=0.4; h2b.Color(4)=0.4; 
    
    semilogy(nodelen-1,100*(maxpeakbad_geomean)/maxpeakbad_geomean(1,1),'.-','LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(1,:));hold on;
    semilogy(nodelen-1,100*(maxpeakgood_geomean)/maxpeakgood_geomean(1,1),'.-','LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(2,:));hold on;
        
    set(gca,'Ytick',logspace(-6,2,3),'YTickLabel',['10^{-6}';'10^{-2}';'10^{2} '],'ylim',[1e-6 1e2]);
    %legend({'control: weak GBA','control: strong GBA,','no hier.: weak GBA','no hier.: strong GBA'},'Position',[0.68 0.35 0.1 0.1],'Box','off','FontSize',fsize);  
    xlabel('Areas','FontSize',fsize);
    set(gca,'FontSize',fsizelab);
    set(gca,'Xtick',0:1:28,'box','off');
    areapt = ['  V1 ';'  V2 ';'  V4 ';'  DP ';'  MT ';'  8m ';'  5  ';'  8l ';' TEO ';'  2  ';'  F1 ';' STPc';'  7A ';' 46d ';'  10 ';'9/46v';'9/46d';'  F5 ';' TEpd';' PBr ';' 7m  ';' 7B  ';'  F2 ';' STPi';' PROm';' F7  ';' 8B  ';' STPr';' 24c ']; %timept = ['';'0';'';'';'1';'';'';'2'];
    set(gca,'XTickLabel',areapt);
    ax = gca;ax.XTickLabelRotation = 90; 
    h = ylabel('Maximum firing rate (Hz)','FontSize',fsize);%set(h, 'Units', 'Normalized');pos = get(h, 'Position');set(h, 'Position', pos + [-.02, 0., -0.9]);%
  %   saveas(hFig, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/supp fig control geomean', 'pdf');


    hFig = figure(); %control and ari. mean. 
    set(hFig,'units', 'centimeters', 'Position', [0 0 7.5 4.5],'PaperUnits','centimeters','PaperPosition',[.05 .05 7.5 4.5],'PaperSize',[8.6 4.4]) ;
    h2a = semilogy(nodelen-1,100*(maxpeakbad_control)/maxpeakbad_control(1,1),'-','LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(1,:));hold on;
    h2b = semilogy(nodelen-1,100*(maxpeakgood_control)/maxpeakgood_control(1,1),'-','LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(2,:));hold on;
    h2a.Color(4)=0.4; h2b.Color(4)=0.4; 
    
    semilogy(nodelen-1,100*(maxpeakbad_arimean)/maxpeakbad_arimean(1,1),'.-','LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(1,:));hold on;
    semilogy(nodelen-1,100*(maxpeakgood_arimean)/maxpeakgood_arimean(1,1),'.-','LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(2,:));hold on;
        
    set(gca,'Ytick',logspace(-4,2,3),'YTickLabel',['10^{-4}';'10^{-1}';'10^{2} '],'ylim',[1e-4 1e2]);
    %legend({'control: weak GBA','control: strong GBA,','no hier.: weak GBA','no hier.: strong GBA'},'Position',[0.68 0.35 0.1 0.1],'Box','off','FontSize',fsize);  
    xlabel('Areas','FontSize',fsize);
    set(gca,'FontSize',fsizelab);
    set(gca,'Xtick',0:1:28,'box','off');
    areapt = ['  V1 ';'  V2 ';'  V4 ';'  DP ';'  MT ';'  8m ';'  5  ';'  8l ';' TEO ';'  2  ';'  F1 ';' STPc';'  7A ';' 46d ';'  10 ';'9/46v';'9/46d';'  F5 ';' TEpd';' PBr ';' 7m  ';' 7B  ';'  F2 ';' STPi';' PROm';' F7  ';' 8B  ';' STPr';' 24c ']; %timept = ['';'0';'';'';'1';'';'';'2'];
    set(gca,'XTickLabel',areapt);
    ax = gca;ax.XTickLabelRotation = 90; 
    h = ylabel('Maximum firing rate (Hz)','FontSize',fsize);%set(h, 'Units', 'Normalized');pos = get(h, 'Position');set(h, 'Position', pos + [-.02, 0., -0.9]);%
 %    saveas(hFig, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/supp fig control arimean', 'pdf');
   

 load maxpeakbad_arimeannohier.mat; load maxpeakgood_arimeannohier.mat; 
 load maxpeakbad_arimeanlochier.mat; load maxpeakgood_arimeanlochier.mat;
    
    hFig = figure(); %control and no hierarchy plus ari mean. 
    set(hFig,'units', 'centimeters', 'Position', [0 0 7.5 4.5],'PaperUnits','centimeters','PaperPosition',[.05 .05 7.5 4.5],'PaperSize',[8.6 4.4]) ;
    semilogy(nodelen-1,100*(maxpeakbad_arimeannohier)/maxpeakbad_arimeannohier(1,1),'.-','LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(1,:));hold on;
    semilogy(nodelen-1,100*(maxpeakgood_arimeannohier)/maxpeakgood_arimeannohier(1,1),'.-','LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(2,:));hold on;
    h2a = semilogy(nodelen-1,100*(maxpeakbad_control)/maxpeakbad_control(1,1),'-','LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(1,:));hold on;
    h2b = semilogy(nodelen-1,100*(maxpeakgood_control)/maxpeakgood_control(1,1),'-','LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(2,:));hold on;
    h2a.Color(4)=0.4; h2b.Color(4)=0.4;           
    set(gca,'Ytick',logspace(-6,2,3),'YTickLabel',['10^{-6}';'10^{-2}';'10^{2} '],'ylim',[1e-6 1e2]);
    %legend({'weak GBA','strong GBA'},'Position',[0.78 0.85 0.1 0.1],'Box','off','FontSize',fsize);  
    xlabel('Areas','FontSize',fsize);
    set(gca,'FontSize',fsizelab);
    set(gca,'Xtick',0:1:28,'box','off');
    areapt = ['  V1 ';'  V2 ';'  V4 ';'  DP ';'  MT ';'  8m ';'  5  ';'  8l ';' TEO ';'  2  ';'  F1 ';' STPc';'  7A ';' 46d ';'  10 ';'9/46v';'9/46d';'  F5 ';' TEpd';' PBr ';' 7m  ';' 7B  ';'  F2 ';' STPi';' PROm';' F7  ';' 8B  ';' STPr';' 24c ']; %timept = ['';'0';'';'';'1';'';'';'2'];
    set(gca,'XTickLabel',areapt);
    ax = gca;ax.XTickLabelRotation = 90; 
    h = ylabel('Maximum firing rate (Hz)','FontSize',fsize);%set(h, 'Units', 'Normalized');pos = get(h, 'Position');set(h, 'Position', pos + [-.02, 0., -0.9]);%
    %saveas(hFig, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/suppfigcontrol_arimean_nohier', 'pdf');
 
    hFig = figure(); %control and no hierarchy plus ari mean. 
    set(hFig,'units', 'centimeters', 'Position', [0 0 7.5 4.5],'PaperUnits','centimeters','PaperPosition',[.05 .05 7.5 4.5],'PaperSize',[8.6 4.4]) ;
    semilogy(nodelen-1,100*(maxpeakbad_arimeanlochier)/maxpeakbad_arimeanlochier(1,1),'.-','LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(1,:));hold on;
    semilogy(nodelen-1,100*(maxpeakgood_arimeanlochier)/maxpeakgood_arimeanlochier(1,1),'.-','LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(2,:));hold on;
    h2a = semilogy(nodelen-1,100*(maxpeakbad_control)/maxpeakbad_control(1,1),'-','LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(1,:));hold on;
    h2b = semilogy(nodelen-1,100*(maxpeakgood_control)/maxpeakgood_control(1,1),'-','LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(2,:));hold on;
    h2a.Color(4)=0.4; h2b.Color(4)=0.4;           
    %set(gca,'Ytick',logspace(-6,2,3),'YTickLabel',['10^{-6}';'10^{-2}';'10^{2} '],'ylim',[1e-6 1e2]);
    set(gca,'Ytick',logspace(-4,2,3),'YTickLabel',['10^{-4}';'10^{-1}';'10^{2} '],'ylim',[1e-4 1e2]);
    %legend({'weak GBA','strong GBA'},'Position',[0.78 0.85 0.1 0.1],'Box','off','FontSize',fsize);  
    xlabel('Areas','FontSize',fsize);
    set(gca,'FontSize',fsizelab);
    set(gca,'Xtick',0:1:28,'box','off');
    areapt = ['  V1 ';'  V2 ';'  V4 ';'  DP ';'  MT ';'  8m ';'  5  ';'  8l ';' TEO ';'  2  ';'  F1 ';' STPc';'  7A ';' 46d ';'  10 ';'9/46v';'9/46d';'  F5 ';' TEpd';' PBr ';' 7m  ';' 7B  ';'  F2 ';' STPi';' PROm';' F7  ';' 8B  ';' STPr';' 24c ']; %timept = ['';'0';'';'';'1';'';'';'2'];
    set(gca,'XTickLabel',areapt);
    ax = gca;ax.XTickLabelRotation = 90; 
    h = ylabel('Maximum firing rate (Hz)','FontSize',fsize);%set(h, 'Units', 'Normalized');pos = get(h, 'Position');set(h, 'Position', pos + [-.02, 0., -0.9]);%
    %saveas(hFig, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/suppfigcontrol_arimean_loc_hier', 'pdf');

end


if caseval == 12
    
%     colorb = 'yes';
%     d=-100;
%     if colorb == 'yes'
%        fsizelab = 7; fsize=7;mksize = 3;
%        hFig = figure();set(hFig,'units', 'centimeters', 'Position', [0 0 4. 2.],'PaperUnits','centimeters','PaperPosition',[.1 .1 4. 2.],'PaperSize',[4.7 2.]) 
%        surf(peaks);cmap=colormap('hot');h=colorbar;set(h,'Ticks',[],'ycolor',get(gcf,'color'),'box','off'); ylabel(h, 'Activity','FontSize',fsizelab);
%        %w = h.LineWidth;h.LineWidth = 1.;
%        saveas(hFig,'/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/main fig colorbar'); 
%        saveas(hFig, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/main fig colorbar', 'pdf');
% 
%     end
    
    
    load /Users/maddy/Dropbox/rishicode_maddy070115/caret2/brate;  
    load /Users/maddy/Dropbox/rishicode_maddy070115/caret2/grate;    
    load /Users/maddy/Dropbox/rishicode_maddy070115/caret2/bspike;  
    load /Users/maddy/Dropbox/rishicode_maddy070115/caret2/gspike;
    
    load /Users/maddy/Dropbox/rishicode_maddy070115/caret2/badcons_4pt5;
    load /Users/maddy/Dropbox/rishicode_maddy070115/caret2/goodcons_5pt5;
    load /Users/maddy/Dropbox/rishicode_maddy070115/caret2/nofbgoodcons_5pt5;
    load /Users/maddy/Dropbox/rishicode_maddy070115/caret2/cons_5;

    load /Users/maddy/Dropbox/rishicode_maddy070115/caret2/brateS1;  
    load /Users/maddy/Dropbox/rishicode_maddy070115/caret2/grateS1;    

    nNodes = 29;
   
    cmap=colormap('hot');
    % Take value between 0 and 1 and a matrix. Return the nth row of the matrix, where n/length(mat) is closest to x.
    f=@(x,mat) mat(round(x*(length(mat)-1))+1,:);
    %various caret figures. 
    
  %   logTau = min(.8,1.8*(1 - brate)); d=1;
  %   logTau = min(.8,1.8*(1 - grate)); d=2;
  %   logTau = (1-bspike); d=3;
  %   logTau = (1-gspike); d=4;
  %   logTau =  1-  badcons_4pt5;d=5;
  %   logTau =  1 - goodcons_5pt5;d=6;
  %   logTau =  1 - nofbgoodcons_5pt5;d=7;
  %   logTau =  1 - cons_5;d=8;
  
  %   logTau = min(.8,1.8*(1 - brateS1)); d=-1;
  %   logTau = min(.8,1.8*(1 - grateS1)); d=-2;
     
    if d==1
    fid = fopen('/Users/maddy/Dropbox/rishicode_maddy070115/caret2/maddy badrate/Macaque.composite_june2012.areacolor','w');
    end
    if d==2
    fid = fopen('/Users/maddy/Dropbox/rishicode_maddy070115/caret2/maddy goodrate/Macaque.composite_june2012.areacolor','w');
    end
    if d==3
    fid = fopen('/Users/maddy/Dropbox/rishicode_maddy070115/caret2/maddy badspike/Macaque.composite_june2012.areacolor','w');
    end
    if d==4
    fid = fopen('/Users/maddy/Dropbox/rishicode_maddy070115/caret2/maddy goodspike/Macaque.composite_june2012.areacolor','w');
    end
    if d==5
    fid = fopen('/Users/maddy/Dropbox/rishicode_maddy070115/caret2/maddy badcons/Macaque.composite_june2012.areacolor','w');
    end
    if d==6
    fid = fopen('/Users/maddy/Dropbox/rishicode_maddy070115/caret2/maddy goodcons/Macaque.composite_june2012.areacolor','w');
    end
    if d==7
    fid = fopen('/Users/maddy/Dropbox/rishicode_maddy070115/caret2/maddy nofbgoodcons/Macaque.composite_june2012.areacolor','w');
    end
    if d==8
    fid = fopen('/Users/maddy/Dropbox/rishicode_maddy070115/caret2/maddy cons5/Macaque.composite_june2012.areacolor','w');
    end
    
    if d==-1
    fid = fopen('/Users/maddy/Dropbox/rishicode_maddy070115/caret2/maddy badrateS1/Macaque.composite_june2012.areacolor','w');
    end
    if d==-2
    fid = fopen('/Users/maddy/Dropbox/rishicode_maddy070115/caret2/maddy goodrateS1/Macaque.composite_june2012.areacolor','w');
    end
    
    areaList{16}='9_46v'; areaList{17}='9_46d';
    os1='   '; os2='      ';
    fprintf(fid,'%s<Area_Color_File>\n',os1);
    
    if d > 4
    list1rate = 1+[6,9,10,11,15,17,19,22,23,24,25,27,28];
           for j = 1:length(list1rate) %new. 
           logTau(list1rate(j)) = 0;
           end
    end
   
    
    for i=1:nNodes
        
%         if d > 4 %5,6,7 - only look at rate areas that propagate. 
%           %list1rate = [6,9,10,11,15,17,19,22,23,24,25,27,28];
%           if logTau(i) < 0 %> 1
%               logTau(i) = 0; % areas that show weak propagation. 
%           end
%         end
        if d==7
            if logTau(i) < 0 
                logTau(i) = 0;
            end
        end

        currCol=round(255*f(logTau(i),cmap));
        fprintf(fid,'%s<Color>\n',os1);
        fprintf(fid,'%s<name><![CDATA[%s_M132]]></name>\n',os2,areaList{i});
        fprintf(fid,'%s<red>%d</red>\n',os2,currCol(1));
        fprintf(fid,'%s<green>%d</green>\n',os2,currCol(2));
        fprintf(fid,'%s<blue>%d</blue>\n',os2,currCol(3));
        fprintf(fid,'%s<pointSize>2.0</pointSize>\n',os2);
        fprintf(fid,'%s<lineSize>1.0</lineSize>\n',os2);
        fprintf(fid,'%s<symbol><![CDATA[POINT]]></symbol>\n',os2);
        fprintf(fid,'%s<sumscolorid><![CDATA[]]></sumscolorid>\n',os2);
        fprintf(fid,'%s</Color>\n',os1);
    end;
    fprintf(fid,'%s</Area_Color_File>\n',os1);
    fclose(fid);

end

if caseval == 13
    cvosc(1,1:3) = [237/255, 177/255, 32/255]; cvosc(2,1:3) = [217/255 83/255 25/255];%color scheme.

   load oscillations.mat  
   hFig = figure();
   set(hFig,'units', 'centimeters', 'Position', [0 0 5 1],'PaperUnits','centimeters','PaperPosition',[.05 .05 5 1],'PaperSize',[5.1 1.1]) ;
   plot(time, r2,'-','LineWidth',lw,'MarkerSize',fsizelab,'Color',cvosc(1,:));
   set(gca,'xcolor',get(gcf,'color'),'ycolor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',[],'Xtick',[],'XLim',[2.7 3.6],'YLim',[-1 9]);

   saveas(hFig, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/main fig 3Aosc1', 'pdf')
   
   hFig = figure();
   set(hFig,'units', 'centimeters', 'Position', [0 0 5 1],'PaperUnits','centimeters','PaperPosition',[.05 .05 5 1],'PaperSize',[5.1 1.1]) ;
   plot(time, r5,'-','LineWidth',lw,'MarkerSize',fsizelab,'Color',cvosc(2,:));
   set(gca,'xcolor',get(gcf,'color'),'ycolor',get(gcf,'color'),'FontSize',fsizelab,'box','off','Ytick',[],'Xtick',[],'XLim',[2.7 3.6],'YLim',[-2.5 3]);
   hold on;plot(time(501:801),-1.5*ones(size(time(501:801))),'k');
   saveas(hFig, '/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/main fig 3Aosc2', 'pdf')
    
end

if caseval == 14

tParams.dt=2e-4; %new
tParams.nSteps=3e4;

tSteps=(1:tParams.nSteps)*tParams.dt;

%omegaEI = 19.7 ; muEE = 33.7 ; %weak gba
%omegaEI = 25.2 ; muEE = 51.5;  %strong gba


   %contour for  large scale system --- long range muEE and wEI 
   %paramvar1=31:.1:53; paramvar2=19:.1:26;
paramvar1=31:5:53; paramvar2=19:5:26;
ampE1_A = zeros(length(paramvar2),length(paramvar1));
ampElast_A = zeros(length(paramvar2),length(paramvar1));


    for tvar1=1:length(paramvar1)
     for tvar2=1:length(paramvar2)
    
        var1here=paramvar1(tvar1); var2here=paramvar2(tvar2);
        muEE=var1here; omegaEI=var2here;
             disp([muEE,omegaEI]);

        localParams.ee=betaE*omegaEE*(1+netwParams.alpha*netwParams.hier);%rsc*2.29119*(1+netwParams.alpha*netwParams.hier);
        localParams.ie=betaI*omegaIE*(1+netwParams.alpha*netwParams.hier);%rsc*6.09552*(1+netwParams.alpha*netwParams.hier);
        localParams.ei=-betaE*omegaEI; %-1.29954;
        localParams.ii=-betaI*omegaII; %-4.40204;        
        ldParams.ee=betaE*muEE*(1+netwParams.alpha*netwParams.hier);%rsc*2.29119*(1+netwParams.alpha*netwParams.hier);
        ldParams.ie=betaI*muIE*(1+netwParams.alpha*netwParams.hier);%rsc*6.09552*(1+netwParams.alpha*netwParams.hier);
        ldConns.ee= bsxfun(@times,flnMat,ldParams.ee);
        ldConns.ie= bsxfun(@times,flnMat,ldParams.ie);

        nNodes=size(ldConns.ee,1);
        wEe=diag(-1+localParams.ee)+ldConns.ee;
        wEi=localParams.ei*eye(nNodes);%these ws would be 29x29 identity matrices. 
        wIe=diag(localParams.ie)+ldConns.ie;
        wIi=(-1+localParams.ii)*eye(nNodes);%compare this to eqn 4 - where does -1 come from??
        %result=[wEe/localParams.tauE wEi/localParams.tauE;wIe/localParams.tauI wIi/localParams.tauI];
        A = [wEe/localParams.tauE wEi/localParams.tauE;...
            wIe/localParams.tauI wIi/localParams.tauI];

        currParams.A=A; 

        B=currParams.A;%B=params.A;
                B(1:nNodes,:)=B(1:nNodes,:)*localParams.tauE;
                B((nNodes+1):end,:)=B((nNodes+1):end,:)*localParams.tauI;
                currs=-B*currParams.desiredSs; %currs=-B*params.desiredSs;

        localParams.bgExc=currs(1:nNodes); localParams.bgInh=currs((nNodes+1):end);
        bgCurr=[localParams.bgExc; localParams.bgInh];%bg = background

        combMat=A;
        combMat(1:nNodes,:)=combMat(1:nNodes,:)*localParams.tauE;
        combMat((nNodes+1):end,:)=combMat((nNodes+1):end,:)*localParams.tauI;
        sstate=-combMat\bgCurr;%which is -bgCurr/combMat %this is steady state. 

        netwParams.initConds.exc=sstate(1:nNodes); netwParams.initConds.inh=sstate((nNodes+1):end);

        inpParams=struct('tStart',2,'tEnd',2.25,'ampl',22.05*1.9);
        
        areaInput.inp=zeros(tParams.nSteps,1);
        areaInput.inp(round(inpParams.tStart/tParams.dt):round(inpParams.tEnd/tParams.dt))=inpParams.ampl;

            popRates = runNetworkAreaInputNEW(tParams,localParams,netwParams,ldConns,areaInput,'threshLinear');
            %maxpeak=zeros(nNodes,1);
            %for r=1:nNodes
            %maxpeak(r,1)=max(popRates.exc(r,(toplotstart:toplotend))-popRates.exc(r,toplotstart));
            %end
    
    %ampratio(tvar2, tvar1) = (maxpeak(nNodes,1)/maxpeak(1,1));
   maxstate=zeros(nNodes,1); toggle=0; statecheck=[popRates.exc ; popRates.inh];
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%let oscillations be...remove below code for now to

  if((max(statecheck(nNodes,:)) > statecheck(nNodes,tParams.nSteps)) )
    ampE1_A(tvar2,tvar1) = max(statecheck(1,:)-statecheck(1,1));
    ampElast_A(tvar2,tvar1) = max(statecheck(nNodes,:)-statecheck(nNodes,1));
  end
  
     if((max(statecheck(nNodes,:)) == statecheck(nNodes,tParams.nSteps)) || (max(statecheck(nNodes,:))>1e8))%firing rate beyond max
        ampE1_A(tvar2,tvar1) = 1; ampElast_A(tvar2,tvar1) = 1;
     end
     
     end
    end
    
     %save('big_sys_ampE1_A','ampE1_A'); save('big_sys_ampElast_A','ampElast_A');
     ampratio = (ampElast_A./ampE1_A);
     %save('big_sys_ampratio','ampratio');
    load big_sys_ampratio; 
    

    paramvar1=31:.1:53; paramvar2=19:.1:26;
    hFig = figure(); fsizelab = 7; fsize=7;mksize = 4;
    set(hFig,'units', 'centimeters', 'Position', [0 0 7. 4.],'PaperUnits','centimeters','PaperPosition',[.1 .1 7. 4.],'PaperSize',[7 4.3]) 
    %for i = 1:71; for j = 1:221; if ampratio(i,j) ==1;  ampratio(i,j)=1e-7; end;end;end; %revision added.    
    ax=gca; pos=get(gca,'pos');
    [C,h] = contourf(paramvar1,paramvar2,log10(ampratio),20);%,'Position',[pos(1)+pos(3)+.1 pos(2) .1 pos(4)]);    
    xlabel('Global E to E coupling','FontSize', fsize); ylabel('Local I to E coupling','FontSize', fsize);
    annotation('textbox', [0.4,0.35,0.1,0.1],'Color','k','String', 'unstable','FontSize',fsizelab,'LineStyle','None');
    shading flat;set(h,'LineColor','none');h = colorbar;ylabel(h, 'Propagation ratio','FontSize',fsizelab);
    %set(h,'Ticks',[0,4,8]);%colorbar('Ticks',[0,1,2,3]);  
    x1=get(gca,'position');x=get(h,'Position');x(3)=0.03;set(h,'Position',x);set(gca,'position',x1);
    set(gca,'FontSize',fsizelab);

    omegaEIbefore = 19.7 ; muEEbefore = 33.7 ; omegaEIafter = 25.2 ; muEEafter = 51.5;

    hold on; plot(muEEbefore, omegaEIbefore, 'x','LineWidth',.1,'MarkerEdgeColor',cv(1,:),'MarkerFaceColor',cv(1,:),'Markersize',mksize-2);
    hold on;plot(muEEafter,omegaEIafter,'x','LineWidth',.1,'MarkerEdgeColor',cv(2,:),'MarkerFaceColor',cv(2,:),'Markersize',mksize-2);
    %caxis([0 9.5]);
    colormap(hot); %revision added.   
    %saveas(hFig,'/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/suppcontourbigmat.pdf'); 
    
end


if caseval == 15
   mfrange = 0:.05:.2; 
   %mfrange = 0:.005:.2; 
   
   propratbad = zeros(length(mfrange),1);
   propratgood = zeros(length(mfrange),1);
   
   for vary = 1:length(mfrange)
       mf = mfrange(vary); 
       
       for count=1:2
        if count==1
            omegaEI = 19.7 ; muEE = 33.7 ; 
        else
            omegaEI = 25.2 ; muEE = 51.5;
        end
        
        localParams.ee=betaE*omegaEE*(1+netwParams.alpha*netwParams.hier);
        localParams.ie=betaI*omegaIE*(1+netwParams.alpha*netwParams.hier);
        localParams.ei=-betaE*omegaEI;
        localParams.ii=-betaI*omegaII;         
             
        ldParams.ee=betaE*muEE*(1+netwParams.alpha*netwParams.hier);
        ldParams.ie=betaI*muIE*(1+netwParams.alpha*netwParams.hier);    
       
        ldConns.ee= bsxfun(@times,flnMat,ldParams.ee);
        ldConns.ie= bsxfun(@times,flnMat,ldParams.ie);

        
            burkmat = zeros(nNodes, nNodes);
            %mf = .2;  %vary mf. 
            %burkhalter like scaling of long range E to I. 
            for r = 1:nNodes 
                for s = 1:nNodes
                    burkmat(r,s) = max(0,netwParams.hier(r,1) - netwParams.hier(s,1));
                    %burkmat(r,s) = netwParams.hier(r,1) - netwParams.hier(s,1);
                end ;
            end;
            burkmat = 1 + mf*burkmat;
            ldConns.ie = bsxfun(@times,ldConns.ie,burkmat);
        
        
        nNodes=size(ldConns.ee,1);
        wEe=diag(-1+localParams.ee)+ldConns.ee;
        wEi=localParams.ei*eye(nNodes);%these ws would be 29x29 identity matrices. 
        wIe=diag(localParams.ie)+ldConns.ie;
        wIi=(-1+localParams.ii)*eye(nNodes);
        A = [wEe/localParams.tauE wEi/localParams.tauE;...
            wIe/localParams.tauI wIi/localParams.tauI];

        currParams.A=A; 

        B=currParams.A;%B=params.A;
                B(1:nNodes,:)=B(1:nNodes,:)*localParams.tauE;
                B((nNodes+1):end,:)=B((nNodes+1):end,:)*localParams.tauI;
                currs=-B*currParams.desiredSs; 

        localParams.bgExc=currs(1:nNodes); localParams.bgInh=currs((nNodes+1):end);
        bgCurr=[localParams.bgExc; localParams.bgInh];%bg = background

        combMat=A;
        combMat(1:nNodes,:)=combMat(1:nNodes,:)*localParams.tauE;
        combMat((nNodes+1):end,:)=combMat((nNodes+1):end,:)*localParams.tauI;
        sstate=-combMat\bgCurr;

        netwParams.initConds.exc=sstate(1:nNodes); netwParams.initConds.inh=sstate((nNodes+1):end);

        inpParams=struct('tStart',2,'tEnd',2.25,'ampl',22.05*1.9);
        
        areaInput.inp=zeros(tParams.nSteps,1);
        areaInput.inp(round(inpParams.tStart/tParams.dt):round(inpParams.tEnd/tParams.dt))=inpParams.ampl;

        if (count == 1)
            popRatesbad=runNetworkAreaInputNEW(tParams,localParams,netwParams,ldConns,areaInput,'threshLinear');
            maxpeakbad=zeros(nNodes,1);
            for r=1:nNodes
            maxpeakbad(r,1)=max(popRatesbad.exc(r,(toplotstart:toplotend))-popRatesbad.exc(r,toplotstart));
            end
        else
            popRatesgood=runNetworkAreaInputNEW(tParams,localParams,netwParams,ldConns,areaInput,'threshLinear');
            maxpeakgood=zeros(nNodes,1);
            for r=1:nNodes
            maxpeakgood(r,1)=max(popRatesgood.exc(r,(toplotstart:toplotend))-popRatesgood.exc(r,toplotstart));
            end
        end
    
       end
       
       propratbad(vary,1) = maxpeakbad(nNodes,1)/maxpeakbad(1,1);
       propratgood(vary,1) = maxpeakgood(nNodes,1)/maxpeakgood(1,1);
       

   end
  

    %mfrange = 0:.005:.2;
    %load propratgood; load propratbad;
    hFig = figure();  
    set(hFig,'units', 'centimeters', 'Position', [0 0 7.5 4.5],'PaperUnits','centimeters','PaperPosition',[.05 .05 7.5 4.5],'PaperSize',[8.6 4.4]) ;
    semilogy(mfrange,propratbad,'-','LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(1,:));
    hold on;semilogy(mfrange,propratgood,'-','LineWidth',lw,'MarkerSize',fsizelab,'Color',cv(2,:));
    legend({'weak GBA','strong GBA'},'Position',[0.68 0.78 0.1 0.1],'Box','off','FontSize',fsize); 
    %set(gca,'Ytick',logspace(-7,-1,3),'YTickLabel',['10^{-7}';'10^{-4}';'10^{2} '],'ylim',[1e-4 1e2]);set(gca,'FontSize',fsizelab,'box','off');
    set(gca,'Ytick',logspace(-7,-1,3),'YTickLabel',['10^{-7}';'10^{-4}';'10^{-1}'],'ylim',[1e-7 1e-1],'box','off');
    xlabel('Hierarchical inhibitory tuning','FontSize',fsize);
    set(gca,'FontSize',fsizelab);
    %set(gca,'Xtick',0:1:28,'box','off');
    h = ylabel('Propagation ratio','FontSize',fsize);%set(h, 'Units', 'Normalized');pos = get(h, 'Position');set(h, 'Position', pos + [-.02, 0., -0.9]);%
    %saveas(hFig,'/Users/maddy/Dropbox/rishicode_maddy070115/signal propagation paper/paper figures v2/suppburkhalter.pdf'); 
 
end