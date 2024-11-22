% Run simulations of the network with input to a particular area and noise

function result=runNetworkAreaInputNEW(tStruct,localParams,netwParams,ldConns,areaInput,fiCurveType)

nNodes=size(ldConns.ee,1);

state.exc=zeros(nNodes,1); state.inh=zeros(size(state.exc));

if(isfield(netwParams,'initConds'))
    state.exc=netwParams.initConds.exc; state.inh=netwParams.initConds.inh;
end;

%  recording rates across areas
popRates.exc=zeros(nNodes,tStruct.nSteps); 
%29 rows corresponding to 29 nodes..for each row see progress of state over
%time. corresponding to columns. 
popRates.inh=zeros(size(popRates.exc));
popRates.exc(:,1)=state.exc; popRates.inh(:,1)=state.inh;
%setting initial states. 

% Define f-I curve

switch fiCurveType
    case 'threshLinear'
        tmpZeros=zeros(size(state.exc));
        excFI=@(x) max(x,tmpZeros);
        inhFI=@(x) max(x,tmpZeros);
        fprintf('Threshold linear f-I curve');
        excFInew=@(x) max(x,10*ones(29,1));
        inhFInew=@(x) max(x,35*ones(29,1));
        
end;

% Noise std can vary between areas.
noiseStd=localParams.noiseStd/sqrt(tStruct.dt);

format long; toy=0;

for i=2:tStruct.nSteps

    if(areaInput.inp(i)==0 && toy==0)
        state.exc = 10*ones(nNodes,1);state.inh = 35*ones(nNodes,1);
    else    
        toy=1;

    state.excI=ldConns.ee*state.exc+localParams.ee.*state.exc+...
        localParams.ei.*state.inh+localParams.bgExc+noiseStd.*randn(nNodes,1);
    state.inhI=ldConns.ie*state.exc+localParams.ie.*state.exc+...
        localParams.ii.*state.inh+localParams.bgInh+noiseStd.*randn(nNodes,1);

    % Add in the area specific input (single area, exc population)
    state.excI(areaInput.idx)=state.excI(areaInput.idx)+areaInput.inp(i);
    % Now update the state using these currents

   state.exc = excFInew(state.exc+(tStruct.dt/localParams.tauE).*(-state.exc+excFI(state.excI)));
   state.inh = inhFInew(state.inh+(tStruct.dt/localParams.tauI)*(-state.inh+inhFI(state.inhI)));
   
   if(min(min(state.exc))<10)
         disp(i)
   end
     
    end
    % And record
    popRates.exc(:,i)=state.exc; popRates.inh(:,i)=state.inh;
    %this gives firing rates filled columnwise (timewise) 
end;

result=popRates;


