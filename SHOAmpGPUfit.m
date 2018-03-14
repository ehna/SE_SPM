function [GPUout]=SHOAmpGPUfit(freqq,Xhat2)


%developed by Ehsan Nasr Esfahani ehsann@uw.edu Jan. 3rd 2018- University
%of Washington Seattle USA

% This function fits SHO Amplitude 
%Input:
    %freqq: freqq is the indipendent variable. Usually frequency of excitation
    %Xhat2: full matrix of Amplitude data. each column should be the data for each fit. size [number_points, number_fits] and data type single
 	
  
    
%output 
% GPUout: Output parameters including
    % parameters:	Fitted parameters for each fit 2D matrix of size: [number_parameter, number_fits] of data type single
        % parameters(1)= omega_0 (resonance frequency), parameters(2)=
        % qaulity factor, parameters(3)= Amp_0 (intrinsic amplitude response)
    % states:	Fit result states for each fit vector of length number_parameter of data type int32 As defined in constants.h:
    % chi_squares:	?2?2 values for each fit vector of length number_parameter of data type single
    % n_iterations:	Number of iterations done for each fit vector of length number_parameter of data type int32
    % time:	Execution time of call to gpufit In seconds.



Xhat2=single(Xhat2);

q1=70;
[maxx,idma]=max(Xhat2) ;
aa=freqq(idma);
% aa=repmat(mean(freqq),[1,size(Xhat2,2)]); %initial guess for omega0
bb=repmat(q1,[1,size(Xhat2,2)]); %initial guess for quality factor
cc=maxx./q1; %initial guess for intrinsic amplitude
dd=repmat(mean(min(Xhat2)),[1,size(Xhat2,2)]);%initial guess for noise level
% dd=repmat(0,[1,size(Xhat2,2)]);%initial guess for noise level

initials=([aa;bb;cc;dd]);

% initials=single(zeros(4,size(Xhat2,2) ))*0.001;
tolerance=single(1e-30);
max_n_iterations=3000;

% estimator id
estimator_id = EstimatorID.LSE;
% model ID
model_id = ModelID.SHO_Amp ;


tofit=[1 ;1; 1; 0]; %the DC is not fitting
%GPU-based fitting. u will be amazed!
[GPUout.parameters, GPUout.states, GPUout.chi_squares, GPUout.n_iterations, GPUout.time] = gpufit(Xhat2, [],...
    model_id, initials, tolerance, max_n_iterations, tofit, estimator_id, single(freqq'));
% parameters(1,:)=parameters(1,:)./delta; %converting slop to [pm/V]
% parameters(2,:)=min(freqq)+parameters(2,:).*delta; %converting Vcpd to V]
end