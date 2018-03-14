
function [GPUout]=SHOPhaseGPUfit(freqq,Xhat2,init)

%developed by Ehsan Nasr Esfahani ehsann@uw.edu Jan. 3rd 2018- University
%of Washington Seattle USA

% This function fits SHO phase 
%Input:
    %freqq: freqq is the indipendent variable. Usually frequency of excitation
    %Xhat2: full matrix of phase data. each column should be the data for each fit. size [number_points, number_fits] and data type single
    %init: [optional] if given, it is used for omega_0 and Quality factor initial guess  
 	
  
    
%output 
% GPUout: Output parameters including
    % parameters:	Fitted parameters for each fit 2D matrix of size: [number_parameter, number_fits] of data type single
        % parameters(1)= omega_0 (resonance frequency), parameters(2)=
        % qaulity factor, parameters(3)= phi_0 (intrinsic phase response)
            % Note: Initial parameters 2D matrix of size: [number_parameter, number_fits]
    % states:	Fit result states for each fit vector of length number_parameter of data type int32 As defined in constants.h:
    % chi_squares:	?2?2 values for each fit vector of length number_parameter of data type single
    % n_iterations:	Number of iterations done for each fit vector of length number_parameter of data type int32
    % time:	Execution time of call to gpufit In seconds.


 if ~exist('init','var')
     q1=100;
     % third parameter does not exist, so default it to something
       aa=repmat(mean(freqq),[1,size(Xhat2,2)]); %initial guess for omega0
       bb=repmat(q1,[1,size(Xhat2,2)]); %initial guess for quality factor
 else %if initial is given
     aa=init(1,:); %initial guess for omega_0 (resonance freq)
     bb=init(2,:); %initial guess for Quality factor
 end

Xhat2=single(Xhat2); %initall guess for phi_0 is zero! 

cc=zeros(1,size(Xhat2,2)); %initial quess for phi_0

initials=single([aa;bb;cc]); %should be converted to single for GPU

%fit parameters (relative tolerance and max number of iterrations)
tolerance=single(1e-10);
max_n_iterations=100;

% estimator id
estimator_id = EstimatorID.LSE;
% model ID
model_id = ModelID.SHO_Phase ;
%parameters_to_fit:
tofit=[1 ;1; 1]; % A zero indicates that this parameter should not be fitted, everything else means it should be fitted.

%GPU-based fitting. u will be amazed!
[GPUout.parameters, GPUout.states, GPUout.chi_squares, GPUout.n_iterations, GPUout.time] = gpufit(Xhat2, [],...
    model_id, initials, tolerance, max_n_iterations, tofit, estimator_id, single(freqq'));
end