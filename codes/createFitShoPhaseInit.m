function [fitresult, gof] = createFitShoPhaseInit(freq, phase,Qint, Finit)
%CREATEFIT(FREQ,PHASE)
%  Create a fit.
%
%  Data for 'untitled fit 1' fit:
%      X Input : freq
%      Y Output: phase
%  Output:
%      fitresult : a fit object representing the fit.
%      gof : structure with goodness-of fit info.
%
%  See also FIT, CFIT, SFIT.

%  Auto-generated by MATLAB on 17-Aug-2016 16:57:30


%% Fit: 'untitled fit 1'.
[xData, yData] = prepareCurveData( freq, phase);
Phsint=mean(phase); %intial guess for phase offset
% Set up fittype and options.
ft = fittype( '(atan2(freq.*freq0./Q,freq0^2-freq.^2)+offset)', 'independent', 'freq', 'dependent', 'phase' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
opts.Lower = [10 min(freq) -2*pi];
opts.upper = [700 max(freq)   2*pi];
opts.MaxIter = 1000;
% opts.Robust = 'LAR';
opts.StartPoint = [Qint Finit Phsint];

% Fit model to data.
[fitresult, gof] = fit( xData, yData, ft, opts );


