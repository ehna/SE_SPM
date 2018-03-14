function [fitresult,gof] = createFitSHOAmpInitiDC(freq,amp,Qinit,Finit)

%% Fit: 'untitled fit 1'.
[xData, yData] = prepareCurveData( freq, amp );


% [pks,locs]=findpeaks(amp,'NPeaks',1,'SortStr','descend');
pks=max(amp,[],'omitnan');
% Qinit=100; %let's assume Q initaial=100;
Ampp0=pks/Qinit;




% Set up fittype and options.

ft = fittype( 'A0*freq0^2/sqrt((freq0^2-freq^2)^2+(freq*freq0/Q)^2)', 'independent', 'freq', 'dependent', 'amp' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
% opts.Robust = 'LAR';
opts.Lower = [0  10 min(freq) ];
opts.Upper = [pks*1.1  1000 max(freq) ];
opts.StartPoint = [Ampp0  Qinit Finit ];


% Fit model to data.
[fitresult, gof] = fit( xData, yData, ft, opts );


