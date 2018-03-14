%Ehsan Nasr Esfahani, University of Washington, Seattle, USA
% September 28th, 2017 ehsann@uw.edu

%This code takes numbers of AR single frequency PFM images from (igor IBW
%files) and performs PCA cleaning, drift compensation based on topography,
%and SHO fiting to each pixel of image. You have the option to select CPU
%or GPU fitting. GPU fitting is based on KUDA and is adopted from:
% http://gpufit.readthedocs.io/en/latest/index.html

% It will plot few PCA loadings and SHO amplitude, frequency and Quality
% factor mappinsg.

%Make sure the follwoing files are in the same folder or added to MATLAB directoru
%necessary .m files to run this:
%1-IBWread.m,2-dftregistration.m,3-createFitSHOAmpInitiDC.m,4-createFitShoPhaseInit.m,
%5-caxiscorrection.m,  6-caxiscorrPhase.m , 7-readIBWbinheader.m,
%8-readIBWheaders.m, 9-readIBWbinheader.m, 10-freezColors.m,
%11-SHOAmpGPUfit.m , 12-SHOPhaseGPUfit.m


% for GPU-fit, please place the "Gpufit-build-64" directory in specific
% location and the following path to your MATLAB paths using "set path" : ...\Gpufit-build-64\Debug\matlab
clear all;close all;clc;

%% loading data
%read the directory
disp('Select  main folder your IBW files (AR sPFM images) are located in')
folder_name = uigetdir;
cd( folder_name)

File=uigetfile('*.ibw', 'Select all IBW files you want to perform PCA on ','MultiSelect','on');

prompt = 'Do you want to do image drift compensation? 1=Yes, 0=No ';
n0 = input(prompt);
if n0==1
    prompt = 'Starting from 1, what is the Image #layer number you want to use for drift correction? (either Deflection or Height)';
    n1 = input(prompt);
end
prompt = 'Image #layer number for the single Frequency Amp ';
n2 = input(prompt);
prompt = 'Image #layer number for the single Frequency Phase ';
n4 = input(prompt);

prompt = 'CPU or GPU fitting? CPU=1, GPU=2';
n5 = input(prompt);
%% image registration

tic
for ii1=1:length(File) %number of selected files
    %converts IBW files to mat file. this requeres the file IBWread
    dataR=IBWread(cell2mat(File(ii1)));
    fclose('all'); %close the ibw file
    %this part is to obtain the frequency of drive in sPFM
    aa=strfind(dataR.WaveNotes,'DriveFrequency'); %find the string for drive frequency in notes and its position
    dum=dataR.WaveNotes(aa(1)+15:aa(1)+27);
    dum2=strfind(dum, 'S');
    freqq0(ii1)=str2num(dum(1,1:dum2-1)) ; %this vector will contain all the sPFM frequencies for all images
    
    
    
    defl=dataR.y(:,:,n1); %deflection image
    defl=defl-mean2(defl); %sbtract the mean, center around zero (assumption: guasian noise)
    amp=dataR.y(:,:,n2); %sPFM amplitude image
    phs=dataR.y(:,:,n4); %sPFM phase image
    
    %if image registry is on
    if n0==1
        %takes the the first image as the benchmark for image compensation
        if ii1==1
            deflinit=defl; %refference image for registry
        end
        %cross-correlation for aligning deflection images (image registry)
        crr(ii1,:) = dftregistration(fft2(deflinit),fft2(defl),1);
        %         disp(crr(ii1,:))%detals of registry (error,0, translat x,translate y)
        %translate the delfection   (registred)
        deflT=imtranslate(defl,[crr(ii1,4) crr(ii1,3)],'FillValues',NaN);
        ampT=imtranslate(amp,[crr(ii1,4) crr(ii1,3)],'FillValues',NaN); %translate the amplitude   (registred)
        phsT=imtranslate(phs,[crr(ii1,4) crr(ii1,3)],'FillValues',NaN); %translate the phase   (registred)
    else
        deflT=defl;
        ampT=amp;
        phsT=phs;
    end
    % stack all deflection data in one matrix
    MatPhs0(:,ii1)=reshape(phsT,[],1); % stack all sPFM phase data in one matrix
    MatDefl0(:,ii1)=reshape(deflT,[],1); % stack all sPFM deflection data in one matrix
    MatDeflUn0(:,ii1)=reshape(defl,[],1); % stack all sPFM deflection(unregistred) data in one matrix
    MatAmp0(:,ii1)=reshape(ampT.*1e12,[],1); % stack all sPFM amplitude data in one matrix
    
end
toc




%% sort data based on frequency (decesnding)

[freqq, freqindex]=sort(freqq0);

MatPhs=MatPhs0(:,freqindex);
MatAmp=MatAmp0(:,freqindex);
MatDefl=MatDefl0(:,freqindex);
MatDeflUn=MatDeflUn0(:,freqindex);
h1=figure('units','normalized','outerposition',[0 0 .45 .8]);
F(length(File)) = struct('cdata',[],'colormap',[]);

for ii1=1:length(File)
    %movie type, check if registy works well
    subplot(221)
    imagesc(reshape(MatDeflUn(:,ii1),size(defl)));axis square;title('Original');axis off;colormap('bone')	 ;freezeColors;view(-90,90);%plot original deflection;
    set(gca,'color','none')
    subplot(222)
    imagesc(reshape(MatDefl(:,ii1),size(defl)));axis square;title('Registred Defl.');axis off;colormap('bone');	freezeColors;view(-90,90) %plot registered
    set(gca,'color','none')
    subplot(223)
    imagesc(reshape(MatAmp(:,ii1),size(defl)));axis square;axis off ;colormap('default');caxiscorrection(MatAmp(:,ii1));freezeColors;view(-90,90)
    title(strcat('Registre Amp.- Freq=',num2str(freqq(ii1))));	 %plot registered
    set(gca,'color','none')
    subplot(224)
    imagesc(reshape(MatPhs(:,ii1),size(defl)));axis square;axis off ;colormap('copper') ;caxiscorrection(MatPhs(:,ii1));freezeColors;view(-90,90);
    title('Registred Phase')
    set(gca,'color','none')
    %     pause(.01)
    F(ii1) = getframe(h1) ;
end
%make animation
[h, w, p] = size(F(1).cdata);  % use 1st frame to get dimensions
hf = figure;
% resize figure based on frame's w x h, and place at (150, 150)
set(hf, 'position', [150 150 w h])
axis off
%let us make an animation
movie(hf,F);
v = VideoWriter('AnimSE','Motion JPEG AVI');
v.Quality=100;
v.FrameRate = 2;
open(v)
writeVideo(v,F)
close(v)


%% PCA
% nmod=1;

[coeff,score,latent] = pca(MatAmp,'Centered',false) ; %perfroming PCA on Amplitude dataset without centering data
[coeffP,scoreP,latentP] = pca(MatPhs,'Centered',false) ; %perfroming PCA on phase data set without centering data

figure;
% subplot(131)
semilogy(latent,'o');xlabel('#PCs');ylabel('PC Varience');title('scree plot')

figure('units','normalized','outerposition',[0 0 1 1])
for ii2=1:size(latent,1)
    subplot(131);loglog(latent,'o','MarkerFaceColor','b' );xlabel('#PCs');ylabel('PC Varience');title('scree plot of Amplitude data')
    subplot(132);plot(freqq,coeff(:,ii2),'--o');title(strcat('EigenVector#',num2str(ii2)))
    subplot(133);imagesc(((reshape(score(:,ii2),size(defl)))));axis square;axis off;view(-90,90);title('PC loading map')
    caxiscorrPhase(score(:,ii2))
    pause(2)
end

figure('units','normalized','outerposition',[0 0 1 1]);title('Phase data')
for ii2=1:size(latent,1)
    subplot(131);loglog(latentP,'o','MarkerFaceColor','b' );xlabel('#PCs');ylabel('PC Varience');title('scree plot of Phase data')
    subplot(132);plot(freqq,coeffP(:,ii2),'--o');title(strcat('EigenVector#',num2str(ii2)))
    subplot(133);imagesc(((reshape(scoreP(:,ii2),size(defl)))));axis square;axis off;view(-90,90);title('PC loading map')
    caxiscorrPhase(scoreP(:,ii2))
    pause(2)
end

figure
plot(freqq.*1e-3,coeff(:,1),'bo','MarkerSize',4 , 'MarkerFaceColor','b' );hold on
plot(freqq.*1e-3,coeff(:,2),'--gd','MarkerSize',4,'MarkerEdgeColor','none' ,'MarkerFaceColor','g','LineWidth',1.2);
plot(freqq.*1e-3,coeff(:,3),'-.rs','MarkerSize',4,'MarkerFaceColor','r','MarkerEdgeColor','none','LineWidth',1.2);
legend('EV1','EV2','EV3','Location',   'northwest' )
xlabel('Freq. \omega_j [kHz]')
ylabel('PC Coefficient')
axis tight
legend boxoff
set(gca,'color','none')
title('The first 3 PCA Eigenvectors (Amplitude)')
figure;
scatter(freqq,coeffP(:,1),80,'filled');title(strcat('EigenVector#',num2str(1)));hold all
scatter(freqq,coeffP(:,2),80,'filled');title(strcat('EigenVector#',num2str(2)))
scatter(freqq,coeffP(:,3),80,'filled');title(strcat('EigenVector#',num2str(3)))
legend('EV1','EV2','EV3')
xlabel('Freq. [kHz]')
PC.loading1=reshape(scoreP(:,1),size(defl));
PC.loading2=reshape(scoreP(:,2),size(defl));
PC.loading3=reshape(scoreP(:,3),size(defl));
title('The first 3 PCA Eigenvectors (Phase)')

figure;imagesc(PC.loading1);view(-90,90);
caxiscorrPhase (PC.loading1);axis off;axis square;colorbar;title('PCA Loading#1-Amp.')
figure;imagesc(PC.loading2);view(-90,90);
caxiscorrPhase(PC.loading2);axis off;axis square;colorbar;title('PCA Loading#2-Amp.')
figure;imagesc(PC.loading3);view(-90,90);
caxiscorrPhase(PC.loading3);axis off;axis square;colorbar;title('PCA Loading#3- Amp.')


%% PCA reconstrcution (cleaning),i.e. PCA dimensional reduction
prompt='how many PCA mode would you like to use (look at scree plot!)';
n3=input(prompt);
Xhat=score(:,1:n3)*coeff(:,1:n3)'; %reconstructred data
XhatP=scoreP(:,1:n3)*coeffP(:,1:n3)'; %reconstructred data

% converts NaN (from image registry) to empty (reduce image pixel size)
Xhat2=Xhat;
Xhat2(any(isnan(Xhat2),2),:)=[];

XhatP2=XhatP;
XhatP2(any(isnan(XhatP2),2),:)=[];

%to find real pixel size
AA=reshape(Xhat(:,1),size(defl));
out = AA(:,any(~isnan(AA)));  % for columns
out = out(any(~isnan(out),2),:);   %for rows
sizereduced=size(out);

MatDefl2=MatDefl;
MatDefl2(any(isnan(MatDefl2),2),:)=[];



Xhat2=Xhat2';
XhatP2=deg2rad(XhatP2');
MatDefl2=MatDefl2';



%% CPU SHO fitting
if n5==1
    
    
    % let's fit SHO to all reconstructed data.
    %the good thing is the fit initial values are very well known from the
    %previous fitting,i.e. dum1,dum2,dum3,dum4
      
    %Fist let's fit SHO to phase signal
    tic
    parfor ii1=1:size(XhatP2,2)
        %let's find the location of peak from Amplitude data
        [~,locs]=findpeaks(Xhat2(:,ii1),'NPeaks',1,'SortStr','descend');
        locs(isempty(locs))=1;
        %phase values after resonance are usually very unstable, we fit the
        %data on attractive regiem only (prior to resonance)
        kk1=1:min(locs+2,length(freqq));
        [FitRP(ii1).res,FitRP(ii1).res2]=createFitShoPhaseInit( freqq(kk1), (deg2rad( XhatP2(kk1,ii1) )'),70,freqq(locs));
    end
    toc
    
    
    
    for ii1=1:size(Xhat2,2)
        Visu.PQ1(ii1)=(FitRP(ii1).res.Q);
        Visu.PP(ii1)=(FitRP(ii1).res.offset);
        Visu.Pfreq0(ii1)=(FitRP(ii1).res.freq0);
        Visu.PR2(ii1)=       FitRP(ii1).res2.rsquare;
    end
    
    
    
    % let's  fit  amplitude SHO to each pixel from all images from the
    % reconstructred data, using initial condition from phase SHO map
    tic
    parfor ii1=1:size(Xhat2,2)
        [FitR(ii1).res,FitR(ii1).res2]= createFitSHOAmpInitiDC0(freqq,Xhat2(:,ii1)',Visu.PQ1(ii1),  Visu.Pfreq0(ii1));
        
    end
    toc
    
    for ii1=1:size(Xhat2,2)
        Visu.Q1(ii1)=(FitR(ii1).res.Q);
        Visu.A0(ii1)=(FitR(ii1).res.A0);
        Visu.freq0(ii1)=(FitR(ii1).res.freq0);
        Visu.R2(ii1)=       FitR(ii1).res2.rsquare;
        Visu.adjR2(ii1)=       FitR(ii1).res2.adjrsquare;
    end
     
    
    %% Visualization of fitings  and savings
    
    %for visualization purposes, we replace any r2 less than zero with nan
    Visu.R2(Visu.R2<0)=nan;
    Visu.PR2(Visu.PR2<0)=nan;
    
    %here we reshape thevectors to a matrix for visuzation , we also flip data
    %to match with its AR represenation
        % fitting parametters from SHO Amplitude-fitting

    Visu.Qimag=reshape(Visu.Q1,sizereduced);
    Visu.A0imag=reshape(Visu.A0,sizereduced);
    Visu.freq0imag=reshape(Visu.freq0,sizereduced);
    Visu.r2imag=reshape(Visu.R2,sizereduced);
    Visu.adjr2imag=reshape(Visu.adjR2,sizereduced);
    
    
    figure;imagesc(Visu.A0imag);axis    square ;colorbar;view(-90,90);caxiscorrection(Visu.A0imag);title('A_{0}');axis off
    figure;imagesc(Visu.Qimag);axis    square ;colorbar;view(-90,90);title('Q');caxiscorrection(Visu.Qimag);axis off
    figure;imagesc(Visu.freq0imag);axis    square ;colorbar;view(-90,90);title('f_{0}');caxiscorrection(Visu.freq0imag);axis off
    figure;imagesc(Visu.r2imag);axis    square ;colorbar;caxis([0.0 1]);view(-90,90);title('R^2');caxiscorrection(Visu.r2imag);axis off
    
    % fitting parametters from SHO phase-fitting
    Visu.PQimag=reshape(Visu.PQ1,size(defl));
    Visu.PPimag=reshape(rad2deg(Visu.PP),size(defl));
    Visu.Pfreq0imag=reshape(Visu.Pfreq0,size(defl));
    Visu.Pr2imag=reshape(Visu.PR2,size(defl));
    
    
    figure;imagesc( Visu.PPimag);axis square ;colorbar;view(-90,90);caxiscorrPhase(Visu.PPimag);title('Phase');axis off
    figure;imagesc(Visu.PQimag);axis    square ;colorbar;view(-90,90);title('PHASE Q_p');caxiscorrection(Visu.PQimag);axis off
    figure;imagesc(Visu.Pfreq0imag);axis    square ;colorbar;view(-90,90);title('f_{0 P}');caxiscorrection(Visu.Pfreq0imag);axis off
    figure;imagesc(Visu.Pr2imag);axis    square ;colorbar;caxis([0.0 1]);view(-90,90);title('PHASE R^2');caxiscorrection(Visu.Pr2imag);axis off
    
    
    
    save(strcat(File{1,1}(1:end-3),'.mat'),'Visu') %save raw data of all post-processing SHO images
elseif n5==2
    %% GPU SHO FIT
    
    %%PCA reconstrcution (cleaning),i.e. PCA dimensional reduction
    % prompt='how many PCA mode would you like to use (look at scree plot!)';
    

    
    % MatTopo2=MatTopo;
    % MatTopo2(any(isnan(MatTopo2),2),:)=[];
    % MatTopo2=MatTopo2';
    
    %SHO amp data fit
    [GPUout]=SHOAmpGPUfit(freqq,Xhat2);
    
    %calculate R^2
    for ii1=1:size(Xhat2,2)
        GPUout.RSS(ii1)= sumsqr(Xhat2(:,ii1)-mean(Xhat2(:,ii1))); %residual sum of squares
        GPUout.R2(ii1)=1-GPUout.chi_squares(ii1)/GPUout.RSS(ii1);
    end
    
    % SHO phase data fit
    [GPUoutP]=SHOPhaseGPUfit(freqq,( XhatP2) );
    
    %calculate R^2
    for ii1=1:size(XhatP2,2)
        GPUoutP.RSS(ii1)= sumsqr(XhatP2(:,ii1)-mean(XhatP2(:,ii1))); %residual sum of squares
        GPUoutP.R2(ii1)=1-GPUoutP.chi_squares(ii1)/GPUoutP.RSS(ii1);
    end
    %% pointwise test (randomly draws 30 plots of fitting to make sure u r not dumb or fucked!)
    DCB=linspace(min(freqq),max(freqq),300); %readl X axis, fine increments (used for visualizations only)
    figure
    for ii1=1:50
        clf
        iir=randi(size(Xhat2,2));
        yy=Xhat2(:,iir);
        plot(freqq,yy,'o');hold all
        yyf=GPUout.parameters(1,iir)^2*GPUout.parameters(3,iir)./sqrt((GPUout.parameters(1,iir)^2-DCB.^2).^2+(DCB.*GPUout.parameters(1,iir)./GPUout.parameters(2,iir)).^2)+GPUout.parameters(4,iir);
        plot(DCB,yyf)
        dim = [0.2 0.5 0.3 0.3];
        str = strcat('Q=',num2str(GPUout.parameters(2,iir),3), ', \omega_0=',num2str(GPUout.parameters(1,iir),4),'[kHz], A_0=',num2str(num2str(GPUout.parameters(3,iir),4)));
        annotation('textbox',dim,'String',str,'FitBoxToText','on','LineStyle','none' );
        xlabel('DC bias')
        ylabel('Deflection Amp. [pm]')
        title('point-wise fitting')
        set(gca,'color','none')
        yyaxis right
        yyp=atan2(DCB.*GPUoutP.parameters(1,iir),GPUoutP.parameters(2,iir).*(GPUoutP.parameters(1,iir)^2-DCB.^2))+GPUoutP.parameters(3,iir);
        plot(freqq,XhatP2(:,iir),'*');ylabel('Phase [rad]')
        plot(DCB,yyp)
        pause(2)
    end
    
    
    %% Visualization of GPU fitings  and savings
    
    %for visualization purposes, we replace any r2 less than zero with nan
    howwedo=(GPUout.states~=0); %the condition for failing the fit
    GPUout.parameters(howwedo)=nan;
    howwedo=(GPUout.R2<.5); %the condition for R2 
    GPUout.parameters(howwedo)=nan;
    howwedo=(GPUoutP.R2<.5); %the condition for R2 
    GPUoutP.parameters(howwedo)=nan;
    %here we reshape thevectors to a matrix for visuzation , we also flip data
    %to match with its AR represenation
    %amp data
    Visu.Wimag=reshape(GPUout.parameters(1,:),sizereduced);
    Visu.Qimag=reshape(GPUout.parameters(2,:),sizereduced);
    Visu.Aimag=reshape(GPUout.parameters(3,:),sizereduced);
    Visu.DCimag=reshape(GPUout.parameters(4,:),sizereduced);
    Visu.r2imag=reshape(GPUout.R2,sizereduced);
    figure;imagesc((Visu.Wimag));axis    square ;colorbar;view(-90,90);caxiscorrection(Visu.Wimag);title('\omega_0');axis off
    figure;imagesc(abs(Visu.Qimag));axis    square ;colorbar;view(-90,90);title('Q');caxiscorrection(abs(Visu.Qimag));axis off
    figure;imagesc((Visu.Aimag));axis    square ;colorbar;view(-90,90);title('A_0');caxiscorrPhase(Visu.Aimag);axis off
    figure;imagesc((Visu.DCimag));axis    square ;colorbar;view(-90,90);title('offset');caxiscorrPhase(Visu.DCimag);axis off
    figure;imagesc((Visu.r2imag));axis   square ;colorbar;caxis([0.0 1]);view(-90,90);title('R^2');caxiscorrection(Visu.r2imag);axis off
    
    
    
    %phase data
    Visu.PWimag=reshape(GPUoutP.parameters(1,:),sizereduced);
    Visu.PQimag=reshape(GPUoutP.parameters(2,:),sizereduced);
    Visu.PPhiimag=reshape(GPUoutP.parameters(3,:),sizereduced);
    Visu.Pr2imag=reshape(GPUoutP.R2,sizereduced);
    figure;imagesc(abs((Visu.PWimag)));axis square ;colorbar;view(-90,90);caxiscorrPhase(abs(Visu.PWimag));title('f_{0 Phase}');axis off
    figure;imagesc(abs(Visu.PQimag));axis  square ;colorbar;view(-90,90);title('Q_{Phase}');caxiscorrection(abs(Visu.PQimag));axis off
    figure;imagesc((Visu.PPhiimag));axis   square ;colorbar;caxis([0.0 1]);view(-90,90);title('\phi_0');caxiscorrPhase(Visu.PPhiimag);axis off
    figure;imagesc((Visu.Pr2imag));axis   square ;colorbar;caxis([0.0 1]);view(-90,90);title('R^2');caxis([.5 1 ]);axis off

    save(strcat(File{1,1}(1:end-3),'.mat'),'Visu') %save raw data of all post-processing SHO images
end
%% Disclaimer
% I have made every effort to evaluate the proper working of this code
% under many different conditions. However, it is the responsibility of
% the user to ensure that this registration code is adequate and working
% correcntly for their application.
%
% Feel free to e-mail me with questions or comments. ehsann@uw.edu