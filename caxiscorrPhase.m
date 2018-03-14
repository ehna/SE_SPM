function caxiscorrPhase(dataa)
dum1=mean(dataa(:),'omitnan');
dum2=std(dataa(:),'omitnan')*1.5;
caxis([dum1-dum2 dum1+dum2])
