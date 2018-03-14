function caxiscorrection(dataa)
dataa(isinf(dataa))=nan;%replace inf by nan

dumM=mean(dataa(:),'omitnan');
dumS=std(dataa(:),'omitnan');
caxis([max(dumM-1.5*dumS,0) dumM+1.5*dumS])
