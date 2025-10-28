function calculTENSORFFT(GF)

     logGF=log2(GF);
     X=zeros(GF,logGF);

     file=['BINGF',num2str(GF)];
     X=load(file);

     res=[];
     for k=1:logGF,
	x=X;x(:,1)=X(:,k);x(:,k)=X(:,1);
        [i,j]=sort(x*2.^(logGF-1:-1:0)');
        res=[res;[j(1:(GF/2)),j((GF/2+1):GF)]];
     end

     res=res-1;

     file=['TENSORFFT',num2str(GF)];
     fid=fopen(file,'w');
     for (k=1:logGF*GF/2), fprintf(fid,'%d\t%d\n',res(k,1),res(k,2)); end
     fclose(fid);
