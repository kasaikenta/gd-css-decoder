GF=512;

fid=fopen('BINGF512','w');
for k=1:GF,
    for l=1:log2(GF),
        fprintf(fid,'%d\t',GF512(k,l));
    end
    fprintf(fid,'\n');
end
fclose(fid);


fid=fopen('ADDGF512','w');
for k=1:GF,
    for l=1:GF,
        fprintf(fid,'%d\t',GFADD512(k,l));
    end
    fprintf(fid,'\n');
end
fclose(fid);

fid=fopen('MULGF512','w');
for k=1:GF,
    for l=1:GF,
        fprintf(fid,'%d\t',GFMUL512(k,l));
    end
    fprintf(fid,'\n');
end
fclose(fid);

fid=fopen('DIVGF512','w');
for k=1:GF,
    for l=1:GF,
        fprintf(fid,'%d\t',GFDIV512(k,l));
    end
    fprintf(fid,'\n');
end
fclose(fid);

return;


[EbN1,Lambda1,Rho1]=lanceOptim(12); tr=[7 8]; rho=[0.5 0.5];
[EbN2,Lambda2,Rho2]=lanceOptim(20); tr=[8 9]; rho=[0.7 0.3];
[EbN3,Lambda3,Rho3]=lanceOptim(50); tr=[9 10]; rho=[0.1 0.9];
