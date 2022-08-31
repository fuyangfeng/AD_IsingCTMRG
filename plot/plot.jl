using Plots
using DelimitedFiles

function myplot1(path,XYlabel,legendlabel)
    m = length(path)
    res = Array{ Array{ Float64, 2 }, 1 }( undef, m )
    for i in 1:m 
        res[i]=readdlm(path[i])
    end
    n = length(res[1][1,:])
    for i in 2:n
        p = plot(xlabel=XYlabel[1],ylabel=XYlabel[i])
        for j in 1:m
            plot!(p,res[j][:,1],res[j][:,i],shape =:circle ,markersize=4,label=legendlabel[j])
        end
        savefig(XYlabel[i])
    end
end
function myplot2(path,flag,legendlabel,XYlabel)

    m = length(path)
    res = Array{ Array{ Float64, 2 }, 1 }( undef, m )
    for i in 1:m 
        res[i]=readdlm(path[i])
    end
    n = length(flag[:,1])
    p = plot(xlabel=XYlabel[1],ylabel=XYlabel[2])
    for i in 1:n 
        plot!(p,res[flag[i,1]][:,1],res[flag[i,1]][:,flag[i,2]],shape =:circle ,markersize=4,label=legendlabel[i])
    end
    savefig(XYlabel[2])
end
function getmyfig()

    path = []
    path=vcat(path, [  "../data/IsingExact.txt"  ])
    path=vcat(path, [  "../data/ADTRGstep=40.txt"  ])
    path=vcat(path, [  "../data/Ising_AD_CTMRGstep=10_D=80.txt"  ])

    flag = [1 2]
    flag=vcat(flag, [2 2])
    flag=vcat(flag, [3 2])
    legendlabel2 = []  
    legendlabel2=vcat(legendlabel2, [  "exact solution"  ])
    legendlabel2=vcat(legendlabel2, [  "TRGstep=40"  ])
    legendlabel2=vcat(legendlabel2, [  "CTMRGstep = 10"  ])
    myplot2(path,flag,legendlabel2 ,["β","lnz"])



    flag = [1 4]
    flag=vcat(flag, [2 4])
    flag=vcat(flag, [3 6])
    flag=vcat(flag, [3 8])
    legendlabel2 = []  
    legendlabel2=vcat(legendlabel2, [  "exact solution"  ])
    legendlabel2=vcat(legendlabel2, [  "TRGstep=40"  ])
    legendlabel2=vcat(legendlabel2, [  "CTMRGstep=10_AD by U"  ])
    legendlabel2=vcat(legendlabel2, [  "CTMRGstep=10_AD by F"  ])
    myplot2(path,flag,legendlabel2 ,["β","Cv"])


    flag = [1 3]
    flag=vcat(flag, [2 3])
    flag=vcat(flag, [3 4])
    flag=vcat(flag, [3 7])
    legendlabel2 = []  
    legendlabel2=vcat(legendlabel2, [  "exact solution"  ])
    legendlabel2=vcat(legendlabel2, [  "TRGstep=40"  ])
    legendlabel2=vcat(legendlabel2, [  "CTMRGstep = 10"  ])
    legendlabel2=vcat(legendlabel2, [  "CTMRGstep = 10_AD"  ])
    myplot2(path,flag,legendlabel2 ,["β","Uenergy"])

end

@time getmyfig()

