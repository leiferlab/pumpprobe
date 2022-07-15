#!/bin/bash

echo "These scripts expect that you have run identify_stim_neurons.py"

full_array=(
## 0
/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20210826/pumpprobe_20210826_145453/   # F unc13 A12 mutant
/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20210826/pumpprobe_20210826_161811/   # F unc13 A12 mutant
/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20210824/pumpprobe_20210824_104000/   # F
/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20210824/pumpprobe_20210824_114940/   # F
/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20210824/pumpprobe_20210824_144628/   # F
/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20210824/pumpprobe_20210824_161919/   # F
/projects/LEIFER/Sophie/NewRecordings/20210825/pumpprobe_20210825_123600/               # S
## 1
/projects/LEIFER/Sophie/NewRecordings/20210825/pumpprobe_20210825_150349/               # S median
/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20210827/pumpprobe_20210827_104408/   # F median
/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20210827/pumpprobe_20210827_115857/   # F median
/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20210827/pumpprobe_20210827_142226/   # F median
/projects/LEIFER/Sophie/NewRecordings/20210830/pumpprobe_20210830_111646/               # S median
/projects/LEIFER/Sophie/NewRecordings/20210830/pumpprobe_20210830_144155/               # S median
/projects/LEIFER/Sophie/NewRecordings/20210830/pumpprobe_20210830_161543/               # S median
## 2 median
/projects/LEIFER/Sophie/NewRecordings/20210902/pumpprobe_20210902_115146/               # S median
/projects/LEIFER/Sophie/NewRecordings/20210909/pumpprobe_20210909_102637/               # S median
/projects/LEIFER/Sophie/NewRecordings/20210913/pumpprobe_20210913_105324/               # S median
/projects/LEIFER/Sophie/NewRecordings/20210913/pumpprobe_20210913_161934/               # S median
## 2 30
/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20210903/pumpprobe_20210903_153005/   # F 30
/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20210907/pumpprobe_20210907_110334/   # F 30
/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20210907/pumpprobe_20210907_141336/   # F 30
/projects/LEIFER/Sophie/NewRecordings/20210917/pumpprobe_20210917_104948/               # S 30
## 3 30
/projects/LEIFER/Sophie/NewRecordings/20210917/pumpprobe_20210917_132736/	            # S 30
/projects/LEIFER/Sophie/NewRecordings/20210928/pumpprobe_20210928_105933/	            # S 30
/projects/LEIFER/Sophie/NewRecordings/20210928/pumpprobe_20210928_134411/	            # S 30
/projects/LEIFER/Sophie/NewRecordings/20211008/pumpprobe_20211008_115929/	            # S 30
/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20211102/pumpprobe_20211102_101248/   # F 30 lambda 5 RID D20 innexins
/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20211102/pumpprobe_20211102_152524/   # F 30 lambda 5 RID D20 innexins
/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20211104/pumpprobe_20211104_163944/   # F 30
## 4
/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20211226_Sophie/pumpprobe_20211216_164305/ # 30 F unc31 mutant
/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20220113/pumpprobe_20220113_101730/   # F 30 unc31 mutant 508.8.9
/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20220113/pumpprobe_20220113_112922/   # F 30 unc31 mutant 508.8.9
/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20220113/pumpprobe_20220113_142740/   # F 30 unc31 mutant 508.8.9
/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20220113/pumpprobe_20220113_153037/   # F 30
/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20220113/pumpprobe_20220113_165125/   # F 30 unc31 mutant
/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20220120/pumpprobe_20220120_135951/   # F 30
## 5
/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20220120/pumpprobe_20220120_150453/   # F 30 RID
/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20220120/pumpprobe_20220120_151357/   # F 30 
/projects/LEIFER/Sophie/NewRecordings/20211022/pumpprobe_20211022_140822/               # S 30
/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20220211/pumpprobe_20220211_094256/   # F 30 RID unc31 mutant 510.9.b
/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20220211/pumpprobe_20220211_141039/   # F 30 RID unc31 mutant 510.9.b
/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20220211/pumpprobe_20220211_151211/   # F 30 lambda 5 RID unc31 mutant 510.9.b
/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20220224/pumpprobe_20220224_140933/   # F 30 lambda 5 RIC inx1 mutant innexins
/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20220224/pumpprobe_20220224_153744/   # F 30 lambda 5 RIC inx1 mutant innexins
/projects/LEIFER/Sophie/NewRecordings/20220126/pumpprobe_20220126_153426/	            # S 30 
/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20220330/pumpprobe_20220330_110253/   # F 30 lambda 5
## 6
/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20220330/pumpprobe_20220330_123606/   # F 30 lambda 5
/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20220330/pumpprobe_20220330_164449/   # F 30 lambda 5
/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20220331/pumpprobe_20220331_114306/   # F 30 lambda 5
/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20220331/pumpprobe_20220331_142924/   # F 30 lambda 5
/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20220331/pumpprobe_20220331_153103/   # F 30 lambda 5
## 7
/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20220405/pumpprobe_20220405_100245/   # F 30 lambda 5
/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20220405/pumpprobe_20220405_110704/   # F 30 lambda 5
/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20220405/pumpprobe_20220405_125435/   # F 30 lambda 5
/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20220405/pumpprobe_20220405_143326/   # F 30 lambda 5
/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20220412/pumpprobe_20220412_112143/   # F 30 lambda 5 RID unc31 mutant 510.9.b
/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20220422/pumpprobe_20220422_104335/   # F 30 lambda 5 AFD unc31 mutant 510.9.b
)

my_array=(
## 4
#/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20211226_Sophie/pumpprobe_20211216_164305/ # 30 F unc31 mutant
#/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20220113/pumpprobe_20220113_101730/   # F 30 unc31 mutant 508.8.9
/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20220113/pumpprobe_20220113_112922/   # F 30 unc31 mutant 508.8.9
/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20220113/pumpprobe_20220113_142740/   # F 30 unc31 mutant 508.8.9
#/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20220113/pumpprobe_20220113_153037/   # F 30
/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20220113/pumpprobe_20220113_165125/   # F 30 unc31 mutant
#/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20220120/pumpprobe_20220120_135951/   # F 30
)

for folder in "${my_array[@]}"
do
    echo ""
    echo "##################"
    echo "processing $folder"
    #echo "just updating"
    echo ""
    python ../scripts/fconnectivity/responses_detect.py $folder --signal:green -y --ampl-min-time:4 --ampl-thresh:1.3 --deriv-min-time:4 --deriv-thresh:0.02 --smooth-mode:sg_causal --smooth-n:13 --smooth-poly:1 --matchless-nan-th-from-file --matchless-nan-th:0.5
    python ../scripts/fconnectivity/fit_responses_unconstrained_eci.py $folder --signal:green --matchless-nan-th-from-file --matchless-nan-th:0.5 
    python ../scripts/fconnectivity/fit_responses_constrained_stim_eci.py $folder --signal:green --skip-if-not-manually-located --matchless-nan-th-from-file --matchless-nan-th:0.5
done
