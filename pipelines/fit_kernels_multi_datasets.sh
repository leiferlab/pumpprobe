#!/bin/bash

echo "These scripts expect that you have run identify_stim_neurons.py"

full_array=(
/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20211128/pumpprobe_20211128_112504/
...and other folders
)

for folder in "${full_array[@]}"
do
    echo ""
    echo "##################"
    echo "processing $folder"
    echo ""
    python ../scripts/fconnectivity/responses_detect.py $folder --signal:green -y --ampl-min-time:4 --ampl-thresh:1.3 --deriv-min-time:4 --deriv-thresh:0.02 --smooth-mode:sg_causal --smooth-n:13 --smooth-poly:1 --matchless-nan-th:0.5 --matchless-nan-th-added-only --nan-th:0.05
    python ../scripts/fconnectivity/fit_responses_unconstrained_eci.py $folder --signal:green --matchless-nan-th:0.5 --matchless-nan-th-added-only
    python ../scripts/fconnectivity/fit_responses_constrained_stim_eci.py $folder --signal:green --skip-if-not-manually-located --matchless-nan-th:0.5 --matchless-nan-th-added-only
done
