for ((i=0;i<42;i+=1))
do
    echo "processing $i"
    python funatlas_plot_intensity_map.py --nan-th:0.3 --req-auto-response --exclude-ds-tags:mutant --ds-exclude-i:$i --no-plot --no-merge-bilateral --matchless-nan-th-from-file --enforce-stim-crosscheck
done
