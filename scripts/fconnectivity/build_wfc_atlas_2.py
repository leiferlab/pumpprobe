mport numpy as np, sys
import wormfunconn as wfc

fa = wfc.FunctionalAtlas(np.copy(neu_ids),km)
fa.scalar_atlas = np.copy(intensity_map)

dst_folder = os.path.join("/home/frandi/dev/worm-functional-connectivity/atlas/")
if strain == "": 
    strain = "wild-type"
elif strain == "unc31":
    strain = "unc-31"
fa.to_file(dst_folder,strain)
