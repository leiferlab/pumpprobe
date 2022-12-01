__all__ = ['integration']

from ._integration import integral
from .integration import integral as integral_py
from ._convolution import convolution1, convolution,slice_test
from .dyson import dyson
from .Pipeline import Pipeline
from .Fconn import Fconn
from .Anatlas import Anatlas
from .Funatlas import Funatlas
from .ExponentialConvolution import ExponentialConvolution
from .provenance import stamp as provstamp
from .plotutils import make_alphacolorbar, make_colorbar, plot_linlog, scatter_hist, c_in_wt, c_not_in_wt, c_wt, c_unc31
from .statsutils import weighted_corr, pearsonr_sample, R2, p_to_stars, R2nl
