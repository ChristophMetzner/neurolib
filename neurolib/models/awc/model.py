from neurolib.models.awc import loadDefaultParams as dp
from neurolib.models.awc import timeIntegration as ti
from neurolib.models.model import Model


class AWCModel(Model):
    """
    An augmented Wilson-Cowan model including two different inhibitory subpopulations: SOM+ and PV+ interneurons
    (Park&Geffen,bioRxiv, 2019)
    """

    name = "awc"
    description = "Augmented Wilson-Cowan model"

    init_vars = ["us_init", "ps_init", "ss_init", "u_ou", "p_ou", "s_ou"]
    state_vars = ["u", "p", "s", "u_ou", "p_ou", "s_ou"]
    output_vars = ["u", "p", "s"]
    default_output = "u"
    input_vars = ["u_ext", "p_ext", "s_ext"]
    default_input = "u_ext"

    # because this is not a rate model, the input
    # to the bold model must be normalized
    normalize_bold_input = True
    normalize_bold_input_max = 50

    def __init__(
        self, params=None, Cmat=None, Dmat=None, lookupTableFileName=None, seed=None, bold=False,
    ):

        self.Cmat = Cmat
        self.Dmat = Dmat
        self.seed = seed

        # the integration function must be passed
        integration = ti.timeIntegration

        # load default parameters if none were given
        if params is None:
            params = dp.loadDefaultParams(Cmat=self.Cmat, Dmat=self.Dmat, seed=self.seed)

        # Initialize base class Model
        super().__init__(
            integration=integration, params=params
        )
