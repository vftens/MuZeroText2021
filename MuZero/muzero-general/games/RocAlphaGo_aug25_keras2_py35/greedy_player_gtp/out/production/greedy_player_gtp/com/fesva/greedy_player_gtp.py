from AlphaGo.ai import GreedyPolicyPlayer
from AlphaGo.models.policy import CNNPolicy
from interface.gtp_wrapper import run_gtp

MODEL = r'..\RocAlphaGo.data\models\CNNPolicy\48plane_192filter_12layer_keras2.json'

#'tests/test_data/minimodel_policy.json'
WEIGHTS = r'..\RocAlphaGo.data\training_results\supervised\48plane_KGS_6d_higher\accuracy_57_filters_192\weights.00116.hdf5'
#'tests/test_data/traindata/weights.00999.hdf5'

policy = CNNPolicy.load_model(MODEL)
policy.model.load_weights(WEIGHTS)

player = GreedyPolicyPlayer(policy)

run_gtp(player)
