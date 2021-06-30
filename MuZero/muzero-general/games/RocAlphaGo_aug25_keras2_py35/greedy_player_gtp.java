package com.fesva;

import AlphaGo.ai.*;
import AlphaGo.models.policy.*;
import Interface.gtp_wrapper.*;

public class Main {

    public static void main(String[] args) {

        MODEL = r'..\RocAlphaGo.data\models\CNNPolicy\48plane_192filter_12layer_keras2.json';

        WEIGHTS = r'..\RocAlphaGo.data\training_results\supervised\48plane_KGS_6d_higher\accuracy_57_filters_192\weights.00116.hdf5';
        policy = CNNPolicy.load_model(MODEL);
        policy.model.load_weights(WEIGHTS);
        player = GreedyPolicyPlayer(policy);
        run_gtp(player);
    }
}


