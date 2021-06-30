from AlphaGo.models.policy import CNNPolicy 

arch = {'filters_per_layer': 192, 'layers': 12} 
features = ["board", "ones", "turns_since", "liberties", "capture_size", "self_atari_size", "liberties_after", "ladder_capture", "ladder_escape", "sensibleness", "zeros"] 
policy = CNNPolicy(features, **arch) 
policy.save_model('my_model.json')