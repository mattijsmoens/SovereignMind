"""
SovereignMind: Neural Semantic Decoder (HuthLab Integration)

This module wraps the real UT Austin HuthLab semantic decoding pipeline
(Tang et al., Nature Neuroscience 2023) into a callable Python interface
for use by the SovereignMind consensus engine.

Paper: "Semantic reconstruction of continuous language from non-invasive brain recordings"
Authors: Jerry Tang, Amanda LeBel, Shailee Jain, Alexander G. Huth
Repository: https://github.com/HuthLab/semantic-decoding

Requirements:
    - Pre-fit encoding models in semantic-decoding/models/[SUBJECT_ID]/
    - Language model data in semantic-decoding/data_lm/
    - numpy, scipy, h5py, torch
"""

import os
import sys
import json
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SovereignMind.Decoder")

# Add the HuthLab decoding directory to sys.path so we can import their modules
HUTHLAB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "semantic-decoding", "decoding")
if HUTHLAB_DIR not in sys.path:
    sys.path.insert(0, HUTHLAB_DIR)

# Detect device - fallback to CPU if CUDA is not available
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[HuthLab] PyTorch device: {DEVICE}")
except ImportError:
    DEVICE = "cpu"
    logger.warning("[HuthLab] PyTorch not installed. Decoder will not function until 'pip install torch' is run.")


def patch_huthlab_config_for_device():
    """
    The HuthLab config.py hardcodes 'cuda' for all devices.
    This patches it at runtime to use CPU if no GPU is available.
    """
    try:
        import config as huthlab_config
        huthlab_config.GPT_DEVICE = DEVICE
        huthlab_config.EM_DEVICE = DEVICE
        huthlab_config.SM_DEVICE = DEVICE
        logger.info(f"[HuthLab] Patched all devices to: {DEVICE}")
    except ImportError:
        logger.error("[HuthLab] Could not import HuthLab config. Is semantic-decoding cloned?")


def check_models_exist(subject_id: str) -> bool:
    """
    Verifies that the pre-fit encoding models exist for a given subject.
    Returns True if models are found, False otherwise.
    """
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                               "semantic-decoding", "models", subject_id)
    if not os.path.exists(models_dir):
        logger.error(f"[HuthLab] No models found for subject '{subject_id}' at: {models_dir}")
        logger.error("[HuthLab] Download pre-fit models from: https://utexas.box.com/s/ri13t06iwpkyk17h8tfk0dtyva7qtqlz")
        return False
    
    # Check for required model files
    required_files = ["encoding_model_perceived.npz"]
    for f in required_files:
        if not os.path.exists(os.path.join(models_dir, f)):
            logger.error(f"[HuthLab] Missing model file: {f} in {models_dir}")
            return False
    
    logger.info(f"[HuthLab] Pre-fit models verified for subject: {subject_id}")
    return True


def decode_fmri_response(subject_id: str, experiment: str, task: str) -> dict:
    """
    Runs the real HuthLab semantic decoder on pre-recorded fMRI brain responses.
    
    This is the actual decoder from the Nature Neuroscience 2023 paper.
    It uses Ridge Regression encoding models + GPT-based beam search to reconstruct
    continuous language from non-invasive brain recordings.
    
    Args:
        subject_id: The subject identifier (e.g., "UTS01", "UTS02", "UTS03")
        experiment: The experiment type (e.g., "perceived_speech", "imagined_speech", "perceived_movies")
        task: The specific task/story name within the experiment
        
    Returns:
        dict with keys:
            - "decoded_words": list of reconstructed words
            - "word_times": list of predicted word onset times
            - "subject": the subject ID used
            - "experiment": the experiment type
            - "task": the task name
    """
    patch_huthlab_config_for_device()
    
    import config as huthlab_config
    import h5py
    from GPT import GPT
    from Decoder import Decoder, Hypothesis
    from LanguageModel import LanguageModel
    from EncodingModel import EncodingModel
    from StimulusModel import StimulusModel, get_lanczos_mat, affected_trs, LMFeatures
    from utils_stim import predict_word_rate, predict_word_times
    
    # Validate models exist
    if not check_models_exist(subject_id):
        raise FileNotFoundError(f"Pre-fit models not found for subject '{subject_id}'. "
                                 "Download from: https://utexas.box.com/s/ri13t06iwpkyk17h8tfk0dtyva7qtqlz")
    
    # Determine GPT checkpoint based on experiment type
    if experiment in ["imagined_speech"]:
        gpt_checkpoint = "imagined"
    else:
        gpt_checkpoint = "perceived"
    
    # Determine word rate model voxels based on experiment type
    if experiment in ["imagined_speech", "perceived_movies"]:
        word_rate_voxels = "speech"
    else:
        word_rate_voxels = "auditory"
    
    # Load fMRI brain responses
    response_path = os.path.join(huthlab_config.DATA_TEST_DIR, "test_response", 
                                  subject_id, experiment, task + ".hf5")
    logger.info(f"[HuthLab] Loading fMRI responses from: {response_path}")
    hf = h5py.File(response_path, "r")
    resp = np.nan_to_num(hf["data"][:])
    hf.close()
    logger.info(f"[HuthLab] Loaded response matrix: {resp.shape}")
    
    # Load GPT language model
    logger.info(f"[HuthLab] Loading GPT checkpoint: {gpt_checkpoint}")
    with open(os.path.join(huthlab_config.DATA_LM_DIR, gpt_checkpoint, "vocab.json"), "r") as f:
        gpt_vocab = json.load(f)
    with open(os.path.join(huthlab_config.DATA_LM_DIR, "decoder_vocab.json"), "r") as f:
        decoder_vocab = json.load(f)
    gpt = GPT(path=os.path.join(huthlab_config.DATA_LM_DIR, gpt_checkpoint, "model"), 
              vocab=gpt_vocab, device=huthlab_config.GPT_DEVICE)
    features = LMFeatures(model=gpt, layer=huthlab_config.GPT_LAYER, context_words=huthlab_config.GPT_WORDS)
    lm = LanguageModel(gpt, decoder_vocab, nuc_mass=huthlab_config.LM_MASS, nuc_ratio=huthlab_config.LM_RATIO)
    
    # Load pre-fit encoding model and word rate model
    load_location = os.path.join(huthlab_config.MODEL_DIR, subject_id)
    logger.info(f"[HuthLab] Loading pre-fit encoding model from: {load_location}")
    word_rate_model = np.load(os.path.join(load_location, "word_rate_model_%s.npz" % word_rate_voxels), allow_pickle=True)
    encoding_model = np.load(os.path.join(load_location, "encoding_model_%s.npz" % gpt_checkpoint))
    weights = encoding_model["weights"]
    noise_model = encoding_model["noise_model"]
    tr_stats = encoding_model["tr_stats"]
    word_stats = encoding_model["word_stats"]
    em = EncodingModel(resp, weights, encoding_model["voxels"], noise_model, device=huthlab_config.EM_DEVICE)
    em.set_shrinkage(huthlab_config.NM_ALPHA)
    
    # Predict word timing from brain responses
    logger.info("[HuthLab] Predicting word rate from brain responses...")
    word_rate = predict_word_rate(resp, word_rate_model["weights"], word_rate_model["voxels"], word_rate_model["mean_rate"])
    if experiment == "perceived_speech":
        word_times, tr_times = predict_word_times(word_rate, resp, starttime=-10)
    else:
        word_times, tr_times = predict_word_times(word_rate, resp, starttime=0)
    lanczos_mat = get_lanczos_mat(word_times, tr_times)
    
    # Run beam search decoding - THIS IS THE REAL DECODER
    logger.info(f"[HuthLab] Running beam search decoder (width={huthlab_config.WIDTH}) over {len(word_times)} word positions...")
    decoder = Decoder(word_times, huthlab_config.WIDTH)
    sm = StimulusModel(lanczos_mat, tr_stats, word_stats[0], device=huthlab_config.SM_DEVICE)
    for sample_index in range(len(word_times)):
        trs = affected_trs(decoder.first_difference(), sample_index, lanczos_mat)
        ncontext = decoder.time_window(sample_index, huthlab_config.LM_TIME, floor=5)
        beam_nucs = lm.beam_propose(decoder.beam, ncontext)
        for c, (hyp, nextensions) in enumerate(decoder.get_hypotheses()):
            nuc, logprobs = beam_nucs[c]
            if len(nuc) < 1:
                continue
            extend_words = [hyp.words + [x] for x in nuc]
            extend_embs = list(features.extend(extend_words))
            stim = sm.make_variants(sample_index, hyp.embs, extend_embs, trs)
            likelihoods = em.prs(stim, trs)
            local_extensions = [Hypothesis(parent=hyp, extension=x) for x in zip(nuc, logprobs, extend_embs)]
            decoder.add_extensions(local_extensions, likelihoods, nextensions)
        decoder.extend(verbose=False)
    
    decoded_words = decoder.beam[0].words
    logger.info(f"[HuthLab] Decoding complete. Reconstructed {len(decoded_words)} words.")
    logger.info(f"[HuthLab] Decoded text: {' '.join(decoded_words[:50])}...")
    
    return {
        "decoded_words": decoded_words,
        "decoded_text": " ".join(decoded_words),
        "word_times": word_times.tolist() if hasattr(word_times, 'tolist') else list(word_times),
        "subject": subject_id,
        "experiment": experiment,
        "task": task
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SovereignMind HuthLab Semantic Decoder")
    parser.add_argument("--subject", type=str, required=True, help="Subject ID (e.g., UTS01)")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment type (e.g., perceived_speech)")
    parser.add_argument("--task", type=str, required=True, help="Task/story name")
    args = parser.parse_args()
    
    result = decode_fmri_response(args.subject, args.experiment, args.task)
    print("\n=== Decoded Brain Signal ===")
    print(f"Subject: {result['subject']}")
    print(f"Experiment: {result['experiment']}")
    print(f"Task: {result['task']}")
    print(f"Decoded Text: {result['decoded_text']}")
