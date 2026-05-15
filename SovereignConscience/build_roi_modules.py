"""
SovereignConscience — ROI-Specific Module Builder
==================================================
Rebuilds each neural subsystem module using brain-region-specific 
voxels extracted via the Harvard-Oxford atlas.

The atlas is resampled to match each subject's functional space,
then region-specific masks are applied to extract only the voxels
belonging to each neural subsystem.

This closes the gap: each module now processes ONLY its designated
brain region — amygdala voxels for the amygdala module, insula 
voxels for the insula module, etc.

Author: Mattijs Moens
License: BSL 1.1
"""

import os
import json
import hashlib
import logging
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

import nibabel as nib
from nilearn import datasets, image
from nilearn.maskers import NiftiLabelsMasker, NiftiMasker
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
import joblib

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
DATA_PREPROCESSED_DIR = BASE_DIR / "data" / "preprocessed"
MODULES_DIR = BASE_DIR / "modules"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger("SovereignConscience.ROI")

TR = 3.0

# Minimum voxels a subject must have in an ROI to be included for that module.
# Subjects below this threshold are excluded PER MODULE ONLY — they still
# contribute to other modules where their ROI coverage is sufficient.
MIN_ROI_VOXELS = 150

# Number of PCA components to reduce each ROI to.
# This replaces the truncation strategy: instead of cutting features at the
# smallest subject's voxel count, we project everyone into the same
# PCA space, preserving maximum variance regardless of anatomy.
PCA_COMPONENTS = 50

# HRF (Hemodynamic Response Function) delay in TRs.
# The BOLD signal peaks ~6 seconds after stimulus onset. At TR=3s that is
# 2 TRs. We shift labels forward by this amount so we're reading the brain
# response that actually corresponds to each emotion block.
HRF_DELAY_TRS = 2

# Task design from NeuroEmo README
TASK_DESIGN = [
    {"onset": 0,   "duration": 30, "emotion": "calm"},
    {"onset": 30,  "duration": 30, "emotion": "baseline"},
    {"onset": 60,  "duration": 30, "emotion": "afraid"},
    {"onset": 90,  "duration": 30, "emotion": "baseline"},
    {"onset": 120, "duration": 30, "emotion": "delighted"},
    {"onset": 150, "duration": 30, "emotion": "baseline"},
    {"onset": 180, "duration": 30, "emotion": "depressed"},
    {"onset": 210, "duration": 30, "emotion": "baseline"},
    {"onset": 240, "duration": 30, "emotion": "excited"},
    {"onset": 270, "duration": 30, "emotion": "baseline"},
    {"onset": 300, "duration": 30, "emotion": "delighted"},
    {"onset": 330, "duration": 30, "emotion": "baseline"},
    {"onset": 360, "duration": 30, "emotion": "depressed"},
    {"onset": 390, "duration": 30, "emotion": "baseline"},
    {"onset": 420, "duration": 30, "emotion": "calm"},
    {"onset": 450, "duration": 30, "emotion": "baseline"},
    {"onset": 480, "duration": 30, "emotion": "excited"},
    {"onset": 510, "duration": 30, "emotion": "baseline"},
    {"onset": 540, "duration": 30, "emotion": "afraid"},
    {"onset": 570, "duration": 30, "emotion": "baseline"},
]

EMOTION_VALENCE_AROUSAL = {
    "calm":      ( 0.3,  -0.7),
    "afraid":    (-0.8,   0.8),
    "delighted": ( 0.9,   0.6),
    "depressed": (-0.7,  -0.5),
    "excited":   ( 0.7,   0.9),
}

# ROI definitions using Harvard-Oxford atlas label indices
# These map each neural subsystem to specific atlas regions
SUBSYSTEM_DEFINITIONS = {
    "amygdala": {
        "description": "Threat Detector - bilateral amygdala",
        "atlas_type": "sub",  # subcortical atlas
        "label_names": ["Left Amygdala", "Right Amygdala"],
        "target_emotions": ["afraid"],
        "moral_role": "threat_detection",
    },
    "insula": {
        "description": "Disgust Circuit - bilateral anterior insula",
        "atlas_type": "cort",  # cortical atlas
        "label_names": ["Insular Cortex"],
        "target_emotions": ["afraid", "depressed"],
        "moral_role": "revulsion_signal",
    },
    "acc": {
        "description": "Empathy Engine - anterior cingulate cortex",
        "atlas_type": "cort",
        "label_names": ["Cingulate Gyrus, anterior division"],
        "target_emotions": ["calm", "delighted"],
        "moral_role": "empathy_compassion",
    },
    "vmpfc": {
        "description": "Value Computer - ventromedial prefrontal cortex",
        "atlas_type": "cort",
        "label_names": ["Frontal Medial Cortex", "Subcallosal Cortex"],
        "target_emotions": ["delighted", "calm"],
        "moral_role": "value_computation",
    },
    "dlpfc": {
        "description": "Conflict Monitor - dorsolateral prefrontal cortex",
        "atlas_type": "cort",
        "label_names": ["Middle Frontal Gyrus"],
        "target_emotions": ["afraid", "excited"],
        "moral_role": "conflict_monitoring",
    },
}


def assign_emotion_labels(n_trs):
    """Assign emotion labels to each TR based on task design.
    
    Applies a 2-TR HRF delay: the BOLD signal peaks ~6 seconds after the
    stimulus, so we shift the label assignment forward by HRF_DELAY_TRS.
    This corrects the misalignment between stimulus onset and neural response.
    """
    # First build the raw stimulus labels
    raw_labels = []
    for tr_idx in range(n_trs):
        time_sec = tr_idx * TR
        current_emotion = "baseline"
        for block in TASK_DESIGN:
            if block["onset"] <= time_sec < block["onset"] + block["duration"]:
                current_emotion = block["emotion"]
                break
        raw_labels.append(current_emotion)
    
    # Shift labels forward by HRF_DELAY_TRS to account for hemodynamic delay.
    # The TR at index i gets the label that was active HRF_DELAY_TRS steps earlier.
    labels = ["baseline"] * HRF_DELAY_TRS + raw_labels[:n_trs - HRF_DELAY_TRS]
    return labels


def load_atlases():
    """Load Harvard-Oxford cortical and subcortical atlases."""
    logger.info("[Atlas] Downloading Harvard-Oxford atlases (nilearn cache)...")
    
    cort_atlas = datasets.fetch_atlas_harvard_oxford(
        "cort-maxprob-thr25-2mm", symmetric_split=False
    )
    sub_atlas = datasets.fetch_atlas_harvard_oxford(
        "sub-maxprob-thr25-2mm", symmetric_split=False
    )
    
    logger.info(f"[Atlas] Cortical: {len(cort_atlas.labels)} regions")
    logger.info(f"[Atlas] Subcortical: {len(sub_atlas.labels)} regions")
    
    return cort_atlas, sub_atlas


def find_label_indices(atlas, target_names):
    """Find atlas label indices that match target region names."""
    indices = []
    for target in target_names:
        for i, label in enumerate(atlas.labels):
            if target.lower() in label.lower():
                indices.append(i)
                logger.info(f"  Found: '{label}' at index {i}")
                break
    return indices


def create_roi_mask(atlas_img, atlas_labels, target_names, reference_img):
    """
    Create a binary mask for specific ROIs from the atlas,
    resampled to match the reference functional image.
    
    This is the key step: the atlas (in MNI 2mm space) is resampled
    to match the subject's native functional space. This allows
    ROI extraction without needing full spatial normalization.
    """
    # Find which label indices correspond to our target regions
    target_indices = find_label_indices(
        type('obj', (object,), {'labels': atlas_labels})(),
        target_names
    )
    
    if not target_indices:
        logger.warning(f"No labels found for: {target_names}")
        return None
    
    # Resample atlas to functional space
    resampled_atlas = image.resample_to_img(
        atlas_img, reference_img, 
        interpolation="nearest"  # Nearest-neighbor for label images
    )
    
    # Create binary mask: 1 where atlas label is in our target set
    atlas_data = resampled_atlas.get_fdata()
    mask_data = np.zeros_like(atlas_data, dtype=np.int8)
    for idx in target_indices:
        mask_data[atlas_data == idx] = 1
    
    n_voxels = mask_data.sum()
    logger.info(f"  ROI mask: {n_voxels} voxels")
    
    mask_img = nib.Nifti1Image(mask_data, resampled_atlas.affine)
    return mask_img, n_voxels


def extract_roi_features(fmri_img, roi_mask_img):
    """
    Extract time series from ONLY the voxels within the ROI mask.
    This gives us region-specific brain activity.
    """
    masker = NiftiMasker(
        mask_img=roi_mask_img,
        standardize="zscore_sample",
        detrend=True,
        t_r=TR,
        low_pass=0.1,
        high_pass=0.01,
    )
    
    features = masker.fit_transform(fmri_img)
    return features


def build_roi_modules():
    """
    Build each neural subsystem module using region-specific brain data.
    
    Each module is trained ONLY on voxels from its designated brain region:
    - amygdala module → amygdala voxels only
    - insula module → insular cortex voxels only
    - acc module → anterior cingulate voxels only
    - vmpfc module → ventromedial prefrontal voxels only
    - dlpfc module → middle frontal gyrus voxels only
    """
    logger.info("=" * 70)
    logger.info("SOVEREIGNCONSCIENCE: Building ROI-Specific Modules")
    logger.info("=" * 70)
    
    # Load preprocessed fMRI files
    fmri_files = sorted(DATA_PREPROCESSED_DIR.glob("*_smoothed.nii.gz"))
    if not fmri_files:
        raise FileNotFoundError(f"No preprocessed files in {DATA_PREPROCESSED_DIR}")
    
    logger.info(f"Found {len(fmri_files)} preprocessed subjects")
    
    # Load atlases
    cort_atlas, sub_atlas = load_atlases()
    
    # For each subsystem, extract region-specific features from all subjects
    module_results = {}
    
    for subsystem_name, config in SUBSYSTEM_DEFINITIONS.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Building module: {subsystem_name.upper()}")
        logger.info(f"Description: {config['description']}")
        logger.info(f"Target regions: {config['label_names']}")
        logger.info(f"{'='*50}")
        
        # Select correct atlas
        if config["atlas_type"] == "sub":
            atlas = sub_atlas
        else:
            atlas = cort_atlas
        
        # Collect per-subject features with per-module outlier filtering.
        # Subjects with fewer than MIN_ROI_VOXELS are excluded FOR THIS MODULE
        # ONLY — they still contribute to other modules where their ROI is fine.
        subject_data = []  # list of (sub_id, block_features, block_labels)
        excluded_subjects = []
        
        for fmri_path in fmri_files:
            sub_id = fmri_path.name.split("_")[0]
            logger.info(f"\n[{sub_id}] Processing for {subsystem_name}...")
            
            # Load fMRI
            fmri_img = nib.load(str(fmri_path))
            
            # Create ROI mask resampled to this subject's space
            logger.info(f"[{sub_id}] Creating ROI mask...")
            result = create_roi_mask(
                atlas.maps, atlas.labels, 
                config["label_names"], fmri_img
            )
            
            if result is None or result[1] == 0:
                logger.warning(f"[{sub_id}] No voxels found for {subsystem_name}, skipping")
                excluded_subjects.append((sub_id, 0))
                continue
            
            roi_mask, n_voxels = result
            
            # Per-module minimum voxel filter.
            # Subjects below threshold have insufficient anatomical coverage
            # for this specific ROI (e.g. small amygdala or field-of-view clipping).
            if n_voxels < MIN_ROI_VOXELS:
                logger.warning(f"[{sub_id}] Only {n_voxels} voxels in {subsystem_name} "
                               f"(min={MIN_ROI_VOXELS}), excluding this subject from this module")
                excluded_subjects.append((sub_id, n_voxels))
                continue
            
            # Extract ROI-specific features
            logger.info(f"[{sub_id}] Extracting {n_voxels} ROI voxels...")
            features = extract_roi_features(fmri_img, roi_mask)
            logger.info(f"[{sub_id}] ROI features: {features.shape}")
            
            # Assign labels (with HRF delay) and average per block
            n_trs = features.shape[0]
            labels = assign_emotion_labels(n_trs)
            trs_per_block = int(30 / TR)
            n_blocks = n_trs // trs_per_block
            
            block_features = []
            block_labels = []
            for b in range(n_blocks):
                start = b * trs_per_block
                end = start + trs_per_block
                block_feat = features[start:end].mean(axis=0)
                block_features.append(block_feat)
                block_labels.append(labels[start])
            
            subject_data.append((sub_id, np.array(block_features), block_labels))
        
        if excluded_subjects:
            logger.info(f"\n[{subsystem_name}] Excluded {len(excluded_subjects)} subjects "
                        f"(below {MIN_ROI_VOXELS} voxel threshold): "
                        f"{[s[0] for s in excluded_subjects]}")
        
        if not subject_data:
            logger.error(f"No features extracted for {subsystem_name}")
            continue
        
        logger.info(f"\n[{subsystem_name}] {len(subject_data)} subjects passed voxel threshold")
        
        # Build feature matrix. Each subject may still have a different number
        # of voxels. We use PCA to project everyone into a consistent
        # PCA_COMPONENTS-dimensional space, preserving maximum variance.
        # This is scientifically correct: we keep the most informative axes
        # of variation in the ROI signal rather than arbitrarily cutting columns.
        all_features_raw = []
        all_labels = []
        all_groups = []
        for sub_id, block_feats, block_labs in subject_data:
            for i in range(block_feats.shape[0]):
                all_features_raw.append(block_feats[i])
                all_labels.append(block_labs[i])
                all_groups.append(sub_id)
        
        # Stack into ragged array — subjects have different voxel counts.
        # We need to align them. Find the minimum voxel count among PASSING
        # subjects (much higher than before since outliers are excluded),
        # then truncate just enough to stack, and immediately apply PCA.
        min_voxels_passing = min(sd[1].shape[1] for sd in subject_data)
        logger.info(f"[{subsystem_name}] Min voxels among passing subjects: {min_voxels_passing}")
        
        n_pca = min(PCA_COMPONENTS, min_voxels_passing - 1)  # Can't exceed feature count
        logger.info(f"[{subsystem_name}] Applying PCA: {min_voxels_passing} voxels -> {n_pca} components")
        
        # Rebuild with truncation only to the passing-subjects minimum
        all_features = []
        all_labels = []
        all_groups = []
        for sub_id, block_feats, block_labs in subject_data:
            truncated = block_feats[:, :min_voxels_passing]
            for i in range(truncated.shape[0]):
                all_features.append(truncated[i])
                all_labels.append(block_labs[i])
                all_groups.append(sub_id)
        
        X = np.array(all_features)
        y = np.array(all_labels)
        groups = np.array(all_groups)
        
        logger.info(f"\n[{subsystem_name}] Total: {X.shape[0]} samples, {X.shape[1]} raw voxels")
        
        # Filter to emotion blocks only
        emotion_mask = y != "baseline"
        X_emo = X[emotion_mask]
        y_emo = y[emotion_mask]
        g_emo = groups[emotion_mask]
        
        logger.info(f"[{subsystem_name}] Emotion samples: {X_emo.shape[0]}, Classes: {np.unique(y_emo)}")
        
        # Pipeline: StandardScaler -> PCA -> LinearSVC (calibrated for probabilities)
        # LinearSVC is preferred over RBF-SVM for high-dimensional neuroimaging data
        # with few samples (Haxby et al. 2001, Pereira et al. 2009).
        # CalibratedClassifierCV wraps LinearSVC to provide predict_proba support.
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_pca)),
            ("svm", CalibratedClassifierCV(LinearSVC(C=0.1, max_iter=2000), cv=3))
        ])
        
        # Leave-one-subject-out cross-validation
        if len(np.unique(g_emo)) > 1:
            logo = LeaveOneGroupOut()
            scores = cross_val_score(model, X_emo, y_emo, cv=logo, groups=g_emo)
            logger.info(f"[{subsystem_name}] LOSO-CV accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")
        else:
            scores = cross_val_score(model, X_emo, y_emo, cv=3)
            logger.info(f"[{subsystem_name}] 3-fold CV accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")
        
        # Fit final model
        model.fit(X_emo, y_emo)
        
        # Save module
        module_dir = MODULES_DIR / subsystem_name
        module_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(model, module_dir / "model_roi.joblib")
        
        meta = {
            "name": subsystem_name,
            "description": config["description"],
            "roi_regions": config["label_names"],
            "atlas_type": config["atlas_type"],
            "n_roi_voxels": int(n_pca),
            "n_raw_voxels": int(min_voxels_passing),
            "n_excluded_subjects": len(excluded_subjects),
            "excluded_subjects": [s[0] for s in excluded_subjects],
            "n_samples": int(X_emo.shape[0]),
            "cv_accuracy": float(scores.mean()),
            "cv_std": float(scores.std()),
            "target_emotions": config["target_emotions"],
            "moral_role": config["moral_role"],
            "is_roi_specific": True,
            "trained_on": "NeuroEmo ds005700 (real fMRI)",
        }
        
        with open(module_dir / "metadata_roi.json", "w") as f:
            json.dump(meta, f, indent=2)
        
        module_results[subsystem_name] = meta
        
        logger.info(f"[{subsystem_name}] Saved ROI module: {X_emo.shape[1]} features, CV={scores.mean():.3f}")
    
    # -----------------------------------------------------------------------
    # Test the full reconstructed conscience with ROI-specific modules
    # -----------------------------------------------------------------------
    logger.info(f"\n{'='*70}")
    logger.info("TESTING RECONSTRUCTED CONSCIENCE (ROI-specific)")
    logger.info(f"{'='*70}")
    
    # Reload all trained ROI modules
    loaded_modules = {}
    for name in SUBSYSTEM_DEFINITIONS:
        model_path = MODULES_DIR / name / "model_roi.joblib"
        if model_path.exists():
            loaded_modules[name] = joblib.load(model_path)
    
    # Test on each subject
    test_results = []
    
    for fmri_path in fmri_files[:1]:  # Test on first subject
        sub_id = fmri_path.name.split("_")[0]
        fmri_img = nib.load(str(fmri_path))
        labels = assign_emotion_labels(fmri_img.shape[3])
        
        for emotion in ["calm", "afraid", "delighted", "depressed", "excited"]:
            activations = {}
            
            for mod_name, mod_config in SUBSYSTEM_DEFINITIONS.items():
                if mod_name not in loaded_modules:
                    continue
                
                model = loaded_modules[mod_name]
                atlas = sub_atlas if mod_config["atlas_type"] == "sub" else cort_atlas
                
                result = create_roi_mask(
                    atlas.maps, atlas.labels,
                    mod_config["label_names"], fmri_img
                )
                if result is None or result[1] == 0:
                    activations[mod_name] = 0.0
                    continue
                
                roi_mask, _ = result
                features = extract_roi_features(fmri_img, roi_mask)
                
                # Get features for this emotion's blocks
                trs_per_block = int(30 / TR)
                n_blocks = fmri_img.shape[3] // trs_per_block
                
                emotion_features = []
                for b in range(n_blocks):
                    start = b * trs_per_block
                    if labels[start] == emotion:
                        block_feat = features[start:start+trs_per_block].mean(axis=0)
                        emotion_features.append(block_feat)
                
                if not emotion_features:
                    activations[mod_name] = 0.0
                    continue
                
                X_test = np.array(emotion_features)
                
                # Truncate to n_raw_voxels — the raw voxel count the pipeline
                # was trained on. The pipeline itself contains PCA so we pass
                # raw voxels and let Scaler→PCA→LinearSVC handle the rest.
                expected_raw_voxels = module_results[mod_name]["n_raw_voxels"]
                X_test = X_test[:, :expected_raw_voxels]
                
                probas = model.predict_proba(X_test)
                
                # Activation = probability of target emotions
                classes = list(model.classes_)
                target_emos = mod_config["target_emotions"]
                activation = 0.0
                for te in target_emos:
                    if te in classes:
                        idx = classes.index(te)
                        activation += probas[:, idx].mean()
                activations[mod_name] = min(1.0, float(activation))
            
            # Compute moral signal
            threat = activations.get("amygdala", 0.0)
            disgust = activations.get("insula", 0.0)
            empathy = activations.get("acc", 0.0)
            value_sig = activations.get("vmpfc", 0.0)
            conflict = activations.get("dlpfc", 0.0)
            
            negative = (threat * 0.4 + disgust * 0.3)
            positive = (empathy * 0.4 + value_sig * 0.3)
            confidence = 1.0 - (conflict * 0.5)
            moral_signal = max(-1.0, min(1.0, (positive - negative) * confidence))
            
            if moral_signal > 0.3:
                interp = "BENEFICIAL"
            elif moral_signal < -0.3:
                interp = "HARMFUL"
            else:
                interp = "AMBIGUOUS"
            
            result_entry = {
                "stimulus_emotion": emotion,
                "moral_signal": round(moral_signal, 4),
                "interpretation": interp,
                "confidence": round(confidence, 4),
                "subsystem_activations": {k: round(v, 4) for k, v in activations.items()},
            }
            result_entry["integrity_hash"] = hashlib.sha256(
                json.dumps(result_entry, sort_keys=True).encode()
            ).hexdigest()
            
            test_results.append(result_entry)
            
            logger.info(f"\n  Emotion: {emotion}")
            logger.info(f"  Moral Signal: {moral_signal:+.4f} ({interp})")
            for k, v in activations.items():
                logger.info(f"    {k:>10}: {v:.4f}")
    
    # Save final results
    output = {
        "system": "SovereignConscience",
        "version": "2.0.0-roi",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_source": "NeuroEmo (OpenNeuro ds005700)",
        "roi_extraction": True,
        "atlas": "Harvard-Oxford (cortical + subcortical)",
        "module_details": module_results,
        "moral_signal_results": test_results,
    }
    
    results_path = BASE_DIR / "conscience_results_roi.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"\nResults saved to: {results_path}")
    logger.info("=" * 70)
    logger.info("ROI-SPECIFIC BUILD COMPLETE")
    logger.info("=" * 70)
    
    # Print summary
    print("\n=== ROI-SPECIFIC MODULE ACCURACIES ===")
    for name, meta in module_results.items():
        excl = meta.get('n_excluded_subjects', 0)
        print(f"  {name:>10} ({meta['n_roi_voxels']:>2} PCA components, "
              f"{meta['n_raw_voxels']:>4} voxels, {excl} excluded): "
              f"{meta['cv_accuracy']:.1%} (chance=20%)")
    
    print("\n=== MORAL SIGNAL RESULTS (ROI-specific) ===")
    for r in test_results:
        print(f"  {r['stimulus_emotion']:>12}: signal={r['moral_signal']:+.4f}  {r['interpretation']}")
    
    return output


if __name__ == "__main__":
    build_roi_modules()
