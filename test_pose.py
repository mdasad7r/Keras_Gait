import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from config_pose import TEST_PATH, SEQUENCE_LEN
from dataset.casia_pose import load_pose_sequence
from model.pose_model_ensemble import build
import tensorflow as tf

# Map view angles to folder indices
ANGLES = ['000', '018', '036', '054', '072', '090', '108', '126', '144', '162', '180']
ANGLE_IDX = {a: i for i, a in enumerate(ANGLES)}

# Which conditions belong to each protocol
PROTOCOL_CONDITIONS = {
    'NM': {'gallery': ['nm-01', 'nm-02', 'nm-03', 'nm-04'], 'probe': ['nm-05', 'nm-06']},
    'BG': {'gallery': ['nm-01', 'nm-02', 'nm-03', 'nm-04'], 'probe': ['bg-01', 'bg-02']},
    'CL': {'gallery': ['nm-01', 'nm-02', 'nm-03', 'nm-04'], 'probe': ['cl-01', 'cl-02']}
}

def extract_feature(model, sequence):
    sequence = sequence.astype('float32') / 255.0
    sequence = tf.convert_to_tensor(np.expand_dims(sequence, axis=0))  # (1, T, 8, 8, 1)
    return model(sequence).numpy()[0]

def build_gallery_probe_sets():
    gallery_db = defaultdict(lambda: defaultdict(list))  # subject → angle → features
    probe_db = defaultdict(lambda: defaultdict(list))    # protocol → subject → angle → features

    for sid in sorted(os.listdir(TEST_PATH)):
        if not sid.isdigit():
            continue
        subject_path = os.path.join(TEST_PATH, sid)

        for cond in os.listdir(subject_path):
            cond_path = os.path.join(subject_path, cond)
            for seq_id in os.listdir(cond_path):
                seq_path = os.path.join(cond_path, seq_id)
                for file in os.listdir(seq_path):
                    if file.endswith('.pkl'):
                        pkl_path = os.path.join(seq_path, file)
                        angle = file.split('-')[-1].split('.')[0]
                        if cond in PROTOCOL_CONDITIONS['NM']['gallery']:
                            gallery_db[sid][angle].append(pkl_path)
                        else:
                            for protocol, conds in PROTOCOL_CONDITIONS.items():
                                if cond in conds['probe']:
                                    probe_db[protocol][sid].append((angle, pkl_path))
                        break
    return gallery_db, probe_db

def compute_rank1_accuracy(model, gallery_db, probe_db):
    results = {}

    for protocol, subject_map in probe_db.items():
        acc_by_angle = defaultdict(list)

        for sid, probe_list in subject_map.items():
            # Build subject's gallery set
            gallery_feats = {}
            for angle, paths in gallery_db[sid].items():
                # Use first available gallery sequence for that angle
                seq = load_pose_sequence(paths[0])
                feat = extract_feature(model, seq)
                gallery_feats[angle] = feat

            for angle, pkl_path in probe_list:
                if angle not in gallery_feats:
                    continue
                probe_seq = load_pose_sequence(pkl_path)
                probe_feat = extract_feature(model, probe_seq)

                sims = []
                labels = []
                for other_sid, gal_views in gallery_db.items():
                    for gal_angle, gal_paths in gal_views.items():
                        gal_seq = load_pose_sequence(gal_paths[0])
                        gal_feat = extract_feature(model, gal_seq)
                        sim = cosine_similarity([probe_feat], [gal_feat])[0, 0]
                        sims.append(sim)
                        labels.append(other_sid)

                pred = labels[np.argmax(sims)]
                acc_by_angle[angle].append(pred == sid)

        # Final per-angle accuracy
        accs = {angle: np.mean(v) * 100 if v else 0.0 for angle, v in acc_by_angle.items()}
        accs['Mean'] = np.mean([v for v in accs.values() if isinstance(v, float)])
        results[protocol] = accs

    return results

def print_results(results):
    print("\n📊 Rank-1 Accuracy (%)")
    print(f"{'Angle':>6}", end='')
    for a in ANGLES:
        print(f"{a:>6}", end='')
    print(f"{'Mean':>8}")

    for protocol, accs in results.items():
        print(f"{protocol:>6}", end='')
        for a in ANGLES:
            val = accs.get(a, 0.0)
            print(f"{val:6.1f}", end='')
        print(f"{accs['Mean']:8.2f}")

def main():
    print("🏗 Loading model...")
    model = build()
    model.load_weights("path_to_trained_model.h5")

    print("📂 Building gallery/probe sets...")
    gallery_db, probe_db = build_gallery_probe_sets()

    print("🚀 Running benchmark...")
    results = compute_rank1_accuracy(model, gallery_db, probe_db)

    print_results(results)

if __name__ == "__main__":
    main()
