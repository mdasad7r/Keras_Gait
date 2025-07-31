import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import SparseCategoricalAccuracy, SparseTopKCategoricalAccuracy
import pandas as pd
from datetime import datetime
import config
from model.cnn_tkan import build_model
from dataset.casia_dataset_all import load_casia_dataset

def evaluate_model(model, test_conditions, output_csv):
    """
    Evaluates the model on test conditions and saves results to a CSV file.
    
    Args:
        model: Trained Keras model.
        test_conditions: Dict with keys 'nm', 'bg', 'cl' containing (X, y) tuples.
        output_csv: Path to save evaluation results.
    """
    # Initialize metrics
    metrics = {
        'condition': [],
        'num_samples': [],
        'accuracy': [],
        'rank_5_accuracy': []
    }
    
    # Define evaluation metrics
    acc_metric = SparseCategoricalAccuracy()
    rank5_metric = SparseTopKCategoricalAccuracy(k=5)
    
    # Evaluate each condition
    for condition in ['nm', 'bg', 'cl']:
        if condition not in test_conditions:
            print(f"⚠️ Warning: No data for condition {condition}")
            continue
            
        X_test, y_test = test_conditions[condition]
        if X_test.shape[0] == 0:
            print(f"⚠️ Warning: Empty dataset for condition {condition}")
            continue
            
        # Evaluate model
        loss, acc, rank5_acc = model.evaluate(
            X_test, y_test,
            batch_size=config.BATCH_SIZE,
            verbose=0
        )
        
        # Store results
        metrics['condition'].append(condition.upper())
        metrics['num_samples'].append(X_test.shape[0])
        metrics['accuracy'].append(acc)
        metrics['rank_5_accuracy'].append(rank5_acc)
        
        print(f"✅ {condition.upper()} - Samples: {X_test.shape[0]}, "
              f"Accuracy: {acc:.4f}, Rank-5 Accuracy: {rank5_acc:.4f}")
    
    # Evaluate combined test set
    X_all = np.concatenate([test_conditions[cond][0] for cond in test_conditions], axis=0)
    y_all = np.concatenate([test_conditions[cond][1] for cond in test_conditions], axis=0)
    
    loss, acc, rank5_acc = model.evaluate(
        X_all, y_all,
        batch_size=config.BATCH_SIZE,
        verbose=0
    )
    
    metrics['condition'].append('ALL')
    metrics['num_samples'].append(X_all.shape[0])
    metrics['accuracy'].append(acc)
    metrics['rank_5_accuracy'].append(rank5_acc)
    
    print(f"✅ ALL - Samples: {X_all.shape[0]}, "
          f"Accuracy: {acc:.4f}, Rank-5 Accuracy: {rank5_acc:.4f}")
    
    # Save results to CSV
    df = pd.DataFrame(metrics)
    df.to_csv(output_csv, index=False)
    print(f"📊 Results saved to {output_csv}")

def main():
    # Load test data
    train_conditions = ["nm-01", "nm-02", "nm-03", "nm-04", "bg-01", "cl-01"]
    _, _, test_conditions = load_casia_dataset(train_conditions=train_conditions)
    print(f"📂 Test conditions loaded: {list(test_conditions.keys())}")
    
    # Build model
    model = build_model()
    
    # Manually specify checkpoint path
    checkpoint_path = "/content/Keras_Gait/casia-b/checkpoints/epoch_50.keras"
    if os.path.exists(checkpoint_path):
        print(f"🔄 Loading model from checkpoint: {checkpoint_path}")
        model.load_weights(checkpoint_path)
    else:
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")
    
    # Compile model with metrics
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[
            SparseCategoricalAccuracy(),
            SparseTopKCategoricalAccuracy(k=5, name='rank_5_accuracy')
        ]
    )
    
    # Output CSV path
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_csv = os.path.join(config.LOG_DIR, f"test_results_{timestamp}.csv")
    
    # Evaluate model
    evaluate_model(model, test_conditions, output_csv)

if __name__ == "__main__":
    main()
