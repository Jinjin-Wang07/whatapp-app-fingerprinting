"""
Wrapper script for running different machine learning models on PDCP dataset.
Supports SVM, KNN, and MLP models with configurable parameters.
"""

import argparse
import random
import numpy as np
import torch
import os
import sys

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.rawPickleProcessing import *
from utils.dataPrepare import *

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def run_svm(X_train, Y_train, X_test, Y_test, args):
    """Run SVM classifier"""
    print("Running SVM classifier...")
    from SVM import svm_clf
    return svm_clf(X_train, Y_train, X_test, Y_test)

def run_knn(X_train, Y_train, X_test, Y_test, args):
    """Run KNN classifier"""
    print("Running KNN classifier...")
    from KNN_crossval import knn_clf
    return knn_clf(X_train, Y_train, X_test, Y_test)

def run_mlp(X_train, Y_train, X_val, Y_val, X_test, Y_test, args, num_classes):
    """Run MLP classifier"""
    print(f"Running MLP classifier with {args.epochs} epochs...")
    from MLP_withval import mlp_clf
    return mlp_clf(X_train, Y_train, X_val, Y_val, X_test, Y_test, 
                   num_classes=num_classes, epochs=args.epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run machine learning models on PDCP dataset')
    
    # Required arguments
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input folder path containing CSV dataset')
    parser.add_argument('--model', '-m', type=str, required=True,
                       choices=['svm', 'knn', 'mlp', 'all'],
                       help='Model to run: svm, knn, mlp, or all')
    
    # Optional arguments
    parser.add_argument('--sample_duration', type=int, default=2555,
                       help='Sample duration parameter (default: 2555)')
    parser.add_argument('--window_size', type=int, default=60,
                       help='Window size in seconds (default: 60)')
    parser.add_argument('--step', type=int, default=5,
                       help='Step size in seconds (default: 5)')
    parser.add_argument('--epochs', type=int, default=300,
                       help='Number of epochs for MLP training (default: 300)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    args = parser.parse_args()
    
    # Validate input folder
    if not os.path.exists(args.input):
        print(f"Error: Input folder '{args.input}' does not exist!")
        sys.exit(1)
    
    if not os.path.isdir(args.input):
        print(f"Error: '{args.input}' is not a directory!")
        sys.exit(1)
    
    # Fix random seed
    set_seed(42)
    
    if args.verbose:
        print(f"Configuration:")
        print(f"  Input folder: {args.input}")
        print(f"  Model(s): {args.model}")
        print(f"  Sample duration: {args.sample_duration}")
        print(f"  Window size: {args.window_size}")
        print(f"  Step size: {args.step}")
        print(f"  Random seed: {args.seed}")
        if args.model in ['mlp', 'all']:
            print(f"  MLP epochs: {args.epochs}")
        print()
    
    try:
        # Load dataset
        print("Loading CSV dataset...")
        L = load_all_csv_files(args.input)
        
        if args.verbose:
            print(f"Loaded {len(L)} CSV files")
        
        # Extract features
        print("Extracting PDCP features...")
        labels, dic = extract_pdcp_features_from_dfs(L, args.sample_duration, 
                                                    args.window_size, step=args.step)
        
        if args.verbose:
            print(f"Found {len(labels)} unique labels: {labels}")
        
        # Prepare data
        print("Preparing training/validation/test datasets...")
        X_train, Y_train, X_test, Y_test, X_val, Y_val = TransDataApp(labels, dic)
        
        if args.verbose:
            print(f"Training set: {X_train.shape}")
            print(f"Validation set: {X_val.shape}")
            print(f"Test set: {X_test.shape}")
            print()
        
        # Run selected model(s)
        results = {}
        
        if args.model == 'svm' or args.model == 'all':
            results['svm'] = run_svm(X_train, Y_train, X_test, Y_test, args)
            
        if args.model == 'knn' or args.model == 'all':
            results['knn'] = run_knn(X_train, Y_train, X_test, Y_test, args)
            
        if args.model == 'mlp' or args.model == 'all':
            results['mlp'] = run_mlp(X_train, Y_train, X_val, Y_val, X_test, Y_test, 
                                   args, num_classes=len(labels))
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)