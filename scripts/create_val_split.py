import argparse
import random
import shutil
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Create a stratified validation split.")
    parser.add_argument("--train-path", type=str, required=True, help="Path to the training data directory containing subdirectories.")
    parser.add_argument("--val-path", type=str, required=True, help="Path to the validation data directory to be created.")
    parser.add_argument("--percent", type=int, default=1, help="Percentage of files to move to validation (default: 1).")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for shuffling (default: 123).")
    
    args = parser.parse_args()
    
    train_path = Path(args.train_path)
    val_path = Path(args.val_path)
    
    if not train_path.exists() or not train_path.is_dir():
        print(f"Error: --train-path '{train_path}' does not exist or is not a directory.")
        sys.exit(1)
        
    random.seed(args.seed)
    
    moved_total = 0
    # Iterate over subdirectories in train_path
    for subdir in sorted(train_path.iterdir()):
        if subdir.is_dir():
            files = sorted([f for f in subdir.iterdir() if f.is_file()])
            
            if not files:
                continue
                
            # Randomly shuffle files
            random.shuffle(files)
            
            # Calculate number of files to move. 
            # We use max(1, ...) to ensure at least one file is moved for each category, 
            # as long as the category isn't empty.
            num_to_move = max(1, int(len(files) * (args.percent / 100.0)))
            
            # Create corresponding directory in val_path
            val_subdir = val_path / subdir.name
            val_subdir.mkdir(parents=True, exist_ok=True)
            
            files_to_move = files[:num_to_move]
            
            print(f"Moving {len(files_to_move)} out of {len(files)} files from '{subdir.name}' to validation split...")
            
            for f in files_to_move:
                # Move the file
                shutil.move(str(f), str(val_subdir / f.name))
                
            moved_total += len(files_to_move)
                
    print(f"Done! Moved {moved_total} files in total.")

if __name__ == "__main__":
    main()
