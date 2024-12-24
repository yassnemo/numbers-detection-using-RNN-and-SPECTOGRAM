import argparse
import os
from src.train.train import train

def main():
    parser = argparse.ArgumentParser(description='Train digit voice recognition model')
    parser.add_argument('--data_dir', type=str, required=True, 
                      help='Path to dataset directory containing train and val folders')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Learning rate for optimizer')
    
    args = parser.parse_args()
    
    # Validate data directory
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory {args.data_dir} does not exist")
    
    if not all(os.path.exists(os.path.join(args.data_dir, split)) 
              for split in ['train', 'val']):
        raise FileNotFoundError("Data directory must contain 'train' and 'val' subdirectories")
    
    # Start training
    train(
        data_dir=args.data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

if __name__ == '__main__':
    main()