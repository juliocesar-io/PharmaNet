import argparse

def get_args():

        ap = argparse.ArgumentParser()
        
        ap.add_argument("-crval", "--cross_val",
                        type=int,default=1,
                        help="Number of the four fold cross validation model")

        ap.add_argument("-trpa", "--train_path", 
                        default='./data/datasets/DUDE/', 
                        help="Train Data Path ")

        ap.add_argument("-tef", "--test_file", 
                        default='./data/datasets/DUDE/Smiles_Test.csv', 
                        help="Train Data Path ")

        ap.add_argument("-rd", "--results_dir", 
                        default = './PharmaNet', 
                        help="Path for Saving Results")

        ap.add_argument("-rp", "--results_path", 
                        default = 'Best_Config', 
                        help="Experiment name")

        ap.add_argument("-md", "--model", 
                        type= int, default = 5, 
                        help="Choose a Model for the net")

        ap.add_argument("-bd", "--bidireccional", 
                        type= bool, default = True, 
                        help="RNN direction")

        ap.add_argument("-nl", "--num_layers", 
                        type= int, default=10, 
                        help="RNN depth")

        ap.add_argument("-bs", "--batch_size", 
                        type=int, default = 128, 
                        help="Batch Size")

        ap.add_argument("-hs", "--hidden_size", 
                        type=int, default = 256, 
                        help="Size of the Hidden State")

        ap.add_argument("-e", "--epochs", 
                        type=int, default = 30, 
                        help="Number of Epochs")

        ap.add_argument("-av", "--add_val", 
                        type=int, default = 0, 
                        help="Value to add in the embedding")

        ap.add_argument("-nb", "--neighbours", 
                        type=int, default = 0, 
                        help="Consider neighbours in one hot embedding")

        ap.add_argument("-ng", "--ngpu", 
                        type=int, default = 4, 
                        help="Available GPU(s)")

        ap.add_argument("-sv", "--save", 
                        type=bool, default=True, 
                        help="Save the parameters and results of the experiment")

        ap.add_argument("-ks", "--kernel_size", 
                        type=int, default=5, 
                        help="Kernel Size")

        ap.add_argument("-pd", "--padding", 
                        type=bool, default=False, 
                        help="Change type of padding")

        ap.add_argument("-lr", "--learning_rate", 
                        type=float, default=0.0005, 
                        help="Learning Rate")
                        
        ap.add_argument("-sd", "--seed", 
                        type=int, default=1, 
                        help="Random seed")

        ap.add_argument("-bl", "--balanced_loader", 
                        type = bool, default=False, 
                        help="Balanced weights in the dataloader")

        ap.add_argument("-cp", "--checkpoint", 
                        type = bool, default=False, 
                        help="Load best model")

        ap.add_argument("-wo", "--workers", 
                        type = int, default=4, 
                        help="Number of workers for the Dataloader")
        
        return ap.parse_args()
