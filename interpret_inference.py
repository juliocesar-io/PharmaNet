import torch

# TIP: You can inspect a pth file with https://netron.app/ online.

# 1) Load the file (returns a dictionary because you saved it that way):
checkpoint = torch.load("PharmaNet/Best_Config/model1/test_predictions.pth")

# 2) Extract the 'prediction' entry:
predictions_map = checkpoint['prediction']

# 3) Since predictions_map = [smiles, maps, lab], unpack the three elements:
smiles = predictions_map[0]  # np.array of SMILES strings
maps   = predictions_map[1]  # np.array of model probabilities (N x num_classes)
lab    = predictions_map[2]  # np.array of ground-truth class labels

print("SMILES array shape:", smiles.shape)
print("Probability array shape:", maps.shape)
print("Labels array shape:", lab.shape)

# 4) Look at the contents:
for i in range(60):  # just print 5 entries as an example
    print("Sample #", i)
    print("  SMILES:   ", smiles[i])
    print("  ProbDist: ", maps[i])  # array of length = num_classes
     # Convert each probability to percentage with 1 decimal place
    # percentage_probs = [f"{p*100:.5f}%" for p in maps[i]]
    # print("  ProbDist %: ", percentage_probs)
    print("  GT Label: ", lab[i])