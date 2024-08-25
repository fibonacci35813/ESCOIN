import torch 
import torch.nn as nn
import torch.nn.utils.prune as prune

model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)

#the model parameters 
print(model.parameters)

#pruning the convolutational neural networks of the pretrained model 
idx = []
for i,module in enumerate(model.features.modules()):
    if isinstance(module,nn.Conv2d):
        prune.l1_unstructured(module,name="weight",amount=0.8)
        idx.append(i)

print(idx)

#we store the pruned weights in a txt file to be read later by the 
for i in idx:
    weights = model.features[i-1].weight.flatten().tolist()
    bias = model.features[i-1].bias.tolist()
    with open("Conv2D_layer" + str(i) +".txt","w") as f:
        f.write(" ".join(list(map(str,list(model.features[i-1].weight.shape)))))
        f.write("\n")
        f.write(" ".join(list(map(str,bias))))
        f.write("\n")
        f.write(" ".join(list(map(str,weights))))