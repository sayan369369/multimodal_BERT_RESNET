import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import BertTokenizer, BertModel

# ---------- LOAD DATA ----------
data = pd.read_csv("data.csv")

# ---------- MODELS ----------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased")           #768 features 

resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet = nn.Sequential(*list(resnet.children())[:-1])           #2048 features 
resnet.eval()

classifier = nn.Linear(2816, 2)                                 #input = (768 + 2048) = 2816 features.    #output = 2 classes 

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

transform = transforms.Compose([
    transforms.Resize((224,224)),                               #image preprocessing 
    transforms.ToTensor()
])

# ---------- TRAIN LOOP ----------
for epoch in range(10):                        #10 epochs 

    total_loss = 0

    for i in range(len(data)):

        text = data.loc[i,"text"]
        label = torch.tensor([data.loc[i,"label"]])

        # TEXT FEATURE
        inputs = tokenizer(text, return_tensors="pt")
        outputs = bert(**inputs)
        text_feat = outputs.last_hidden_state[:,0,:]

        # IMAGE FEATURE
        img = Image.open("images/"+data.loc[i,"image"]).convert("RGB")
        img = transform(img).unsqueeze(0)
        img_feat = resnet(img).view(1,-1)

        # FUSION
        fused = torch.cat((text_feat, img_feat), dim=1)

        # PREDICTION
        out = classifier(fused)                     #linear classifier (2 classes)

        loss = criterion(out, label)

        optimizer.zero_grad()
        loss.backward()                             #backpropagation
        optimizer.step()                            #weight_update 

        total_loss += loss.item()                   #total loss value 

    print("Epoch:", epoch, "Loss:", total_loss)





    print("\nTesting model...")

correct = 0

for i in range(len(data)):

    text = data.loc[i,"text"]
    label = data.loc[i,"label"]

    inputs = tokenizer(text, return_tensors="pt")    #
    outputs = bert(**inputs)                         # text to features
    text_feat = outputs.last_hidden_state[:,0,:]     #    

    img = Image.open("images/"+data.loc[i,"image"]).convert("RGB")    #
    img = transform(img).unsqueeze(0)                                 # image to features 
    img_feat = resnet(img).view(1,-1)                                 #

    fused = torch.cat((text_feat, img_feat), dim=1)      # text feature + image features

    out = classifier(fused)                    #prediction scores 

    pred = torch.argmax(out).item()            #final class 

    print("Actual:", label, "Predicted:", pred)

    if pred == label:
        correct += 1

accuracy = correct / len(data)       #correct = 4 and total length = 4 #accuracy = 1.0

print("\nAccuracy:", accuracy)       
