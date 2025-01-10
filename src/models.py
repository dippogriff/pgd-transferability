import torch
import torch.nn as nn
import torchvision.models as models
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        self.conv_layer = nn.Sequential( 
            nn.Conv2d(3, 4, 3), # 30x30x4
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 15x15x4
            
            
            nn.Conv2d(4, 16, 4), # 12x12x16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 6x6x16
            
            nn.Conv2d(16, 32, 3), # 4x4x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 2x2x32
        )
        
        self.fc_layer = nn.Sequential(
            nn.Linear(2*2*32, 56), 
            nn.BatchNorm1d(56),
            nn.ReLU(),
            nn.Linear(56, num_classes),
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = torch.flatten(x, 1)
        x = self.fc_layer(x)
        return x

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
    
class VitB16(nn.Module):
    def __init__(self, num_classes=10):
        super(VitB16, self).__init__()
        self.model = models.vit_b_16(
            weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        num_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

class ViT(nn.Module):
    def __init__(self, num_classes=10, image_size=32):
        super(ViT, self).__init__()
        self.model = models.VisionTransformer(
            image_size=image_size, patch_size=4, num_layers=6, num_heads=4, 
            hidden_dim=32, mlp_dim=32*4, num_classes=num_classes)
        
    def forward(self, x):
        return self.model(x)
    
class LlamaVision(nn.Module):
    def __init__(self, classes):
        super(LlamaVision, self).__init__()
        model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.classes = classes
        prompt = f"""
            Use exactly one word for the output. 
            Classify the image into one of the following labels: 
            {", ".join(self.classes)}.
        """
            
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]
        
        self.input_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True)
        
    def predict(self, images):
        if len(images) == 1:
            images = images[0]
            prompts = self.input_text
            batch_decode = False
        else:
            prompts = [self.input_text]*len(images)
            batch_decode = True
            
        inputs = self.processor(
            images,
            prompts,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True
        ).to(self.model.device)
        
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=10)
        
        if batch_decode:
            answers = self.processor.batch_decode(output)
        else:
            answers = [self.processor.decode(output[0])]
            
        answers = [a.split('<|end_header_id|>')[-1].split('<|eot_id|>')[0]
                   for a in answers]
        
        preds = [-1]*len(answers)
        for i, answer in enumerate(answers):
            for l, label in enumerate(self.classes):
                if label in answer:
                    preds[i] = l
                    break
        return torch.Tensor(preds).to(self.model.device)
            
    def forward(x):
        raise NotImplementedError()