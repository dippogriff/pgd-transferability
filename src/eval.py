import torch
import torch.nn as nn

is_vlm = lambda x: "LlamaVision" in str(type(x))

def pgd_attack(model, images, labels, epsilon, alpha, num_iter):
    if is_vlm(model):
        raise NotImplementedError("Training attacks on a vlm "
                                  "not yet implemented.")
    orig_images = images.data
    
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(num_iter):
        images.requires_grad = True
        output = model(images)
        
        loss = criterion(output, labels)
        model.zero_grad()
        loss.backward()
        
        # Take step in the direction of the gradient sign
        adv_images = images + alpha * images.grad.sign()
        
        # Project the perturbations to the epsilon ball
        eta = torch.clamp(adv_images - orig_images, min=-epsilon, max=epsilon)
        
        images = torch.clamp(orig_images + eta, min=0, max=1).detach_()
    return images

# Test function (handles both clean and adversarial examples)
def test(models_to_eval, dataloader, dev, model_to_train_attack=None, 
         attack=None, epsilon=0.03, alpha=0.01, num_iter=1):
        
    assert len(models_to_eval) > 0, "At least one model must be provided"
    
    if model_to_train_attack is None:
        if len(models_to_eval) > 1 and attack is not None:
            raise ValueError("model_to_train_attack must be provided"
                             "in case of multiple models to attack")
        # If model_to_train_attack is not provided, use model for eval to train
        else:
            model_to_train_attack = list(models_to_eval.values())[0]
        
    results = {name: {"correct": 0, "total": 0} 
               for name in models_to_eval.keys()}
    
    for name, model in models_to_eval.items():
         
        for images, labels in dataloader:
            images = images.to(dev)
            labels = labels.to(dev)
            
            if attack is not None and attack != "pgd":
                raise ValueError("Currently only pgd attack is supported")
                
            if attack == "pgd":               
                images = pgd_attack(model_to_train_attack, images, 
                                    labels, epsilon, alpha, num_iter)
            with torch.no_grad():
                if is_vlm(model):
                    images = [*images]
                    predicted = model.predict(images)
                else:
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                
            results[name]["total"] += labels.size(0)
            results[name]["correct"] += (predicted == labels).sum().item()   

    # Calculate final accuracies
    accuracies = {name: 100 * results[name]["correct"] / results[name]["total"]
                  for name in results.keys()}
    return accuracies
