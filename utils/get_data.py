import os
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms


def get_COVID10_dataloader(args):
    # Define the transforms to be applied to each image
    transform = transforms.Compose([
        transforms.Resize(224), # Resize the image to 224x224 pixels
        transforms.CenterCrop(224), # Crop the image at the center to make it a square image
        transforms.ToTensor(), # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize the image
    ])

    # Define the paths to the train, test and validation folders
    

    # Load the datasets using the ImageFolder class
    train_dataset = ImageFolder(args.train_root, transform=transform)
    test_dataset = ImageFolder(args.test_root, transform=transform)
#     val_dataset = test_dataset #ImageFolder(args.val_root, transform=transform)
    
    val_dataset = ImageFolder(args.val_root, transform=transform)

    # Define the dataloaders for each dataset
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                                   num_workers=args.workers, pin_memory=True)
    
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                                                  num_workers=args.workers, pin_memory=True)
    
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.workers, pin_memory=True)


    return train_dataloader, test_dataloader, val_dataloader 


if __name__ == '__main__':

    train_path = ""
    test_path = ""
    val_path = ""
    
    _,_,_ = get_COVID10_dataloader(train_path, test_path, val_path)