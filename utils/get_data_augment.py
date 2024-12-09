import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data

def get_COVID10_dataloader_augment(args):
    
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),  transforms.RandomHorizontalFlip(p=0.5), 
        transforms.RandomRotation(degrees=15), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    train_dataset = datasets.ImageFolder(args.train_root, transform=transform)
    test_dataset = datasets.ImageFolder(args.test_root, transform=transform)
    val_dataset = datasets.ImageFolder(args.val_root, transform=transform)
    
    train_augmented = datasets.ImageFolder(args.train_root, transform=transform)
    
    for i in range(args.num_aug):
        train_augmented += datasets.ImageFolder(args.train_root, transform=transform)
    
    train_dataset = data.ConcatDataset([train_dataset, train_augmented])
    
    train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=args.workers, pin_memory=True)
    test_dataloader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=args.workers, pin_memory=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,num_workers=args.workers, pin_memory=True)
    
    return train_dataloader, test_dataloader, val_dataloader