# Functions called for Training
import os, random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import csv
from load_data import TransformsAugments

class Trainer():
    def __init__(self, optimizer, criterion, scheduler, transformer, device, batchsize, numworkers):
        self.optimizer=optimizer
        self.criterion=criterion
        self.scheduler=scheduler
        self.transformer = transformer
        self.device=device
        self.batch_size = batchsize
        self.num_workers = numworkers

    def accuracy(self, loader, model):
        model.eval()
        correct = 0
        total = 0
        for data, label in loader:
            data, label = data.to(self.device), label.to(self.device)
            output = model(data)
            prediction=output.argmax(1)
            total += label.size(0)
            correct += prediction.eq(label.view_as(prediction)).sum().item()

        return 100 * correct / total

    def run_one_epoch(self, loader, model, epoch):
        model.train()
        total_loss = 0.0

        for data, label in loader:
            # Applying Cutmix/Mixup 50% of the time
            if torch.rand(1).item() > 0.7:
                data, label = self.transformer.cutmix_or_mixup(data, label)

            # Train loop
            data, label = data.to(self.device), label.to(self.device)
            self.optimizer.zero_grad() # reset optimizer gradients
            output = model(data)
            loss = self.criterion(output, label)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        acc = self.accuracy(loader, model)
        return total_loss / len(loader), acc

    def validate(self, loader, model):
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for data, label in loader:
                data, label = data.to(self.device), label.to(self.device)
                output = model(data)
                loss = self.criterion(output, label)
                total_loss += loss.item()

        acc = self.accuracy(loader, model)
        return total_loss / len(loader), acc

    def generate_labels(self, loader, model):
        model.eval()
        labels = []
        with torch.no_grad():
            for data, _ in loader:
                data = data.to(self.device)
                output = model(data)
                prediction = torch.argmax(output, dim=1)
                labels.append(prediction.cpu())
        
        return torch.cat(labels)
                

    # Training Functions
    def one_fold(self, fold, model, num_epochs, ckpt_path, csv_path, train_data, train_loader, val_loader, trainer, transformer, starting_epoch=0):
        results = []
        best_val_acc = 0.0
        best_epoch = -1
        no_improvement = 0

        for epoch in range(starting_epoch, starting_epoch + num_epochs):

            if epoch >= starting_epoch + 5:
                train_data.transform =  transformer.get_transforms(resize=224, magnitude=9)
                train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
            elif epoch >= starting_epoch + 10:
                train_data.transform =   transformer.get_transforms(resize=288, magnitude=12)

            train_loss, train_acc = trainer.run_one_epoch(loader=train_loader, model=model, epoch=epoch)
            val_loss, val_acc = trainer.validate(loader=val_loader, model=model)

            results.append(
                {
                    "epoch": int(epoch),
                    "train loss": round(float(train_loss), 4),
                    "train acc": round(float(train_acc), 4),
                    "val loss": round(float(val_loss), 4),
                    "val acc": round(float(val_acc), 4),
                }
            )
            print(f"Epoch ({epoch})- Train Loss: {train_loss}, Train Acc: {train_acc}, Val Loss: {val_loss}, Val Acc: {val_acc}")

            if best_val_acc < val_acc:
                no_improvement = 0
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save({'model_state_dict':model.state_dict(), 'optim_state_dict':trainer.optimizer.state_dict(), 'scheduler_state_dict':trainer.scheduler.state_dict(), 'epoch':epoch}, ckpt_path + "fold" + str(fold))
            else:
                no_improvement += 1
                if no_improvement > 4:
                    print("Early stopping")
                    break

            trainer.scheduler.step()
            torch.cuda.empty_cache()

        with open(csv_path + "fold" + str(fold) + ".csv", "w") as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'train loss', 'train acc', 'val loss', 'val acc'])
            writer.writeheader()
            writer.writerows(results)

        return best_val_acc

    # Fine tune loop
    def fine_tuning(self, model, num_epochs, ckpt_path, csv_path, train_data, train_loader, val_loader, last_acc, trainer, transformer, starting_epoch=0, learning_rate=0.0001):
        # unfreezing layers for finetuning
        results = []
        best_val_acc = last_acc
        best_epoch = starting_epoch
        no_improvement = 0
        # Unfreeze entire feature extraction
        for param in model.base_model.features.parameters():
            param.require_gradient=True
        # update optimizer with new parameters
        trainer.optimizer.add_param_group({'params': model.base_model.features.parameters(), 'lr': 0.001})

        # Update transforms for dataset:
        train_data.transform =  transformer.get_transforms(resize=384, magnitude=14)
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        # Unfreeze a few layers at a time
        # frozen_params = list(filter(lambda p: not p.requires_grad, model.parameters()))
        for epoch in range(starting_epoch, starting_epoch + num_epochs):

            train_loss, train_acc = trainer.run_one_epoch(loader=train_loader, model=model, epoch=epoch)
            val_loss, val_acc = trainer.validate(loader=val_loader,model=model)

            results.append(
                {
                    "epoch": int(epoch),
                    "train loss": round(float(train_loss), 4),
                    "train acc": round(float(train_acc), 4),
                    "val loss": round(float(val_loss), 4),
                    "val acc": round(float(val_acc), 4),
                }
            )
            print(f"Epoch ({epoch})- Train Loss: {train_loss}, Train Acc: {train_acc}, Val Loss: {val_loss}, Val Acc: {val_acc}")

            if best_val_acc < val_acc:
                no_improvement = 0
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save({'model_state_dict':model.state_dict(), 'epoch':epoch}, ckpt_path + "finetuned")
            else:
                no_improvement += 1
                if no_improvement > 7:
                    print("Early stopping")
                    break

            torch.cuda.empty_cache()

        with open(csv_path + "finetuned.csv", "w") as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'train loss', 'train acc', 'val loss', 'val acc'])
            writer.writeheader()
            writer.writerows(results)
        return best_val_acc