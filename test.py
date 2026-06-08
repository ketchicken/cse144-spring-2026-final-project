# Testing, save results as .csv file with imgID | class
import csv
import torch

def test_model(loader, model, device, outfile='submission.csv'):
    with open(outfile, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'Label']) # HEADER
        with torch.no_grad():
            for input, id in loader:
                input = input.to(device)
                id = id[0]
                prediction = 0.5 * torch.argmax(model(input), dim=1).item()
                writer.writerow([id, prediction])

print("Completed")
