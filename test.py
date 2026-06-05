# Testing, save results as .csv file with imgID | class
import csv
import torch

def test_model(loader, model1, model2, device, outfile='submission.csv'):
    with open(outfile, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'Label']) # HEADER
        with torch.no_grad():
            for input, id in loader:
                input = input.to(device)
                id = id[0]
                output1 = model1(input)
                output2 = model2(input)
                avg_prediction = 0.5 * torch.argmax(output1+output2, dim=1).item()
                writer.writerow([id, avg_prediction])

print("Completed")
