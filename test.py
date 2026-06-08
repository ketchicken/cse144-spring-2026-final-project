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

def remap_values(file, corrected_file, idx_to_class):

    with open(corrected_file, 'w', newline='') as out_f, open(file, 'r') as in_f:
        reader = csv.reader(in_f)
        writer = csv.writer(out_f)

        writer.writerow(['ID', 'Label']) # header

        all_rows = list(reader)
        for row in all_rows[1:]: # skip the header
            row[0], row[1] = row[0], idx_to_class[row[1]]