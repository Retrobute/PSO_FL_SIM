from os import mkdir , path , makedirs
import json 
import csv 

def save_data_to_csv(cols , rows , csv_path) :
    csv_dir = "/".join(csv_path.split("/")[:-1])
    makedirs(csv_dir , exist_ok=True)

    with open(csv_path, 'w') as csvfile :
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(cols)
        csvwriter.writerows(rows)    

def save_metadata_to_json(data , json_path) :
    csv_dir = "/".join(json_path.split("/")[:-1])
    makedirs(csv_dir , exist_ok=True)  

    with open(json_path, "w") as jsonfile :
        json.dump(data , jsonfile , indent=4)