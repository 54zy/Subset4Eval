import csv
import pandas as pd

def read_responses_from_csv(file_path):
    student_responses = {}
    with open(file_path, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            student_id = int(row["student_id"])
            question_id = int(row["question_id"])
            correct = float(row["correct"])
            if student_id not in student_responses:
                student_responses[student_id] = {}
            student_responses[student_id][question_id] = correct
    return student_responses

def process_data(dataset):
    data_path = "../data/" + dataset + "/triples.csv"
    avg_correct_path = "../data/" + dataset + "/triples_avg.csv"
    getavgdata = pd.read_csv(data_path)
    average_scores = getavgdata.groupby("question_id")["correct"].mean().reset_index()
    average_scores.columns = ["question_id", "avg_score"]
    average_scores.to_csv(avg_correct_path, index=False)
    return avg_correct_path
