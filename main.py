import torch
import pandas as pd
import numpy as np
from scripts.utilities.sampling import sample
from scripts.utilities.irt import one_parameter_irt
from scripts.utilities.data_processing import read_responses_from_csv, process_data
from scripts.utilities.evaluation import estimate_student_ability, mean_squared_error

def main():
    dataset = 'math'
    student_ids = range(0, 60)
    
    # Load data
    data_path = "../data/" + dataset + "/triples.csv"
    dict_path = "../scripts/model/" + dataset + "/beta_params.pth"
    
    state_dict = torch.load(dict_path)
    beta_params = state_dict["beta.weight"].squeeze().tolist()
    response = read_responses_from_csv(data_path)
    
    store = sample(dataset, beta_params)
    
    df = pd.read_csv(data_path)
    data_length = len(store)
    repeat = 1
    times = 1
    save = []
    ability = {}
    
    for i, student_id in enumerate(student_ids):
        print(student_id)
        student_data = df[df["student_id"] == student_id]
        theta_guess = np.random.normal(-0.5, 0.5, size=1)[0]
        for re in range(repeat):
            test = []
            acc_bec = []
            correct = []
            true = student_data["correct"].mean()
            x_arange = []
            for j in range(1 * times, 100 * times):
                if j % 5 == 0:
                    index = int(float((j / (100 * times))) * data_length)
                    estimated_ability = estimate_student_ability(
                        response, store[:index], student_id, beta_params, theta_guess
                    )
                    theta_guess = estimated_ability
                    x_arange.append(j / times)

                    pre = []
                    cor = 0
                    blank = 0
                    for id in range(0, index):
                        qid = store[id]
                        if qid >= len(response[student_id]):
                            blank += 1
                            continue
                        label = response[student_id][qid]
                        cor += label
                    cor = cor / (index - blank)
                    for id in range(index, data_length):
                        qid = store[id]
                        beta = beta_params[qid]
                        predicted_prob = one_parameter_irt(estimated_ability, beta)
                        pre.append(predicted_prob)
                    acc = sum(pre) / (data_length - index)
                    lamd = float(index / len(response[student_id]))
                    res = lamd * cor + (1 - lamd) * acc
                    acc_bec.append(res)
                    correct.append(mean_squared_error(true, res))
            ability[student_id] = acc_bec
            test.append(correct)
            tuple_data = (student_id, re, acc_bec)
            save.append(tuple_data)
    
    plbe = []
    for index in range(len(ability[student_ids[0]])):
        result = [value[index] for value in ability.values()]
        result = sorted(range(len(result)), key=lambda i: result[i])
        plbe.append(result)
    
    print(plbe)

if __name__ == "__main__":
    main()
