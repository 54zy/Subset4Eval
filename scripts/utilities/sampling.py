import random

def sample(number, beta_params_list):
    d = 3.0
    n = len(beta_params_list)
    w = []
    for i in range(n):
        tmp = []
        for j in range(n):
            num = d - abs(beta_params_list[i] - beta_params_list[j])
            tmp.append(num)
        w.append(tmp)
    
    s = []
    for item in range(number):
        print(item)
        if item == 0:
            init = random.randint(0, n)
            s.append(init)
        else:
            delta = {}
            for q in range(n):
                if q not in s:
                    random_numbers = random.sample(range(n), 10)
                    F_s = sum(max(w[q_f_s][j] for j in s) for q_f_s in random_numbers)
                    s.append(q)
                    F_s_q = sum(max(w[q_f_s][j] for j in s) for q_f_s in random_numbers)
                    s.pop()
                    delta[q] = F_s_q - F_s
            max_key = max(delta, key=delta.get)
            s.append(max_key)
    return s
