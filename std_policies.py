import numpy as np

def eps_greeedy_policy(Qs,eps=0.1):
    assert isinstance(Qs, np.ndarray) or isinstance(Qs,list), "Qs should be an array or list"

    pr_action = np.ones_like(Qs)*eps/len(Qs)

    max_occurances = np.argwhere(Qs == np.max(Qs))

    for idx in max_occurances:
        pr_action[idx] += (1-eps)/len(max_occurances)

    action_idx = np.random.choice(len(Qs),p=pr_action)
    return action_idx