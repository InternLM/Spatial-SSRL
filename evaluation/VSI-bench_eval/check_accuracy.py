import json
import re
def extract(response):
    pattern = r"\\boxed\{([^}]*)\}"
    # 使用 re.findall 查找所有匹配的内容
    matches = re.findall(pattern, response)
    if len(matches) == 0:
        return ''
    return matches[-1]

def critic_multichoice(pred, ans):
    if pred.lower()==ans.lower():
        return 1
    else:
        return 0

def mra(pred, ans):
    try:
        ans_num = float(ans)
        pred_num = float(pred)
        acc = 0
        for i in range(20):
            theta = 0.5 + i * 0.05
            if abs(pred_num-ans_num)/ans_num<1-theta:
                acc += 1
        return acc / 10
    except Exception as e:
        #print(e)
        return 0

path = 'CoT/Spatial-SSRL-7B.jsonl'
cot = True #for CoT version
records = []
cnt_all = 0
score = 0
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        records.append(json.loads(line))

all = {}
cnt_item = {}

for obj in records:
    pred = obj['pred']
    ans = obj['answer']
    score_item = 0
    if cot:
        pred = extract(pred)
    if len(pred) == 1 and pred.isalpha():
        score_item = critic_multichoice(pred, ans) 
        
    else:
        score_item = mra(pred, ans)
    score += score_item
    cnt_all += 1
    if obj['question_type'] not in cnt_item.keys():
        all[obj['question_type']] = 1
        cnt_item[obj['question_type']] = 0
        cnt_item[obj['question_type']] += score_item
    else:
        all[obj['question_type']] += 1
        cnt_item[obj['question_type']] += score_item
print(score, cnt_all, score/cnt_all)
print(all)
print(cnt_item)
result = [cnt_item[k] / all[k] for k in cnt_item]
print(result)
