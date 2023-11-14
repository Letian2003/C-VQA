import csv

# formatted response files
files = [
    'C-VQA_vipergpt_results.csv', 
    'C-VQA_visprog_results.csv',
    'C-VQA_instructblip-flant5xxl_results.csv', 
    'C-VQA_instructblip-vicuna7b_results.csv', 
    'C-VQA_instructblip-vicuna13b_results.csv', 
    'C_VQA_LLaVA-7B-v0_results.csv',
    'C_VQA_LLaVA-13B-v0_results.csv', 
    'C-VQA_blip2_flant5_results.csv', 
    'C-VQA_minigpt4_7B_results.csv',
    'C-VQA_LLaVa-v1.5-7b_results.csv',
    'C-VQA_LLaVa-v1.5-13b_results.csv',
    'C-VQA_minigpt4v_results.csv',
    'C-VQA_vipergpt_CodeLlama-34b-Instruct_results.csv',
    'C-VQA_vipergpt_CodeLlama-13b-Instruct_results.csv',
    'C-VQA_vipergpt_CodeLlama-7b-Instruct_results.csv',
    'C-VQA_vipergpt_WizardCoder-15B_results.csv',
    'C-VQA_vipergpt_WizardCoder-Python-13B-V1.0_results.csv',
    'C-VQA_vipergpt_WizardCoder-Python-7B_results.csv',
    ]

for file in files:
    
    data = {}
    with open(file, 'r', encoding = "utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        cnt = 0
        for row in reader:
            if row['type'] not in data:
                data[row['type']] = []
            data[row['type']].append(row)

    print("----------------------")
    print(file)
    for type_category, rows in data.items():
        print(f"Type: {type_category}")
        acc_diff = 0
        correct_count = 0
        total_count = len(rows)
        for row in rows:
            if row['response'] == row['answer']:
                correct_count += 1
        ori_correct_ratio = correct_count / total_count * 100
        diff_ratio = 1 - ori_correct_ratio


        print(f"ori acc: {ori_correct_ratio:.1f}")
        acc_diff = ori_correct_ratio
        
        correct_count = 0
        total_count = len(rows)
        for row in rows:
            if row['new_response'] == row['new answer']:
                correct_count += 1
        cf_correct_ratio = correct_count / total_count * 100
        diff_ratio = 1 - cf_correct_ratio
        
        print(f"cf acc: {cf_correct_ratio:.1f}")
        acc_diff -= cf_correct_ratio
        print(f"diff(ori_acc - cf_acc): {acc_diff:.1f}")
        print()