import csv
import re
import os

# raw response files
input_files = [
    'llava_v15_7b_formatting_responses.csv',
    'llava_v15_13b_formatting_responses.csv',
    'qwen_responses.csv'
    ]
# formatted response files
output_files = [
    'llava_v15_7b_formatting_results.csv',
    'llava_v15_13b_formatting_results.csv',
    'qwen_results.csv'
    ]

def get_numbers(numbers):
    for number in numbers:
        if number.isdigit():
            number = int(number)
            return number
        elif number in {'no','zero','none', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'}:
            number_dict = {'no':0,'zero':0,'none':0,'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}
            number = number_dict[number]
            return number
    return "fail"
    
def get_bool(bools):
    
    if 'no' in bools or 'false' in bools or 'not' in bools:
        return 'no'
    elif 'yes' in bools or 'true' in bools or 'be' in bools or 'is' in bools or 'are' in bools:
        return 'yes'
    else:
        return 'fail'
    
for input_file,output_file in zip(input_files,output_files):
    if not input_file.endswith('.csv'):
        continue

    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', newline='', encoding='utf-8') as f_out:
        print("start")
        
        reader = csv.DictReader(f_in)
        fieldnames = ['img_path', 'query', 'answer', 'new query', 'new answer', 'type', 'response', 'new_response']
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        cnt = 0
        for row in reader:
            cnt+=1
            if row['type'] != "boolean":
                
                response = row['response']
                
                words = re.findall(r'(\b\w+\b)|\,', response.lower())
                
                if len(words) == 0:
                    words = ["-1"]
                elif words[0] == 'if':
                    if '' in words:
                        index = words.index('')
                        words = words[index+1:]
                    else:
                        words = ["-1"]
                elif 'if' in words:
                    index = words.index('if')
                    words = words[:index]
            
                res1 = get_numbers(words)

                response = row['new_response']
                
                words = re.findall(r'(\b\w+\b)|\,', response.lower())
                
                if len(words) == 0:
                    words = ["-1"]
                elif words[0] == 'if':
                    if '' in words:
                        index = words.index('')
                        words = words[index+1:]
                    else:
                        words = ["-1"]
                elif 'if' in words:
                    index = words.index('if')
                    words = words[:index]
                
                res2 = get_numbers(words)



                writer.writerow({'img_path': row['img_path'], 'query': row['query'], 'answer': row['answer'], 'new query': row['new query'], 'new answer': row['new answer'], 'type': row['type'], 'response': res1, 'new_response': res2})
            else:
                response = row['response']
                words = re.findall(r'\b\w+\b', response.lower())

                res1 = get_bool(words)
                response = row['new_response']
                words = re.findall(r'\b\w+\b', response.lower())
                res2 = get_bool(words)
                
                writer.writerow({'img_path': row['img_path'], 'query': row['query'], 'answer': row['answer'], 'new query': row['new query'], 'new answer': row['new answer'], 'type': row['type'], 'response': res1, 'new_response': res2})

        print(cnt)