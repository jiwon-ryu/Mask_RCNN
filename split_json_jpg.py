'''
    jpg 파일을 train, val, test로 랜덤하게 나눠줌
    & 하나로 합친 json 파일에서 train, val, test jpg에 대응되는 것 추출함
    
    1. jpg_list, jpg_capital_list 경로 수정 (이미지 들어있는 폴더로)
    2. train_path, val_path, test_path 경로 수정 (split 한다음에 저장할 경로)
    3. ratio 정해주기
    4. json_file 변수에 하나로 합쳐져있는 json 파일 경로 쓰기

    cf. train, test, val 모두 json 파일 이름 동일하게 설정하기
    cf. 여기서 설정한 json 파일 이름을 cucumber.py (training 코드)에도 그대로 써야함
    
    2020-07-24 Jiwon Ryu
'''

import json
import glob
import shutil
import os
import random

# split and move jpg files into train/val/test folders
jpg_list = glob.glob('/data/JW/temp_dataset/*.jpg')   ## 현재 jpg 모아 놓은 경로 쓰기
jpg_capital_list = glob.glob('/data/JW/temp_dataset/*.JPG')   ## capital JPG 파일이 존재하는 경우를 대비한 코드

for capital in jpg_capital_list:
    jpg_list.append(capital)
    
random.shuffle(jpg_list)

train_path = '/data/JW/temp_dataset/train/'   ## train 폴더 쓰기
val_path = '/data/JW/temp_dataset/val/'   ## val 폴더 쓰기
test_path = '/data/JW/temp_dataset/test/'   ## test 폴더 쓰기

ratio = [2, 1, 1]   ## train, val, test 비율 정해주기 (현재 2:1:1)

train_jpg = []
val_jpg = []
test_jpg = []
count = 1
for jpg in jpg_list:
    jpg_name = jpg[jpg.rfind('/', )+1 : len(jpg)]
    
    if count < len(jpg_list) * (ratio[0] / sum(ratio)):
        train_jpg.append(jpg_name)
        shutil.move(jpg, train_path+jpg_name)
    elif len(jpg_list) * (ratio[0] / sum(ratio)) <= count < len(jpg_list) * ((ratio[0]+ratio[1]) / sum(ratio)):
        val_jpg.append(jpg_name)
        shutil.move(jpg, val_path+jpg_name)
    else:
        test_jpg.append(jpg_name)
        shutil.move(jpg, test_path+jpg_name)
    count += 1

print('---Number of train, val, test (image)---\n', len(train_jpg), len(val_jpg), len(test_jpg))   ## train, val, test 폴더에 들어간 이미지 개수   


# split json into corresponding images
json_file = '/data/JW/temp_dataset/json_file.json'

train_json = dict()
val_json = dict()
test_json = dict()
with open(json_file, 'r', encoding='utf-8') as json_file:
    json_file = json.load(json_file)   
    for key in json_file.keys():           
        img_name = json_file[key]['filename']
        
        if img_name in train_jpg:
            train_json.update({key: json_file[key]})
        elif img_name in val_jpg:
            val_json.update({key: json_file[key]})
        elif img_name in test_jpg:
            test_json.update({key: json_file[key]})

train_json = json.dumps(train_json,indent='\t')
val_json = json.dumps(val_json,indent='\t')
test_json = json.dumps(test_json,indent='\t')

with open(train_path+'json_file.json','w') as outfile:
    outfile.write(train_json)
with open(val_path+'json_file.json','w') as outfile:
    outfile.write(val_json)
with open(test_path+'json_file.json','w') as outfile:
    outfile.write(test_json)

