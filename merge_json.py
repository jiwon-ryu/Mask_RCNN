'''
    여러번의 라벨링 동안 생성된 json 파일들을 하나의 json 파일로 합친 후 저장
    
    2020-07-24 Jiwon Ryu
'''

import glob
import json

file_path = '/data/JW/merge_json/'    ## json 파일들이 위치한 폴더 쓰기
json_files = glob.glob(file_path+'*.json')
print(json_files)

new_json = dict()

for json_file in json_files:       
    with open(json_file, 'r', encoding='utf-8') as f:
        f = json.load(f)   # type: dict           
        new_json.update(f)   # type: dict
                   
new_json = json.dumps(new_json,indent='\t')   # type: str

with open('json_file.json','w') as outfile:   ## 새로 저장할 파일 이름 설정 (필요시 경로 따로 설정)
    outfile.write(new_json)
