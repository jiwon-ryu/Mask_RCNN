'''
    동일 폴더 내에서 jpg 리스트와 json 파일 리스트 중 일치하지 않는 파일이 있는지 확인하는 코드
    FAIL이 존재하면 어떤 이미지인지 출력함
    해당 이미지를 폴더에서 삭제하거나 JSON 파일에서 지우면 됨
    
    path = '/data/JW/temp_dataset/test/' 부분 적절히 수정
    
    2020-07-28 Jiwon Ryu
'''

import json
import glob

path = '/data/JW/Mask_RCNN/dataset/test/'   ## jpg, json 위치 폴더
json_file = path+'json_file.json'
jpg_file = glob.glob(path+'*.jpg')
jpg_capital_file = glob.glob(path+'*.JPG')   ## capital .JPG 파일 존재할 경우를 대비한 코드

for capital in jpg_capital_file:
    jpg_file.append(capital)

jpg_list = []
for jpg in jpg_file:
    jpg_name = jpg[jpg.rfind('/', )+1 : len(jpg)]
    jpg_list.append(jpg_name)
    
with open(json_file, 'r', encoding='utf-8') as json_file:
    json_file = json.load(json_file)
    
    ### json에 있는 파일이 jpg에도 있는지 확인
    success = 0
    fail = 0
    json_list = []
    print('-------IS JSON IN JPG LIST?-------')
    for key in json_file.keys():                      
        img_name = json_file[key]['filename']
        json_list.append(img_name)       
                    
        if img_name in jpg_list:
            success += 1
        else:
            fail += 1
            print('error:'+img_name)
        
    print('success: ', success)
    print('fail: ', fail)

    ### jpg에 있는 파일이 json에도 있는지 확인
    print('--------IS JPG IN JSON LIST?-------')
    success = 0
    fail = 0    
    for jpg in jpg_list:
        if jpg in json_list:
            success += 1
        else:
            fail += 1
            print('error:'+jpg)
    
    print('success: ', success)
    print('fail: ', fail)
            


