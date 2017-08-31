# coding: utf-8
import sys
import importlib
importlib.reload(sys)
import re
import jieba
from jieba import analyse
import string
import config
def segment(input, output):
    input_file = open(input, "rb")
    output_file = open(output, "w")
    line_num = 1
    segments = ""
    read_num = 0
    total_num = 0
    last_line = ""
    last_last_line = ""
    while True:
        line = input_file.readline()
        
        if line:
            read_num += 1
            if read_num > 10000:
                break
            try:
                line = str(line, encoding = "utf-8")
            except:
                continue
            line = line.strip('\n').strip()
            #test = line.decode('utf-8')
            err_flag = 0
            
            #如果上下两句话相同,则删除
            err_char = []
            for uchar in line:
                '''
                if (uchar >= u'\u4e00' and uchar<=u'\u9fa5') or (uchar >= u'\u0030' and uchar<=u'\u0039') or (uchar >= u'\u0041' and uchar<=u'\u005a') or (uchar >= u'\u0061' and uchar<=u'\u0080') or (uchar >= u'\u2000' and uchar<=u'\u206f') or (uchar >= u'\uff00' and uchar<=u'\uff5e'):
                    err_flag = err_flag
                else:
                    err_flag = 1
                '''
                #(uchar >= u'\u2027' and uchar<=u'\u27ff')
                #(uchar in [u'\u2764',u'\u2027',u'\u266a'])  (uchar >= u'\u2000' and uchar<=u'\u206f') 
                if (uchar >= u'\u2027' and uchar<u'\u3000') or (uchar >= u'\u3016' and uchar<u'\u4e00') or (uchar > u'\u0080' and uchar<u'\u2010') or (uchar >= u'\ua000' and uchar <= u'\uff00') or (uchar >= u'\uff5f') or (uchar == u'\u0020') or (uchar in [u'\u2022',u'\u200b',u'\u2006']):
                    #err_flag = 1
                    err_char.append(uchar)
            #print(line)
            for uchar in err_char:
                line = line.replace(uchar,"")
            line = line.replace("|","")
            #print(line)
            #highpoints = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')  
            #line = highpoints.sub(u'', line) 
        else:
            break
        if total_num > 0:
            segments = last_line+"|"+line+"\n"
            if (last_last_line==last_line) and (last_line==line):
                #print("repeate 3 times")
                #print(last_last_line)
                #print(last_line)
                #print(line)
                continue
            #print(segments)
            output_file.write(segments)
        total_num += 1
        last_last_line = last_line
        last_line = line
    print("read_num:"+str(read_num))
    print("total_num:"+str(total_num))
    input_file.close()
    output_file.close()

if __name__ == '__main__':
    
    if 3 != len(sys.argv):
        print("Usage: ", sys.argv[0], "input output")
        sys.exit(-1)
    segment(config.DATA_DIR+sys.argv[1], config.DATA_DIR+sys.argv[2]);
    