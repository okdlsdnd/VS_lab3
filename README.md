<!--21700150 김인웅-->

Lab3
======

목표 : 도로 주행 영상에서 차선 검출 및 차선 이탈 시 경고문 출력  

기본 영상 상태

![default_vid](https://user-images.githubusercontent.com/80805040/166679975-555b0a6b-cc1f-417d-a4f8-e6ce3ee01f26.png)


위 영상에서 좌우 차선을 검출하는 것으로 시작한다.

### Lane Detection

##### Post Processing

원활한 차선 검출을 위해 전처리를 하였다. 

우선 원본 이미지를 Gray Scale 이미지로 바꾸고 threshold 125로 Binary 이미지로 변환한다.

이후 사용할 차선만 검출하기 위해 사다리꼴 모양의 ROI를 지정하고 Canny를 이용해 윤곽선만 남긴다.

![post_processing](https://user-images.githubusercontent.com/80805040/166872185-904e10f8-8a57-46bd-8bf7-462e50ce9d7d.png)

최종 Canny 이미지에서 HoughLinesP를 이용하여 직선들을 검출한다. 이때 Treshold는 50, minLineLength는 70, maxLineGap은 100으로 지정하였다.

HoughLinesP로 구해진 직선 중 첫번째 array를 대표선으로 사용하며 이 때 기울기의 절대값이 0.6이상인 직선만 사용한다. 그리고 검출된 직선이 영상 중앙으로 기준으로 좌우에 있는지를 판단하여 좌,우 직선 각각 하나씩만 남긴다.

np.polyfit을 이용하여 직선의 식을 찾기 위해 사용될 m과 b 값을 구할 수 있다. 이것으로 검출된 직선을 바탕으로 차선을 모두 잇는 직선을 그린다.



##### Algorithm

선이 검출 될 때 좌우 각각 draw_left와 draw_right라는 parameter가 True가 된다. 반대로 선이 검출 되지 않으면 각 parameter는 False가 된다.

두 선이 모두 검출 될 경우에 그 프레임의 직선 정보를 idx로 저장한다.

저장된 idx를 바탕으로 소실점, 차선의 중앙, bias를 계산한다.

선이 검출되면 화면에는 검출된 선이 파란색으로 그려진다.

선이 검출되지 않으면서 차선 이동중이 아닐 때 이전 프레임에서 검출된 선이 다시 노란색으로 그려진다.

차선이 검출된 상황

![Yes_line](https://user-images.githubusercontent.com/80805040/166874740-7e4657ec-e54f-49b5-bfb4-cb0cdf663258.png)

차선이 검출되지 않은 상황

![No_line](https://user-images.githubusercontent.com/80805040/166874745-62c0ad69-bd2e-4502-ba43-bb9f15925208.png)

두 상황 모두 bias가 17 이하이며 초록색 소실점과 초록색 반투명 삼각형이 나타남을 확인 할 수 있다.



좌, 우 차선이 검출되지 않을 경우 각각 No_left와 No_right에 1씩 더해진다.

좌, 우로 차선을 변경하는 flag는 Left와 Right 이며 이동 중이지 않을 때는 모두 False이다.

이 때 왼쪽 직선이 10프레임 이상 감지되지 않았으며 Left == False 일 때 Right는 True가 된다.

반대로 오른쪽 직선이 10프레임 이상 감지되지 않고 Right == False 일 때 Left는 True가 된다.

Right나 Left가 True이면 차선이 변경 중인 것으로 판단하며 감지된 하나의 차선만을 화면에 표시하고 그 직선의 끝점을 빨간색 원으로 표시하며 차선 이동 전의 마지막 프레임의 삼각형을 붉은색 반투명 삼각형으로 화면에 표시한다. 

또한 Right가 True인 경우를 오른쪽으로 이동한다고 판단하고 화면에 오른쪽 화살표를 표시한다.

반대로 Left가 True인 경우는 왼쪽으로 이동한다고 판단하고 화면에 왼쪽 화살표를 표시한다.

차선 이동중

![Line_Changing](https://user-images.githubusercontent.com/80805040/166874760-f9a7add2-42d3-45ac-91fb-f029d4c947bc.png)

위 이미지는 Left가 True인 경우다.




##### 개선점 

위에서 사용한 알고리즘으로 차선 이동을 검출한다면 차선을 변경 중이 아닌데 차선 중앙에서 벗어난 경우를 판단하지 못한다. 즉, 차선 이동중이 아닌 데도 불구하고 자동차가 차선 중앙에서 벗어난다면 차선 이동 중인것으로 판단할 수도 있다는 것이다. 실제로 영상 마지막 즈음에 그런 경우가 있었으며 이는 개선이 필요하다.



이 값을 이용해서 영상을 출력한다. 아래에 영상 링크를 첨부한다.  

https://youtu.be/YhwjUeVS3us

### Appendix

##### 코드

    from pickle import FALSE
    from turtle import Turtle
    import numpy as np
    import cv2 as cv
    from cv2 import *
    from cv2.cv2 import *
    import time
    import copy
    
    from matplotlib import pyplot as plt
    
    cap = cv.VideoCapture('road_lanechange_student.mp4')
    
    prevTime = 0
    
    No_left = 0
    No_right = 0
    
    draw_right = True
    draw_left = True
    
    Right = False
    Left = False
    
    while True:
        timeLoopStart = cv.getTickCount()
        ret, img = cap.read()
    
        curTime = time.time()
        sec = curTime - prevTime
        prevTime = curTime
        fps = 1 / (sec)
        string = "FPS : {0:.1f}".format(fps)
        cv.putText(img, string, (20, 90), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)
    
        if ret == False :
            break
    
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(gray,125,255,cv.THRESH_BINARY)
    
        height = img.shape[0]
        width = img.shape[1]
        rect = np.array([[(250,height-100),(1050,height-100),(700,440),(580,440)]])
        mask = np.zeros_like(thresh)
        cv.fillPoly(mask,rect,255)
        roi = cv.bitwise_and(thresh,mask)
        canny = cv.Canny(roi,50,200,3)
        lines= cv.HoughLinesP(canny, 1, np.pi / 180, 50, minLineLength = 70, maxLineGap = 100)
    
        slopes = []
        new_lines = []
        try :
            for line in lines:
                x1, y1, x2, y2 = line[0]  
    
                if x2 - x1 == 0.: 
                    slope = 999. 
                else:
                    slope = (y2 - y1) / (x2 - x1)
    
                if abs(slope) > 0.6:
                    slopes.append(slope)
                    new_lines.append(line)
        except :
            continue
    
        lines = new_lines
    
        right_lines = []
        left_lines = []
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            img_x_center = width / 2  
            if slopes[i] > 0 and x1 > img_x_center and x2 > img_x_center:
                right_lines.append(line)
            elif slopes[i] < 0 and x1 < img_x_center and x2 < img_x_center:
                left_lines.append(line)
    
        right_lines_x = []
        right_lines_y = []
    
        for line in right_lines:
            x1, y1, x2, y2 = line[0]
    
            right_lines_x.append(x1)
            right_lines_x.append(x2)
    
            right_lines_y.append(y1)
            right_lines_y.append(y2)
    
        if len(right_lines_x) > 0:
            right_m, right_b = np.polyfit(right_lines_x, right_lines_y, 1) 
            draw_right = True
        else:
            right_m, right_b = 1, 1
            draw_right = False
    
        left_lines_x = []
        left_lines_y = []
    
        for line in left_lines:
            x1, y1, x2, y2 = line[0]
    
            left_lines_x.append(x1)
            left_lines_x.append(x2)
    
            left_lines_y.append(y1)
            left_lines_y.append(y2)
    
        if len(left_lines_x) > 0:
            left_m, left_b = np.polyfit(left_lines_x, left_lines_y, 1)
            draw_left = True
        else:
            left_m, left_b = 1, 1
            draw_left = False
    
        y1 = img.shape[0]
        y2 = img.shape[0] * 0.57
    
        left_x1 = (y1 - left_b) / left_m
        left_x2 = (y2 - left_b) / left_m
    
        right_x1 = (y1 - right_b) / right_m
        right_x2 = (y2 - right_b) / right_m
    
        y1 = int(y1)
        y2 = int(y2)
        right_x1 = int(right_x1)
        right_x2 = int(right_x2)
        left_x1 = int(left_x1)
        left_x2 = int(left_x2)
    
        if line[0][0]-line[0][2] == 0 :
            x1 = int(line[0][0])
            x2 = int(line[0][2])
        else :
            m = (line[0][1]-line[0][3])/(line[0][0]-line[0][2])
            b = line[0][1]-m*line[0][0]
            x1 = (y1-b)/m
            x2 = (y2-b)/m
            x1 = int(x1)
            x2 = int(x2)     
    
        if draw_right == False :
            No_right += 1
            right_x1 = idx_rx1
            right_x2 = idx_rx2
    
        if draw_left == False :
            No_left += 1
            left_x1 = idx_lx1
            left_x2 = idx_lx2
    
        if No_left > 10 and Left == False :
            Right = True      
        if No_right > 10 and Right == False :
            Left = True
    
        if draw_left == True and draw_right == True :
        	Right = False
            Left = False
            No_left = 0
            No_right = 0
    
        if Left == False and Right == False :
            idx_rx1 = copy.deepcopy(right_x1)
            idx_rx2 = copy.deepcopy(right_x2)
            idx_lx1 = copy.deepcopy(left_x1)
            idx_lx2 = copy.deepcopy(left_x2)
    
        x = (idx_rx2+idx_lx2)/2
        mp = (idx_rx1+idx_lx1)/2
        bias = (640-(idx_rx1+idx_lx1)/2)/640*100
    
        if abs(bias) < 17 and draw_right == True :
            cv.line(img, (right_x1, y1), (right_x2, y2), (255,0,0), 5)
        if abs(bias) < 17 and draw_left == True :
            cv.line(img, (left_x1, y1), (left_x2, y2), (255,0,0), 5)
    
        if Right == False and draw_right == False and abs(bias) < 17 :
            cv.line(img, (right_x1, y1), (right_x2, y2), (0,255,255), 5)
        if Left == False and draw_left == False and abs(bias) < 17 :
            cv.line(img, (left_x1, y1), (left_x2, y2), (0,255,255), 5)
    
        tri=np.array([[left_x1, y1],[right_x1, y1],[x,y2]],np.int32)
        tri_idx=np.array([[idx_rx1, y1],[idx_lx1, y1],[x,y2]],np.int32)
        img_copy = img.copy()
        if 17 >= bias >= 0 :
            text = str('Bias : Left {0:.2f}'.format(abs(bias))+'%')
            cv.fillConvexPoly(img_copy,tri,(0,255,0))
            img = cv.addWeighted(img, 0.7, img_copy, 0.3, gamma = 0)
            cv.putText(img, text, (20, 30), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)
            cv.putText(img, 'In line? : Safe', (20, 60), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv.circle(img, (int(x), y2), 5, (0,255,0), 10)
            cv.line(img, (int(mp), height), (int(mp), height-100), (255, 0, 255), 5)
        elif -17 <= bias < 0 :
            text = str('Bias : Right {0:.2f}'.format(abs(bias))+'%')
            cv.fillConvexPoly(img_copy,tri,(0,255,0))
            img = cv.addWeighted(img, 0.7, img_copy, 0.3, gamma = 0)
            cv.putText(img, text, (20, 30), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)
            cv.putText(img, 'In line? : Safe', (20, 60), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv.circle(img, (int(x), y2), 5, (0,255,0), 10)
            cv.line(img, (int(mp), height), (int(mp), height-100), (255, 0, 255), 5)
        elif Right == True :
            text = str('Bias : Right {0:.2f}'.format(abs(bias))+'%')
            cv.fillConvexPoly(img_copy,tri_idx,(0,0,255))
            img = cv.addWeighted(img, 0.7, img_copy, 0.3, gamma = 0)
            cv.putText(img, text, (20, 30), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)
            cv.putText(img, 'In line? : Changing Line', (20, 60), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv.circle(img, (x2, y2), 5, (0,0,255), 10)
            cv.line(img, (x1, y1), (x2, y2), (0,0,255), 5)
            cv.line(img, (580, 300), (780, 300), (0,0,255), 5)
            cv.line(img, (700, 260), (780, 300), (0,0,255), 5)
            cv.line(img, (700, 340), (780, 300), (0,0,255), 5)   
        elif Left == True :
            text = str('Bias : Left {0:.2f}'.format(abs(bias))+'%')
            cv.fillConvexPoly(img_copy,tri_idx,(0,0,255))
            img = cv.addWeighted(img, 0.7, img_copy, 0.3, gamma = 0)
            cv.putText(img, text, (20, 30), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)
            cv.putText(img, 'In line? : Changing Line', (20, 60), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv.circle(img, (x2, y2), 5, (0,0,255), 10)
            cv.line(img, (x1, y1), (x2, y2), (0,0,255), 5)
            cv.line(img, (580, 300), (780, 300), (0,0,255), 5)
            cv.line(img, (660, 260), (580, 300), (0,0,255), 5)
            cv.line(img, (660, 340), (580, 300), (0,0,255), 5) 
        elif 17 >= bias >= 0 and draw_left == True or draw_right == True :
            text = str('Bias : Left {0:.2f}'.format(abs(bias))+'%')
            cv.fillConvexPoly(img_copy,tri,(0,255,0))
            img = cv.addWeighted(img, 0.7, img_copy, 0.3, gamma = 0)
            cv.putText(img, text, (20, 30), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)
            cv.putText(img, 'In line? : Safe', (20, 60), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv.circle(img, (int(x), y2), 5, (0,255,0), 10)
            cv.line(img, (int(mp), height), (int(mp), height-100), (255, 0, 255), 5)
        elif 0 >= bias >= -17 and draw_right == True or draw_left == True :
            text = str('Bias : Right {0:.2f}'.format(abs(bias))+'%')
            cv.fillConvexPoly(img_copy,tri,(0,255,0))
            img = cv.addWeighted(img, 0.7, img_copy, 0.3, gamma = 0)
            cv.putText(img, text, (20, 30), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)
            cv.putText(img, 'In line? : Safe', (20, 60), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv.circle(img, (int(x), y2), 5, (0,255,0), 10)
            cv.line(img, (int(mp), height), (int(mp), height-100), (255, 0, 255), 5)
        else :
            text = str('Bias : {0:.2f}'.format(abs(bias))+'%')
            cv.fillConvexPoly(img_copy,tri,(0,255,0))
            img = cv.addWeighted(img, 0.7, img_copy, 0.3, gamma = 0)
            cv.putText(img, text, (20, 30), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)
            cv.putText(img, 'In line? : Safe', (20, 60), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv.circle(img, (int(x), y2), 5, (0,255,0), 10)
            cv.line(img, (int(x+bias/100*640), height), (int(x+bias/100*640), height-100), (255, 0, 255), 5)
    
        cv.line(img, (640, height), (640, height-50), (255, 255, 0), 5)
    
        cv.imshow('result', img)
        if cv.waitKey(1) == ord('q'):
            break
    
        timeLoopEnd = cv.getTickCount()
    
    cv.destroyAllWindows()
    cap.release()


##### Flow Chart

![diagram](https://user-images.githubusercontent.com/80805040/166881966-b7f7aae2-ff92-4a54-a383-1672b86a0f45.png)
