import cv2
import numpy as np
import imutils
from imutils import contours

def percentage(x,w,y,h,img):
    count=0
    for i in range(x,x+w):
        for j in range(y,y+h):
            #print(img[i,j])
            if(img[j,i]==255 ):
                count+=1
    #print(count)
    return (count*100)/(w*h)

def is_contour_bad(c):
    # approximate the contour
    peri = cv2.arcLength(c,False)
    # the contour is 'bad' if it is not a rectangle
    return peri>20

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)
def cavity(cap):
  cap = ResizeWithAspectRatio(cap, width=600)
  frame=cap
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  mask1 = cv2.inRange(hsv,(0, 20, 20), (15, 255, 255) )


  img1=cv2.bitwise_and(frame,frame,mask=mask1)
  hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
  mask2=cv2.inRange(hsv,(9, 20, 20), (35, 255, 255) )
  des = mask2.copy()
  #des= cv2.bitwise_not(des)
  contour,hier = cv2.findContours(des,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
  for cnt in contour:
      cv2.drawContours(des,[cnt],0,255,-1)

  mask2=des

  pixels = cv2.countNonZero(mask1) # OR
  image_area = frame.shape[0] * frame.shape[1]
  area_ratio = (pixels / image_area) * 100
  print(area_ratio)
  if(area_ratio<80):
      des = mask2.copy()
      #des= cv2.bitwise_not(des)
      contour,hier = cv2.findContours(des,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
      for cnt in contour:
          cv2.drawContours(des,[cnt],0,255,-1)
      
      mask1=des
  teeth = cv2.bitwise_and(frame,frame, mask=mask2)
  image=teeth.copy()

  edges = cv2.Canny(image,100,150)

  contours, hierarchy= cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cv2.drawContours(image, contours, -1, (0,255,0),1)
  img1=cap.copy()
  for i in range(0,img1.shape[0]):
    for j in range(0,img1.shape[1]):
        if(image[i,j,0]==0 and image[i,j,1]==0 and image[i,j,2]==0 ):
            img1[i,j]=(0,255,0)
        else:
            break
    for j in range(img1.shape[1]-1,0,-1):
        if(image[i,j,0]==0 and image[i,j,1]==0 and image[i,j,2]==0 ):
            img1[i,j]=(0,255,0)
        else:
            break
  hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
  mask1 = cv2.inRange(img1,(0, 255, 0), (0, 255, 0) )
  mask1=cv2.bitwise_not(mask1)
  mask2=cv2.inRange(hsv,(9, 20, 20), (35, 255, 255) )


  des = mask2.copy()
  #des= cv2.bitwise_not(des)
  contour,hier = cv2.findContours(des,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

  for cnt in contour:
      cv2.drawContours(des,[cnt],0,255,-1)

  mask2=des
  des = mask2.copy()
#des= cv2.bitwise_not(des)
  contour,hier = cv2.findContours(des,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

  for cnt in contour:
      cv2.drawContours(des,[cnt],0,255,-1)
  #cv2.imshow("m3", des);
  mask2=des
  #
  #

  mask= cv2.bitwise_and(mask1,mask2)
  #mask=cv2.bitwise_not(mask)
  res=cv2.bitwise_and(img1,img1,mask=mask)
  res[np.where((res==[0,0,0]).all(axis=2))] = [0,255,0]
  rgb = cv2.cvtColor(res.copy(), cv2.COLOR_BGR2HSV)
  mask = cv2.inRange(rgb, (9, 117, 50),(18,255,114) )
  cavity = cv2.bitwise_and(frame,frame, mask=mask)
  #cv2.imshow('Green',res)
  kernel = np.ones((20,20), np.uint8)
  img = cv2.dilate(mask, kernel, iterations=1)
  output_img=frame.copy()
  avg=0
  total=0
  contours,x= cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  for c in contours:
      # get the bounding rect
      x, y, w, h = cv2.boundingRect(c)
      # draw a white rectangle to visualize the bounding rect
      cv2.rectangle(output_img, (x, y), (x + w, y + h), 255, 1)
      avg=avg+percentage(x,w,y,h,img)
      total=total+1
      
  print("Percentage: ", avg/total)
  output_img=cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
  return output_img


def plaque(cap):
  frame=cap
  image=frame.copy()
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

  lower_red = np.array([30,150,50])
  upper_red = np.array([255,255,180])

  mask = cv2.inRange(hsv, lower_red, upper_red)
  res = cv2.bitwise_and(frame,frame, mask= mask)


  edges = cv2.Canny(frame,100,150)
  edged=edges


  # Exit Condition wait 5 msec and Esc is pressed
  
  
# load the shapes image clone it, convert it to grayscale, and
# detect edges in the image

# find contours in the edge map
  cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
      cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)
  mask = np.ones(image.shape[:2], dtype="uint8") * 255
  # loop over the contours
  for c in cnts:
    # if the contour is bad, draw it on the mask
    if is_contour_bad(c):
      cv2.drawContours(image, [c], -1, (0, 255, 0), -1)
  # remove the contours from the image and show the resulting images


  
  
  image2=image.copy()
  himg = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)

  mask1 = cv2.inRange(hsv,(12, 100, 100), (30, 255, 255) )

  
  mask = cv2.bitwise_or(mask1, mask1)
  kernel = np.ones((3,3),np.uint8)
  opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
  no_contour = cv2.bitwise_and(frame,frame, mask=opening)

  
  def find_if_close(cnt1,cnt2):
      row1,row2 = cnt1.shape[0],cnt2.shape[0]
      for i in range(row1):
          for j in range(row2):
              dist = np.linalg.norm(cnt1[i]-cnt2[j])
              if abs(dist) < 10 :
                  return True
              elif i==row1-1 and j==row2-1:
                  return False
              
  img=frame.copy()
  thresh=opening.copy()
  contours,hier = cv2.findContours(thresh,cv2.RETR_EXTERNAL,2)
  LENGTH = len(contours)
  status = np.zeros((LENGTH,1))

  for i,cnt1 in enumerate(contours):
      x = i    
      if i != LENGTH-1:
          for j,cnt2 in enumerate(contours[i+1:]):
              x = x+1
              dist = find_if_close(cnt1,cnt2)
              if dist == True:
                  val = min(status[i],status[x])
                  status[x] = status[i] = val
              else:
                  if status[x]==status[i]:
                      status[x] = i+1

  unified = []
  maximum = int(status.max())+1
  for i in range(maximum):
      pos = np.where(status==i)[0]
      if pos.size != 0:
          cont = np.vstack(contours[i] for i in pos)
          hull = cv2.convexHull(cont)
          unified.append(hull)

  cv2.drawContours(img,unified,-1,(0,255,0),2)
  cv2.drawContours(thresh,unified,-1,255,-1)

  
  
  #TODO for identifying tooth instead of plaque draw rectangles via initial edges that contain plaque rectangles
  cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
  cnts = imutils.grab_contours(cnts)
  uy=frame.shape[0]
  ux=frame.shape[1]
  avg=0;
  total=0;
  print(ux)
  print(uy)
  img=frame.copy()
  # loop over the contours
  for c in cnts:
      x,y,w,h = cv2.boundingRect(c)
      # The size limitation of the boxes over the plaque
      if(x>ux/10 and x<ux/1.1 and y>uy/20 and x<ux/1.05):
      #if(w>ux/60 and h>uy/25 and x>ux/10 and x<ux/1.1 and y>uy/20 and x<ux/1.05):
          avg=avg+percentage(x,w,y,h,thresh)
          total=total+1
          img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
          
  print("Percentage: ", avg/total)
  output_img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return output_img
  #cv2.imshow('op', opening)

def staining(cap):

  cap =  ResizeWithAspectRatio(cap, width=600)
  frame=cap
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  mask1 = cv2.inRange(hsv,(7, 142, 67), (15, 255, 212))
  mask = cv2.bitwise_and(frame,frame,mask=mask1)
  kernel = np.ones((5,5),np.uint8)
  opening = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)
  no_contour = cv2.bitwise_and(frame,frame, mask=opening)

  contour,hier = cv2.findContours(opening,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

  for cnt in contour:
      cv2.drawContours(opening,[cnt],0,255,-1)
      

  thresh=opening
  kernel = np.ones((20,20), np.uint8)
  img = cv2.dilate(thresh, kernel, iterations=1)
  #print(img.shape)
  avg=0
  total=0
  output_img=frame.copy()
  contours,x= cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  for c in contours:
      # get the bounding rect
      x, y, w, h = cv2.boundingRect(c)
      #Area of cavity
      #print( 100* count_pixel(x,w,y,h,img)/(w*h))
      # draw a white rectangle to visualize the bounding rect
      cv2.rectangle(output_img, (x, y), (x + w, y + h), 255, 1)
      avg=avg+percentage(x,w,y,h,img)
      total=total+1
      
  print("Percentage: ", avg/total)
  output_img=cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
  return output_img

    