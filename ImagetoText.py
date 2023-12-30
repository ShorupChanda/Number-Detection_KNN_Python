import cv2
print('Package Imported')

path1 = r"C:\Users\User\PycharmProjects\DroneProject\ImagesQuery\419.PNG"
path2 = r"C:\Users\User\PycharmProjects\DroneProject\ImagesTrain\419T.jpg"

img1 = cv2.imread(path1, 0)
img2 = cv2.imread(path2, 0)

orb = cv2.ORB_create(nfeatures=1000)

kp1 , des1 = orb.detectAndCompute(img1, None)
kp2 , des2 = orb.detectAndCompute(img2, None)

#bin of numbers (Array for our operation)
#imgKp1 = cv2.drawKeypoints(img1,kp1,None)
#imgKp2 = cv2.drawKeypoints(img2,kp2,None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

print(len(good))
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)


#cv2.imshow('kp1',imgKp1)
#cv2.imshow('kp2',imgKp2)
cv2.imshow('Output1',img1)
cv2.imshow('Output2',img2)
cv2.imshow('Output3',img3)
cv2.waitKey(0)
