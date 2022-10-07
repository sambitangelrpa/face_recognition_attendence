import encodings
import face_recognition as face_rec
import cv2

#resiging images
def resize(img, size) :
    width = int(img.shape[1]*size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation= cv2.INTER_AREA)

#loading images
sambit=face_rec.load_image_file('sample_images_compare/sachin.jpg')
sambit=cv2.cvtColor(sambit,cv2.COLOR_BGR2RGB)
sambit=resize(sambit,0.5)
sambit_test=face_rec.load_image_file('sample_images_compare/sachin_mask.jpg')
sambit_test=cv2.cvtColor(sambit_test,cv2.COLOR_BGR2RGB)
sambit_test=resize(sambit_test,0.5)

#finding face locations
face_location_sambit=face_rec.face_locations(sambit)[0]
encodings_sambit=face_rec.face_encodings(sambit)[0]
cv2.rectangle(sambit,(face_location_sambit[3],face_location_sambit[0]),(face_location_sambit[1],face_location_sambit[2]),(0,0,255),3)

face_location_sambit_test=face_rec.face_locations(sambit_test)[0]
encodings_sambit_test=face_rec.face_encodings(sambit_test)[0]
cv2.rectangle(sambit_test,(face_location_sambit_test[3],face_location_sambit_test[0]),(face_location_sambit_test[1],face_location_sambit_test[2]),(0,0,255),3)

#comparing images
results=face_rec.compare_faces([encodings_sambit],encodings_sambit_test)
print(results)
cv2.putText(sambit_test,f'{results} ',(90,30),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(1,1,255),2)

#displaying images
cv2.imshow('main_img',sambit)
cv2.imshow('test_img',sambit_test)



cv2.waitKey(0)
cv2.destroyAllWindows()