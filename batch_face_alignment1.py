import dlib
import cv2
import os
from skimage import io


def extract_algned_face(face_file_path):
    predictor_path = 'C:\\Dlib\\dlib-19.15.0\\python_examples\\shape_predictor_5_face_landmarks.dat'

    # Load all the models we need: a detector to find the faces, a shape predictor
    # to find face landmarks so we can precisely localize the face
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(predictor_path)

    # Load the image using Dlib
    img = dlib.load_rgb_image(face_file_path)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)

    num_faces = len(dets)
    if num_faces == 0:
        print("Sorry, there were no faces found in '{}'".format(face_file_path))
        exit()

    # Find the 5 face landmarks we need to do the alignment.
    faces = dlib.full_object_detections()
    for detection in dets:
        faces.append(sp(img, detection))

    image = dlib.get_face_chip(img, faces[0], size=320)
    return image


# the folder containing all the input data
input_folder = 'C:\\Users\\Kai\\Desktop\\CBSR_face\\CBSR_database.images'

# the folder to store the output results
output_face_aligned_folder = 'C:\\Users\\Kai\\Desktop\\CBSR_face\\CBSR_database.faces_aligned'

# list all the item of the input folder
items = os.listdir(input_folder)
# make all the items full-path
for i in range(len(items)):
    items[i] = '{0}\\{1}'.format(input_folder, items[i])

while len(items) > 0:
    # pop the first folder from folders
    item = items.pop(0)
    # check the type of the item
    if os.path.isdir(item):
        # the item is type of folder so append it to folders
        # print('folder: {0}\n'.format(item))
        # list all the items in the item
        sub_items = os.listdir(item)
        for i in range(len(sub_items)):
            items.append('{0}\\{1}'.format(item, sub_items[i]))
    elif os.path.isfile(item) :
        # the item is type of file so append it to folders
        print('file: {0}'.format(item))

        new_folder = item[:item.rfind('\\')]
        new_folder = new_folder.replace(input_folder, output_face_aligned_folder)
        filename_face = item[item.rfind('\\')+1:item.rfind('.png')]
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
            print('New folder created: {}'.format(new_folder))
        face = extract_algned_face(item)

        io.imsave('{0}\\{1}.png'.format(new_folder, filename_face), face)
