import csv #provides functionality for reading and writing Comma-Separated Values
import copy # provides functions for creating copies of objects in various ways
import argparse #define command-line interfaces for Python scripts; easier for users to interact with program through the terminal
import itertools #allow to create, manipulate, and combine iterators 
from collections import Counter #creates a dictionary-like object where the elements of the iterable are keys, and their counts are the corresponding values.
from collections import deque #versatile data structure that provides a list-like interface with efficient and fast operations for adding and removing elements from both ends of the sequence(append and pop elements)
 
import cv2 as cv #provides various functions and tools for working with images, videos, and computer vision algorithms. cv2: perform a wide range of tasks related to image processing, computer vision, and machine learning
import numpy as np #mathematical and numerical operations on arrays, manipulate data, and perform tasks like linear algebra, statistical analysis
import mediapipe as mp #rocess of building applications that involve tasks such as hand tracking, pose estimation, facial recognition

from utils import CvFpsCalc # used when you want to use a specific class, function, or variable defined in a separate module within your own code.
from model import KeyPointClassifier # bring in specific classes, functions, or variables defined in separate Python modules 
from model import PointHistoryClassifier #used to import specific classes, functions, or variables defined in separate Python modules


def get_args(): #start of a function named get_args()
    parser = argparse.ArgumentParser() #define and manage command-line arguments and options for Python script.

    parser.add_argument("--device", type=int, default=0) # specify an integer value on the command line when running the script.
    parser.add_argument("--width", help='cap width', type=int, default=960) #specify an integer value on the command line when running the script, controlling the width of some aspect
    parser.add_argument("--height", help='cap height', type=int, default=540) # specify an integer value on the command line when running the script, controlling the height of some aspect

    parser.add_argument('--use_static_image_mode', action='store_true') # is a boolean flag that doesn't require a value; its presence on the command line indicates that a specific behavior should be enabled.
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7) #allows the user to specify a floating-point value on the command line when running the script. This value is likely used to set a threshold confidence level for some kind of detection operation.
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5) #allows the user to specify an integer value on the command line when running the script. This value is likely used to set a threshold confidence level for some kind of tracking operation.

    args = parser.parse_args() #collects the arguments and options specified on the command line and stores them in an object called args.

    return args #signifies the end of the function's execution and provides the value of args to the caller of the function.


def main():  #start of a function named main()
    args = get_args() # calls a function named get_args()

    cap_device = args.device #takes the value of the device attribute from the args object and assigns it to the cap_device variable;use the cap_device variable in your code instead of repeatedly accessing the args.device attribute
    cap_width = args.width #takes the value of the width attribute from the args object and assigns it to the cap_width variable.
    cap_height = args.height #takes the value of the height attribute from the args object and assigns it to the cap_width variable.

    use_static_image_mode = args.use_static_image_mode # takes the value of the use_static_image_mode attribute from the args object and assigns it to the use_static_image_mode variable.
    min_detection_confidence = args.min_detection_confidence 
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True #creates a variable named use_brect and assigning the value True to it.

    cap = cv.VideoCapture(cap_device) #creates an instance of the cv.VideoCapture class and opens the video capture device specified by the cap_device variable. 
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width) #sets the frame width of the video capture instance (cap) to the value specified by the cap_width variable. 
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height) #sets the frame height of the video capture instance (cap) to the value specified by the cap_height variable.

    mp_hands = mp.solutions.hands # creates an instance of the Hands solution provided by mediapipe; stored in the mp_hands variable, allows to use the capabilities of the Hands model to perform hand tracking and landmark detection on images or frames.
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) # creates an instance of the Hands class with the specified configuration parameters; stored in the hands variable; used to process images or frames and detect hand landmarks according to the specified parameters.

    keypoint_classifier = KeyPointClassifier() #creates an instance of the KeyPointClassifier class; stores it in the keypoint_classifier variable. Used to access the methods and attributes of the KeyPointClassifier class.

    point_history_classifier = PointHistoryClassifier() #access the methods and attributes of the PointHistoryClassifier class.

    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f: # nsure proper handling of file resources, including automatic closing of the file when done.
        keypoint_classifier_labels = csv.reader(f) #creates a CSV reader object named keypoint_classifier_labels that's associated with the opened file f; use this object to read and process the CSV data line by line.
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ] # creates a new list that contains the first elements from each row of the CSV data; often used to extract labels or identifiers from a CSV file.

    with open( 'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    cvFpsCalc = CvFpsCalc(buffer_len=10) #creates an instance of the CvFpsCalc class with buffer length of 10; stored in the cvFpsCalc variable; calculates the frames per second of a video stream or sequence of images.

    history_length = 16 #sets the value of the history_length variable to 16; this variable can be used throughout the code to refer to this value.
    point_history = deque(maxlen=history_length) #allows to efficiently maintain a history of elements while automatically discarding older elements when the maximum length is reached.

    finger_gesture_history = deque(maxlen=history_length) #creates instance of the deque class named finger_gesture_history with a maximum length as specified by the value of the history_length variable

    mode = 0 #sets the value of the mode variable to 0

    while True:
        fps = cvFpsCalc.get() # calculates the current FPS using the CvFpsCalc instance and stores the calculated value in the fps variable. 

        key = cv.waitKey(10) # captures user input by waiting for a key press event for up to 10 milliseconds. If a key is pressed within that time, the ASCII value of the key is stored in the key variable. If no key is pressed, the key variable will hold the value -1.
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode) #calling the select_mode() function with the current values of key and mode as arguments; returns 2 values, which are then assigned to the variables number and mode

        ret, image = cap.read() #captures a frame from the video capture source, stores the success status in the ret variable; stores the captured frame in the image variable.
        if not ret:
            break
        image = cv.flip(image, 1) # performs a horizontal flip on the captured image, creating a mirror image of the original frame.
        debug_image = copy.deepcopy(image) # create a new instance of the image array that is entirely independent of the original array; useful for preventing unintended side effects when modifying one copy of the image, while keeping the other copy unchanged.

        
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB) # converts the color space of the captured BGR image to RGB.

        image.flags.writeable = False # sets the writeable flag of the image array to False, making it so that you can't modify the data of the array.
        results = hands.process(image) #  using the Hands class instance and stores the results of the hand tracking process in the results variable
        image.flags.writeable = True # sets writeable flag of the image array to True, allowing to modify the data of the array if you need to.

        
        if results.multi_hand_landmarks is not None: # checks if the multi_hand_landmarks attribute of the results object contains information about detected hand landmarks. If there are detected landmarks, the code block inside the if statement will be executed.
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness): #sets up a loop to iterate over pairs of detected hand landmarks and their corresponding handedness information; allows to process and analyze each detected hand and its characteristics separately.
                
                brect = calc_bounding_rect(debug_image, hand_landmarks) # invokes the calc_bounding_rect function with the debug_image and hand_landmarks as arguments; stores the calculated bounding rectangle in the brect variable.
                
                landmark_list = calc_landmark_list(debug_image, hand_landmarks) # invokes the calc_landmark_list function with the debug_image and hand_landmarks as arguments; stores the calculated list of landmark coordinates in the landmark_list variable.

                
                pre_processed_landmark_list = pre_process_landmark(landmark_list) # invokes the pre_process_landmark function with the landmark_list as an argument and storing the result of the preprocessing in the pre_processed_landmark_list variable.

                pre_processed_point_history_list = pre_process_point_history( debug_image, point_history) # invokes  the pre_process_point_history function with the debug_image and point_history as arguments and storing the result of the preprocessing in the pre_processed_point_history_list variable.
                
                logging_csv(number, mode, pre_processed_landmark_list, pre_processed_point_history_list) # invokes the logging_csv function and passing these four pieces of information as arguments; then performs some logging operation based on this data.

                
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list) #invokes the keypoint_classifier function, passes the pre-processed landmark data as an argument. The function or model processes the data and returns a classification result (stored in the hand_sign_id variable).
                if hand_sign_id == 2:  
                    point_history.append(landmark_list[8]) # adds the value at index 8 of landmark_list to the end of the point_history collection, updates the history of points. 
                else:
                    point_history.append([0, 0]) # adds the point [0, 0] to the end of the point_history collection.

                 
                finger_gesture_id = 0 # sets the value of the variable finger_gesture_id to 0. This initial value can be later updated or used in code logic.
                point_history_len = len(pre_processed_point_history_list) # calculates the length of the pre_processed_point_history_list and stores the result (the number of elements in the list) in the point_history_len variable.
                if point_history_len == (history_length * 2): # checks  if the length of the point history (point_history_len) is equal to two times the history_length. If this condition is true, the code block following the if statement will be executed.
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list) # invokes the point_history_classifier function; passes the pre-processed point history data as an argument. The function or model processes the data and returns a classification result (stored in the finger_gesture_id variable).

                
                finger_gesture_history.append(finger_gesture_id) # adds the current value of finger_gesture_id to the end of the finger_gesture_history collection, effectively updating the history of finger gesture IDs.
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common() # creates a Counter object to count the occurrences of different finger gesture IDs in the finger_gesture_history list. The .most_common() method is then called on the Counter object to retrieve a list of tuples where each tuple contains a finger gesture ID and its count, sorted in descending order of frequency.

                
                debug_image = draw_bounding_rect(use_brect, debug_image, brect) # calls he draw_bounding_rect function with the provided arguments and assigning the modified image (with the bounding rectangle drawn) back to the debug_image variable.
                debug_image = draw_landmarks(debug_image, landmark_list) # calls draw_landmarks function with the provided arguments and assigning the modified image (with the landmarks drawn) back to the debug_image variable.
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                ) # calls he draw_info_text function with the provided arguments and assigning the modified image (with the information text drawn) back to the debug_image variable.

        else:
            point_history.append([0, 0]) #  adds the list [0, 0] to the end of the point_history collection. This could represent a default or placeholder data point added to the history for some specific purpose in the code.

        debug_image = draw_point_history(debug_image, point_history) # calls the draw_point_history function with the provided arguments and assigning the modified image (with the point history representation drawn) back to the debug_image variable.
        debug_image = draw_info(debug_image, fps, mode, number) # calls the draw_info function with the provided arguments and assigning the modified image (with the drawn information) back to the debug_image variable.
        
        cv.imshow('Hand Gesture Recognition', debug_image) # opens a window with the title "Hand Gesture Recognition" and displays the content of the debug_image in that window.

    cap.release() # to ensure that any camera or file resources that were being used for capturing frames are properly closed and released; done when you're done with the video processing task or when you're exiting your application.
    cv.destroyAllWindows() # to ensure that all windows created by OpenCV are properly closed, releasing the associated resources; done when you're done with the image or video processing tasks and want to close all windows before exiting your application.


def select_mode(key, mode): #calls a function names select_mode with 2 arguments
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks): # calls a function named calc_bounding_rect that has 2 arguments
    image_width, image_height = image.shape[1], image.shape[0] # extracts the width and height dimensions of the image array and assigning them to the variables image_width and image_height. 

    landmark_array = np.empty((0, 2), int) # creates an empty NumPy array that's intended to store pairs of x and y coordinates (landmarks). As you process data and gather landmark coordinates, you can append them to this array.

    for _, landmark in enumerate(landmarks.landmark): # a loop that iterates through each landmark in the landmarks.landmark collection, ignoring the index of the iteration by using _. The landmark variable is assigned the value of each individual landmark during each iteration of the loop.
        landmark_x = min(int(landmark.x * image_width), image_width - 1) # calculates the x-coordinate of the landmark in pixels while making sure the calculated value is within the valid range of pixel coordinates for the image.
        landmark_y = min(int(landmark.y * image_height), image_height - 1) # calculates the y-coordinate of the landmark in pixels while making sure the calculated value is within the valid range of pixel coordinates for the image.

        landmark_point = [np.array((landmark_x, landmark_y))] #  creates a NumPy array that represents a single point (landmark) with the x and y pixel coordinates.

        landmark_array = np.append(landmark_array, landmark_point, axis=0) # appends a new landmark point to the existing landmark_array along the vertical axis, effectively adding a new row to the array.

    x, y, w, h = cv.boundingRect(landmark_array) # calculates the bounding rectangle around the landmark points stored in landmark_array and assigns the position, width, and height of the rectangle to the variables x, y, w, and h.

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks): # calls a function named calc_kandmark_list that has 2 arguments
    image_width, image_height = image.shape[1], image.shape[0] # extracts the width and height dimensions of the image array and assigning them to the variables image_width and image_height

    landmark_point = [] #  sets up the landmark_point variable to hold a list that will be used to store landmark points.

    
    for _, landmark in enumerate(landmarks.landmark): #  a loop that iterates through each landmark in the landmarks.landmark collection, ignoring the index of the iteration by using _, and assigning the value of each landmark to the landmark variable during each iteration.
        landmark_x = min(int(landmark.x * image_width), image_width - 1) # calculates the x-coordinate of the landmark point in pixels while ensuring it doesn't exceed the valid range of pixel coordinates for the image width.
        landmark_y = min(int(landmark.y * image_height), image_height - 1) # calculates the y-coordinate of the landmark point in pixels while ensuring it doesn't exceed the valid range of pixel coordinates for the image height.
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y]) # adds a new landmark point (in the form of a list) to the landmark_point list.

    return landmark_point


def pre_process_landmark(landmark_list): # calls a function named pre_process_landmark that takes 1 parameter
    temp_landmark_list = copy.deepcopy(landmark_list) # creates a deep copy of landmark_list and assigns it to the temp_landmark_list variable.

    
    base_x, base_y = 0, 0 # intializes base_x and base_y with the values 0 for both variables; common technique to set initial values for variables at the beginning of a code section.
    for index, landmark_point in enumerate(temp_landmark_list): #  sets up a loop that iterates through each element (landmark point) in temp_landmark_list, assigning both the index and the value of the element to the variables index and landmark_point, respectively.
        if index == 0: 
            base_x, base_y = landmark_point[0], landmark_point[1] # extracts the x and y coordinates from a landmark point and assigning them to the variables base_x and base_y.

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x # calculates the new x-coordinate for the specified landmark point by subtracting the base_x value from the current x-coordinate and updates the list with the new x-coordinate.
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y # calculates the new y-coordinate for the specified landmark point by subtracting the base_y value from the current y-coordinate and updates the list with the new y-coordinate.

    
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list)) # flattens the nested list structure, converting it into a single list of all the individual elements.

    
    max_value = max(list(map(abs, temp_landmark_list))) # calculates the maximum absolute value among the numbers in temp_landmark_list.

    def normalize_(n): # function named normalize_ that takes a single parameter n.
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list)) # applies the normalize_ function to each element in temp_landmark_list, creating a new list with the normalized values.

    return temp_landmark_list 


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0] # extracts the width and height of the image and assigning them to the respective variables image_width and image_height.



    temp_point_history = copy.deepcopy(point_history) #  creates a new deque (temp_point_history) that is a completely independent copy of the original point_history, including its contents.

    
    base_x, base_y = 0, 0 # initializes the base_x and base_y variables with the initial values of 0.
    for index, point in enumerate(temp_point_history): # creates a loop that iterates through each element in the temp_point_history deque, and for each iteration, it assigns the current index to index and the current element to point.
        if index == 0:
            base_x, base_y = point[0], point[1] # extracts the x and y coordinates from the point list and assigns them to the respective variables base_x and base_y.

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width # calculates the normalized x-coordinate of the specified point by subtracting the base_x value from the current x-coordinate and then dividing the result by the image width.


        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height #  calculates the normalized y-coordinate of the specified point by subtracting the base_y value from the current y-coordinate and then dividing the result by the image height.

    
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history)) # takes the nested list temp_point_history, flattens it, and creates a new list containing all the individual elements.

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv' # sets the variable csv_path to the file path 'model/keypoint_classifier/keypoint.csv'.
        with open(csv_path, 'a', newline="") as f: #establishes a context in which the CSV file specified by csv_path is opened in append mode. The file is assigned to the variable f, which you can use to interact with the file within the context.
            writer = csv.writer(f) #creates a CSV writer object that is associated with the file f, allowing you to write data to the CSV file using this writer object.
            writer.writerow([number, *landmark_list]) # writes a single row to the CSV file, where the row contains the number followed by all the elements in the landmark_list.
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv' # sets the variable csv_path to the file path 'model/point_history_classifier/point_history.csv'.
        with open(csv_path, 'a', newline="") as f: # establishes a context in which the CSV file specified by csv_path is opened in append mode. The file is assigned to the variable f, which you can use to interact with the file within the context.
            writer = csv.writer(f) # creates a CSV writer object that is associated with the file f, allowing you to write data to the CSV file using this writer object.
            writer.writerow([number, *point_history_list]) # writes a single row to the CSV file, where the row contains the number followed by all the elements in the point_history_list.
    return


def draw_landmarks(image, landmark_point):

    if len(landmark_point) > 0:
   
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6) # draws a black line on the image connecting the third and fourth points specified in the landmark_point list.
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2) # draws a white line on the image connecting the fourth and fifth points specified in the landmark_point list.

        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)


        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

     
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

      
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

       
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)
 

    for index, landmark in enumerate(landmark_point):
        if index == 0:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1) #draws a filled white circle on the image at the coordinates (landmark[0], landmark[1]) with a radius of 5 pixels.
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1) #draws a black circle on the image at the coordinates (landmark[0], landmark[1]) with a radius of 5 pixels and an outline thickness of 1 pixel.
       
        if index == 1:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        
        if index == 2:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)   
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        
        if index == 3:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        
        if index == 4:  
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        
        if index == 5:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
       
        if index == 6:   
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        
        if index == 7:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        
        if index == 8:   
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        
        if index == 9: 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        
        if index == 10:   
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        
        if index == 11:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
       
        if index == 12: 
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),-1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        
        if index == 13:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:   
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:   
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:   
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:   
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:   
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect: # if use_brect is True, the code block following if use_brect: will be executed. If use_brect is False, the code block will be skipped.
   
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1) # draws a black rectangle on the image with the specified corners and outline thickness.

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,finger_gesture_text):
    
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1) #draws a filled black rectangle on the image with the specified corners.

    info_text = handedness.classification[0].label[0:] # assigns the he label (which likely indicates whether the hand is left or right) to the variable info_text.
    if hand_sign_text != "": # If the value of hand_sign_text is not an empty string, the code block indented below this line will be executed. If hand_sign_text is an empty string, the code block will be skipped.
        info_text = info_text + ':' + hand_sign_text # combines he values of info_text, a colon, and hand_sign_text to create a new string, and then storing that new string in the variable info_text.
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA) #adds the specified info_text to the image at the given position with the specified font properties.

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA) #adds the specified text ("Finger Gesture:" followed by the value of finger_gesture_text) to the image at the given position with the specified font properties.
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history): #used to loop through each element in the point_history list. On each iteration, the index variable will contain the index of the current element, and the point variable will contain the value of the current element.
        if point[0] != 0 and point[1] != 0: # checks whether both the x-coordinate (point[0]) and the y-coordinate (point[1]) of the point tuple are not equal to 0. If both conditions are true, the code block indented below this line will be executed.
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2) # draws a circle on the image at the position specified by the point tuple, with a radius that increases with each iteration of the loop (using the index variable), and a green color. The circle's outline is drawn with a thickness of 2 pixels.

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA) #adds the text "FPS:" followed by the value of fps to the image at the given position with the specified font properties.
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA) # adds the text "FPS:" followed by the value of fps to the image at the given position with the specified font properties. The text is white with a thicker stroke compared to the previous example.

    mode_string = ['Logging Key Point', 'Logging Point History'] #creates a list with two string elements, each describing a different mode for an application or program. This list can be used later to display or select the mode description based on the current mode value.
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA) # adds the text "MODE:" followed by the appropriate mode description from the mode_string list to the image at the given position with the specified font properties.
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)  #adds the text "NUM:" followed by the value of the number variable to the image at the given position with the specified font properties.
    return image


if __name__ == '__main__':
    main()
