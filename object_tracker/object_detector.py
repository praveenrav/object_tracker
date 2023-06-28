
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np


class ObjectDetector(Node):

    def __init__(self):
        super().__init__('object_detector')

        self.image_sub = self.create_subscription(Image, "/image_in", self.receive_image_callback, rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value)
        self.image_out_pub = self.create_publisher(Image, "/image_out", 1)
        self.image_tuning_pub = self.create_publisher(Image, "/image_tuning", 1)
        self.ball_pub = self.create_publisher(Point, "/detected_object", 1)

        # Retrieving parameters:
        self.declare_parameter("tuning_mode", False)
        self.declare_parameter("x_min",0)
        self.declare_parameter("x_max",100)
        self.declare_parameter("y_min",0)
        self.declare_parameter("y_max",100)
        self.declare_parameter("h_min",0)
        self.declare_parameter("h_max",180)
        self.declare_parameter("s_min",0)
        self.declare_parameter("s_max",255)
        self.declare_parameter("v_min",0)
        self.declare_parameter("v_max",255)
        self.declare_parameter("sz_min",0)
        self.declare_parameter("sz_max",100)   

        # Retrieving parameters:
        self.tuning_mode = self.get_parameter('tuning_mode').get_parameter_value().bool_value
        self.tuning_params = {
            'x_min': self.get_parameter('x_min').get_parameter_value().integer_value,
            'x_max': self.get_parameter('x_max').get_parameter_value().integer_value,
            'y_min': self.get_parameter('y_min').get_parameter_value().integer_value,
            'y_max': self.get_parameter('y_max').get_parameter_value().integer_value,
            'h_min': self.get_parameter('h_min').get_parameter_value().integer_value,
            'h_max': self.get_parameter('h_max').get_parameter_value().integer_value,
            's_min': self.get_parameter('s_min').get_parameter_value().integer_value,
            's_max': self.get_parameter('s_max').get_parameter_value().integer_value,
            'v_min': self.get_parameter('v_min').get_parameter_value().integer_value,
            'v_max': self.get_parameter('v_max').get_parameter_value().integer_value,
            'sz_min': self.get_parameter('sz_min').get_parameter_value().integer_value,
            'sz_max': self.get_parameter('sz_max').get_parameter_value().integer_value
        }

        self.bridge = CvBridge()

        if(self.tuning_mode):
            self.createTuningWindow(self.tuning_params)



    def receive_image_callback(self, data):
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            self.get_logger().info(e)

        try:
            
            if(self.tuning_mode):
                self.tuning_params = self.getTuningParams()

            keypoints_norm, out_image, tuning_image, equalized_image = self.find_circles(cv_image, self.tuning_params)

            img_to_pub = self.bridge.cv2_to_imgmsg(out_image, "bgr8")
            img_to_pub.header = data.header
            self.image_out_pub.publish(img_to_pub)

            # cv2.imshow('Equalized', equalized_image)

            img_to_pub = self.bridge.cv2_to_imgmsg(tuning_image, "bgr8")
            img_to_pub.header = data.header
            self.image_tuning_pub.publish(img_to_pub)

            point_out = Point()

            # Keep the biggest point
            # They are already converted to normalized coordinates
            for i, kp in enumerate(keypoints_norm):
                x = kp.pt[0]
                y = kp.pt[1]
                s = kp.size

                self.get_logger().info(f"Pt {i}: ({x},{y},{s})")

                if (s > point_out.z):                    
                    point_out.x = x
                    point_out.y = y
                    point_out.z = s

            if (point_out.z > 0):
                self.ball_pub.publish(point_out) 

        except CvBridgeError as e:
            self.get_logger().info(e)



    def find_circles(self, image, tuning_params):

        blur = 5

        working_image = cv2.blur(image, (blur, blur))

        # Add a way to receive search window as input - in this case, the entire picture frame serves as the search window:
        # search_window = [0.0, 0.0, 1.0, 1.0]
        # search_window_px = self.searchWindowConvert(search_window, image)

        # Convert image from BGR to HSV:
        equalized_image = working_image
        b, g, r = cv2.split(equalized_image)
        
        # Apply histogram equalization to each color channel:
        b_eq = cv2.equalizeHist(b)
        g_eq = cv2.equalizeHist(g)
        r_eq = cv2.equalizeHist(r)

        # Merge the equalized color channels back into a color image:
        equalized_image = cv2.merge((b_eq, g_eq, r_eq))


        working_image = cv2.cvtColor(working_image, cv2.COLOR_BGR2HSV)
        # working_image = cv2.cvtColor(equalized_image, cv2.COLOR_BGR2HSV)


        # Apply HSV Threshold:
        thresh_min = (tuning_params['h_min'], tuning_params['s_min'], tuning_params['v_min'])
        thresh_max = (tuning_params['h_max'], tuning_params['s_max'], tuning_params['v_max'])

        # Threshold the image to obtain only the regions satisfying the HSV threshold:
        working_image = cv2.inRange(working_image, thresh_min, thresh_max)

        # Dilate and erode:
        working_image = cv2.dilate(working_image, None, iterations=2)
        working_image = cv2.erode(working_image, None, iterations=2)

        tuning_image = cv2.bitwise_and(image, image, mask = working_image)


        # Invert the image to suit the blob detector:
        working_image = 255-working_image


        # Set up the SimpleBlobdetector with default parameters:
        params = cv2.SimpleBlobDetector_Params()
            
        # Change thresholds:
        params.minThreshold = 0
        params.maxThreshold = 100
            
        # Filter by area:
        params.filterByArea = True
        params.minArea = 700
        params.maxArea = 20000
            
        # Filter by circularity:
        params.filterByCircularity = True
        params.minCircularity = 0.1
            
        # Filter by convexity:
        params.filterByConvexity = True
        params.minConvexity = 0.75
            
        # Filter by inertia:
        params.filterByInertia = True
        params.minInertiaRatio = 0.5

        detector = cv2.SimpleBlobDetector_create(params)

        # Run detection
        keypoints = detector.detect(working_image)

        size_min_px = tuning_params['sz_min']*working_image.shape[1]/100.0
        size_max_px = tuning_params['sz_max']*working_image.shape[1]/100.0

        keypoints = [k for k in keypoints if k.size > size_min_px and k.size < size_max_px]

        
        # Set up main output image
        line_color=(0, 0, 255)

        out_image = cv2.drawKeypoints(image, keypoints, np.array([]), line_color, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Set up tuning output image
        tuning_image = cv2.drawKeypoints(tuning_image, keypoints, np.array([]), line_color, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # tuning_image = draw_window(tuning_image, search_window)
        # cv2.rectangle(image,(x_min_px,y_min_px),(x_max_px,y_max_px),color,line)

        keypoints_normalised = [self.normalise_keypoint(working_image, k) for k in keypoints]

        return keypoints_normalised, out_image, tuning_image, equalized_image


    def normalise_keypoint(self, cv_image, kp):
        rows = float(cv_image.shape[0])
        cols = float(cv_image.shape[1])
        # print(rows, cols)
        center_x    = 0.5*cols
        center_y    = 0.5*rows
        # print(center_x)
        x = (kp.pt[0] - center_x)/(center_x)
        y = (kp.pt[1] - center_y)/(center_y)
        return cv2.KeyPoint(x, y, kp.size/cv_image.shape[1])


    def createTuningWindow(self, tuning_params):
        cv2.namedWindow("Tuning", 0)
        cv2.createTrackbar("x_min","Tuning",tuning_params['x_min'],100,self.no_op)
        cv2.createTrackbar("x_max","Tuning",tuning_params['x_max'],100,self.no_op)
        cv2.createTrackbar("y_min","Tuning",tuning_params['y_min'],100,self.no_op)
        cv2.createTrackbar("y_max","Tuning",tuning_params['y_max'],100,self.no_op)
        cv2.createTrackbar("h_min","Tuning",tuning_params['h_min'],180,self.no_op)
        cv2.createTrackbar("h_max","Tuning",tuning_params['h_max'],180,self.no_op)
        cv2.createTrackbar("s_min","Tuning",tuning_params['s_min'],255,self.no_op)
        cv2.createTrackbar("s_max","Tuning",tuning_params['s_max'],255,self.no_op)
        cv2.createTrackbar("v_min","Tuning",tuning_params['v_min'],255,self.no_op)
        cv2.createTrackbar("v_max","Tuning",tuning_params['v_max'],255,self.no_op)
        cv2.createTrackbar("sz_min","Tuning",tuning_params['sz_min'],100,self.no_op)
        cv2.createTrackbar("sz_max","Tuning",tuning_params['sz_max'],100,self.no_op)               


    def getTuningParams(self):
        # Helper method to retrieve all tuning parameters:
        
        trackbar_names = ["x_min","x_max","y_min","y_max","h_min","h_max","s_min","s_max","v_min","v_max","sz_min","sz_max"]
        return {key:cv2.getTrackbarPos(key, "Tuning") for key in trackbar_names}



    # def searchWindowConvert(self, perc, image):
    #     rows = image.shape[0]
    #     cols = image.shape[1]

    #     scale = [cols rows cols rows]

    #     return [int(a * b) for a, b in zip(perc, scale)]


    def wait_on_gui(self):
        cv2.waitKey(2)

    def no_op(self, x):
        pass


def main(args=None):
    rclpy.init(args=args)

    object_detector = ObjectDetector()

    while rclpy.ok():
        rclpy.spin_once(object_detector)
        object_detector.wait_on_gui()
        
    object_detector.destroy_node()
    rclpy.shutdown()
        