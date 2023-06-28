
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.conditions import UnlessCondition

import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    params_file = LaunchConfiguration('params_file')
    params_file_dec = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(get_package_share_directory('object_tracker'),'config','object_tracker_params.yaml'),
        description='Full path to params file for all ball_tracker nodes.')
    
    tune_detection = LaunchConfiguration('tune_detection')
    tune_detection_dec = DeclareLaunchArgument(
    'tune_detection',
    default_value='true',
    description='Enables tuning mode for the detection')


    image_topic = LaunchConfiguration('image_topic')
    image_topic_dec = DeclareLaunchArgument(
        'image_topic',
        default_value='/image_raw',
        description='The name of the input image topic.')

    detect_node = Node(
            package='object_tracker',
            executable='object_detector',
            parameters=[params_file, {'tuning_mode': tune_detection}],
            remappings=[('/image_in',image_topic)]
         )
    
    follow_node = Node(
            package='object_tracker',
            executable='object_follower',
            parameters=[params_file]
         )

    return LaunchDescription([
        params_file_dec,
        tune_detection_dec,
        image_topic_dec,
        detect_node,
        follow_node,
    ])