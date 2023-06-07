# Orin_Depth
 This Repo contains the applications to run YOLO on a ZED 2 camera and 
 
 extract the depth distance and the X & Y coordinates of the object being measured
 
 It sends the data to a MQTT server to be used by other applications
 
 I have also included an application from a HMI programm called "Quick HMI"
 
 that application is displaying the data.
  
 *******************************************************
The application is based on this repo from sterolabs
 
 https://github.com/stereolabs/zed-yolo
 
 After you get YOLO setup just drop
 
 darknet_zed.py
 
fromm this repo  into the folder

**********************************************************

You will need to install the mosquitto MQTT server for this to work.
 
INSTALL MOSQUITTO MQTT SERVER

$sudo apt-add-repository ppa:mosquitto-dev/mosquitto-ppa

$sudo apt-get update

$sudo apt-get install mosquitto

$sudo apt-get install mosquitto-clients

$sudo apt-get install nano

$sudo systemctl status mosquitto

 $sudo service mosquitto stop

ACCESS MOSQUITTO .conf

$sudo nano /etc/mosquitto/mosquitto.conf

COPY AND PASTE THIS in .conf
*************************************
persistence true

persistence_location /var/lib/mosquitto/

log_dest file /var/log/mosquitto/mosquitto.log

listener 1883 0.0.0.0

allow_anonymous true

include_dir /etc/mosquitto/conf.d
********************************************
RESTART SERVER

$sudo service mosquitto start

TEST SERVER
run the following in one terminal

$mosquitto_pub -d -t testTopic -m "Hello world!"

Run the follwing in another terminal

$mosquitto_sub -d -t testTopic

INSTALL PYTHON MQTT APP

$sudo apt-get install python3-pip

$sudo pip3 install paho-mqtt

