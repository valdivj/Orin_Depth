# Orin_Depth
 Orin Depth with MQTT
 
 
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

