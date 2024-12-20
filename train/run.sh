slaver="b@192.168.123.47"

./master.sh &

ssh $slaver "cd /home/bigdata/train && ./slaver.sh"