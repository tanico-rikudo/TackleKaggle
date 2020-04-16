for i in `seq 0 24`
do
  echo "[$i]" ` date '+%y/%m/%d %H:%M:%S'` "connected."
  open $1 
  sleep 3600
done
