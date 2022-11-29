docker run -it --rm \
    -p 0.0.0.0:6006:6006 \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --runtime=nvidia \
    -v ${PWD}:/workspace sgan2ada_ws:latest
