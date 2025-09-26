#set up the rtsp audio stream live inferencing.
1. cd rtsp-audio
2. run docker compose up -d -----> the rtsp server is up now push the audio file
3. ffmpeg -re -stream_loop -1 -i test-audio/test.wav \-c:a aac -b:a 128k -ar 44100 -ac 2 -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/mystream
4. git clone the fb demucs to the folder for dependency modules -----> git clone git@github.com:facebookresearch/demucs.git
5. pip install -e . librosa
6. run live11.py for live colreg signal prediction.

