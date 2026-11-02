@echo off
chcp 65001 >nul
rem Edit the camera address below. Results go to the "camera_nomerok" folder.
rem --find: plates to highlight (latin or cyrillic, may be removed).
"%~dp0nomerok.exe" "rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101" -o "%~dp0camera_nomerok" --no-open --find "A123BC77"
pause
