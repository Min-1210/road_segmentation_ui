@echo off
title Road Segmentation Launcher
echo Dang khoi dong ung dung... Vui long cho...


call C:\Users\Weed\anaconda3\Scripts\activate.bat C:\Users\Weed\anaconda3

call conda activate Map

start /B pythonw gui_1.py

exit