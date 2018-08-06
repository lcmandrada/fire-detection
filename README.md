# Fire Detection

The project detects fire in a video through image processing using the following functions:
- background subtraction to detect motion
- color analysis to highlight fire color and intensity
- display analysis to filter virtual and static fire
- shape analysis to filter virtual and static fire
- variance analysis to observe the behavior of fire color
- blob detection to extract detected fire
- alarm decision unit to send the final alert

# Build
It can be started by executing python3 on code.py.
```
python3 code.py
```

And it can be built by executing PyInstaller on code.spec.
```
pyinstaller code.spec
```

# Note
Edit the VIDEO variable to contain the path of the video.  
Also, change the \<directory\> of pathex at code.spec to the path of the folder containing the files.