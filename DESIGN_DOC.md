Voice Activated Video Clipper
Dataflow
•	Inputs:
    •	kick_url := The URL to the streamer you want to AutoClip
    •	activation_phrase := Phrase to trigger the automatic clip
    •	before_buffer_length := amount of video to keep on a rolling buffer
    •	after_buffer_length := how much to record after
    •	resolution := Output video resolution to format video
    •	vram_allocation:= Amount of GPU VRAM to use before offloading buffer to CPU RAM
    •	save_path := where to save
    •	file_name_stub := filename to prepend to incrementor.
•	Outputs:
    •	Mp3 video file of said clip
•	Process
    1.	Url gets pasted, the m3u8 playlist gets extracted from the page and the highest resolution video feed gets extracted. The video gets loaded and the buffer starts filling up, A CPU-bound Whisper model listens for the wake word in the stream audio.
    2.	When the wake word is detected, the after_buffer_length is waited, and then the buffer is deep-copied, and sent for post processing in a queue.
    3.	Post processing using FFMPEG to shrink it for vertical video.
    4.	Save it at the save_path, incrementing the highest pathname number by 1.
    5.	Free deep copy and continue.
    6.	If the playlist ends, try refreshing periodically to see if the stream restarts, or alternatively see if the m3u8 URI has changed.

Coding Language:
Use Rust for low-level optimization, and bare-to-the-metal RAM and VRAM management. 