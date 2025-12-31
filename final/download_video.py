import yt_dlp

url = "https://youtu.be/b8QZJ5ZodTs?si=tgN7ICXPM0NxACom"  # your link

ydl_opts = {
    'format': 'mp4',
    'outtmpl': 'crowd_video2.mp4'
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])

print("Download complete!")