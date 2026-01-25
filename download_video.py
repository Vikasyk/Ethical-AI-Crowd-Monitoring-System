import yt_dlp

url = "https://youtube.com/shorts/tPmcCje8L-w?si=JCB_6tKhmjCS1RoQ"

ydl_opts = {
    # Force standard MP4 compatibility and high quality
    'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
    'outtmpl': 'crowd.mp4',
    
    # Bypass SABR issues by using different client identities
    'extractor_args': {
        'youtube': {
            'player_client': ['android', 'web'],
        }
    },
    
    # Use ffmpeg to handle fragment downloading and merging
    'external_downloader': 'ffmpeg',
    
    # Prevent the script from finishing with a corrupt/empty file if fragments fail
    'skip_unavailable_fragments': False,
}

try:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    print("Download complete!")
except Exception as e:
    print(f"An error occurred: {e}")