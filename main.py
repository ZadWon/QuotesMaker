from reelMaker import reel_maker
from instagramPublisher import post_to_instagram
from dotenv import load_dotenv
import os 
if __name__ == "__main__":
    # os.chdir('/root/ReelsBot')

    load_dotenv()
    hashtags = [
        "Inspirational",
        "Motivational",
        "Love",
        "Life",
        "Success",
        "Happiness",
        "Friendship",
        "Wisdom",
        "Humor",
        "Philosophy",
        "Education"
    ]
    sample_quote, caption, Video_ID, OUTPUT_VIDEO = reel_maker()
    post_to_instagram(sample_quote, caption, Video_ID, OUTPUT_VIDEO)
    description = "Always remember: You control what you see, and what you see shapes your thoughts and ideas. 👀💭 \nTap 'like' if you agree and follow us for your daily dose of inspiration!\n"
    #publish_to_youtube(OUTPUT_VIDEO, sample_quote, description + caption, hashtags)

# caption = "Always remember: You control what you see, and what you see shapes your thoughts and ideas. 👀💭 \nTap 'like' if you agree and follow us for your daily dose of inspiration!\n \""+ quote +"\"\n" + video_caption
