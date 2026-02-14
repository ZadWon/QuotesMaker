import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv
import os
import requests
import time

load_dotenv()

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv('cloud_name'),
    api_key=os.getenv('api_key'),
    api_secret=os.getenv('api_secret')
)

def cloudinary_upload(OUTPUT_VIDEO, Video_ID):
    """
    Upload video to Cloudinary and return the public URL
    """
    try:
        # Upload video to Cloudinary
        print(f"Uploading {Video_ID} to Cloudinary...")
        response = cloudinary.uploader.upload(
            OUTPUT_VIDEO,
            resource_type="video",
            public_id=f"quotes-reels/{Video_ID.replace('.mp4', '')}",
            folder="quotes-reels"
        )
        
        video_url = response['secure_url']
        print(f"Video uploaded successfully to Cloudinary: {video_url}")
        return video_url
        
    except Exception as e:
        print(f"An error occurred during Cloudinary upload: {e}")
        return None

# Upload the video to Instagram using the REELS media type
def upload_video_to_instagram(instagram_account_id, video_url, caption, access_token, cover_url=None):
    url = f'https://graph.facebook.com/v21.0/{instagram_account_id}/media'
    params = {
        'access_token': access_token,
        'media_type': 'REELS',
        'video_url': video_url,
        'caption': caption,
    }
    if cover_url:
        params['cover_url'] = cover_url

    response = requests.post(url, data=params)
    data = response.json()
    media_id = data.get('id')

    if media_id:
        print(f"Video uploaded successfully with Media ID: {media_id}")
        return media_id
    else:
        print("Error uploading video to Instagram:", data)
        return None

# Check if the media is ready to be published
def check_media_status(media_id, access_token):
    url = f'https://graph.facebook.com/v21.0/{media_id}?fields=status_code&access_token={access_token}'
    while True:
        response = requests.get(url)
        data = response.json()
        status_code = data.get('status_code')
        print(data)
        if status_code == 'FINISHED':
            print("Media is ready for publishing.")
            return True
        elif status_code == 'ERROR':
            print("Media processing failed:", data)
            return False
        else:
            print("status code is:", status_code)
            print("Media is still being processed, waiting for 10 seconds...")
            time.sleep(10)  # Wait for 10 seconds before checking again

# Publish the uploaded Reel
def publish_instagram_reel(instagram_account_id, media_id, access_token):
    url = f'https://graph.facebook.com/v21.0/{instagram_account_id}/media_publish'
    params = {
        'access_token': access_token,
        'creation_id': media_id,
    }
    response = requests.post(url, data=params)
    data = response.json()

    if 'id' in data:
        print(f"Reel published successfully with ID: {data['id']}")
    else:
        print("Error publishing Reel:", data)

def check_and_refresh_token(access_token):
    """
    Check token expiry and provide information about refreshing it.
    Facebook long-lived tokens last 60 days and need to be refreshed.
    """
    url = f'https://graph.facebook.com/v21.0/debug_token'
    params = {
        'input_token': access_token,
        'access_token': access_token
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if 'data' in data:
            token_info = data['data']
            is_valid = token_info.get('is_valid', False)
            expires_at = token_info.get('expires_at', 0)
            
            if is_valid:
                if expires_at == 0:
                    print("Token is valid and does not expire (user access token)")
                else:
                    import datetime
                    expiry_date = datetime.datetime.fromtimestamp(expires_at)
                    days_remaining = (expiry_date - datetime.datetime.now()).days
                    print(f"Token is valid. Expires on: {expiry_date}")
                    print(f"Days remaining: {days_remaining}")
                    
                    if days_remaining < 7:
                        print("⚠️ Warning: Token expires soon! Consider refreshing it.")
            else:
                print("❌ Token is invalid or expired!")
                
            return token_info
        else:
            print("Error checking token:", data)
            return None
            
    except Exception as e:
        print(f"Error checking token: {e}")
        return None

def create_caption(result_labels):
    caption = ""
    for result in result_labels[:10]:
        caption = caption + "#" + result.replace(" ", "_") + " "

    return caption

def push_to_instagram(Video_ID, video_url, quote, video_caption, instagram_account_id, access_token, cover_url=None):
    print(f"Publishing to Instagram: {Video_ID}")

    caption = "Always remember: You control what you see, and what you see shapes your thoughts and ideas. 👀💭 \nTap 'like' if you agree and follow us for your daily dose of inspiration!\n \""+ quote +"\"\n" + video_caption
    
    # Check token validity before posting
    check_and_refresh_token(access_token)
    
    # Step 1: Upload and get media ID 
    media_id = upload_video_to_instagram(instagram_account_id, video_url, caption, access_token, cover_url=cover_url)
    if not media_id:
        print("Failed media ID bitch!")


    # Step 2: Wait for the media to be ready
    if not check_media_status(media_id, access_token):
        print("Failed media check bitch!")


    # Step 3: Publish the uploaded video as a Reel
    publish_instagram_reel(instagram_account_id, media_id, access_token)



def post_to_instagram(sample_quote, caption, Video_ID, OUTPUT_VIDEO):
    load_dotenv()
    instagram_account_id = os.getenv('instagram_account_id')
    access_token = os.getenv('access_token')
    
    # Upload video to Cloudinary
    video_url = cloudinary_upload(OUTPUT_VIDEO, Video_ID)
    
    if not video_url:
        print("Failed to upload video to Cloudinary!")
        return
    
    # Push to Instagram with Cloudinary URL
    push_to_instagram(Video_ID, video_url, sample_quote, caption, instagram_account_id, access_token)

