# 🎬 AI ReelsMaker: Automated Quote-to-Video Pipeline

Let’s cut to the results: This script transforms raw quotes into high-quality, social-media-ready videos with dynamic effects, automated branding, and direct-to-Instagram publishing.

**[Click here to see a Sample Video](https://www.instagram.com/p/DH6PrZCOV1c/)**

---

## Quick Start

Follow these steps to set up your local automated content factory:

### 1. Clone & Setup
```bash
# Clone the repository
git clone [https://github.com/ZadWon/QuotesMaker.git](https://github.com/ZadWon/QuotesMaker.git)
cd QuotesMaker

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
### 2. Environment Configuration (.env)
Security is paramount. You must create a .env file in the root directory to store your API keys and credentials. This ensures your secrets aren't pushed to GitHub.

Create a file named .env and add the following:
```
# Cloudinary Credentials
cloud_name=your_cloudinary_name
api_key=your_cloudinary_key
api_secret=your_cloudinary_secret

# Instagram/Meta Graph API
instagram_account_id=your_insta_business_account_id
access_token=your_long_lived_facebook_access_token
```


### 3. Run the Pipeline

```
python main.py
```
Well, you can run the script in a crontab for automation +++ 


## Technical Architecture (For the Engineers)
This project was built to showcase a production-grade automation pipeline, blending NLP, Computer Vision, and Cloud infrastructure.

### Intelligence Layer
[HuggingFace Transformers](https://huggingface.co/docs/transformers/index) : Uses facebook/bart-large-mnli for zero-shot classification, allowing the bot to "understand" and categorize quotes into 80+ distinct emotional niches.

spaCy NLP: Implements linguistic analysis to identify key nouns and proper nouns. These are passed to the renderer to be dynamically bolded for visual emphasis.

### Creative Engine (The "Please HR Hire ME!" Section)
- Pillow (PIL): A custom typography engine that handles complex text wrapping, italic-vs-bold logic, and multi-layer rendering with drop shadows.

- Geometric Masking: Unlike standard fades, we use a Pentagon-shaped dynamic mask. The vertices are mathematically interpolated over time to create a "bloom" reveal effect.

- MoviePy & NumPy VFX: * Waving Effect: Uses SciPy and NumPy to distort frame coordinates via sine waves, creating a custom "dreamy" motion effect.

- Ken Burns Logic: Automated zoom-out/in scaling to maintain viewer retention.

### Cloud & API Integration
Cloudinary: Since the Instagram API requires a publicly accessible URL, the pipeline automatically uploads the video to Cloudinary, manages the asset, and hands the secure URL to the Meta servers.

Instagram Graph API: Handles the three-way handshake: container creation, status polling, and final reel publishing.

SQLite Persistence: Integrated a local database to ensure the bot maintains a "memory" of used quotes, preventing content duplication.

### Documentation Links
[Instagram Graph API Documentation](https://developers.facebook.com/docs/instagram-platform/instagram-api-with-facebook-login)

[Cloudinary Python SDK Guide](https://cloudinary.com/documentation)

[More Examples of Reels (subscribe!!)](https://www.instagram.com/astonishing.quote/)
