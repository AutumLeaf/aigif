import os
import json
from flask import Flask, request, jsonify, render_template, send_file, redirect, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging
from datetime import datetime
from pathlib import Path
import tempfile
import shutil
from aigif_processor import AIGifProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Create directories
UPLOAD_FOLDER = Path('uploads')
OUTPUT_FOLDER = Path('outputs')
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

# Allowed file extensions
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv', 'flv', 'webm', 'mkv'}

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

def cleanup_old_files():
    """Clean up old uploaded files and generated GIFs."""
    try:
        # Remove files older than 24 hours
        import time
        current_time = time.time()
        
        for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
            for file_path in folder.glob('*'):
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > 24 * 60 * 60:  # 24 hours
                        file_path.unlink()
                        logger.info(f"Cleaned up old file: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning up old files: {str(e)}")

@app.route('/')
def index():
    """Main page with upload form."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload and processing."""
    try:
        # Check if file is present
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        user_prompt = request.form.get('prompt', '').strip()
        
        if not user_prompt:
            return jsonify({'error': 'Please provide a prompt describing what you want to find in the video'}), 400
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload a video file.'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{filename}"
        video_path = UPLOAD_FOLDER / safe_filename
        
        file.save(video_path)
        logger.info(f"Video uploaded: {video_path}")
        
        # Process video
        processor = AIGifProcessor()
        
        # Generate output filename
        output_filename = f"gif_{timestamp}.gif"
        output_path = OUTPUT_FOLDER / output_filename
        
        # Process video to GIF
        result = processor.process_video_to_gif(
            str(video_path),
            user_prompt,
            str(output_path),
            max_scenes=3
        )
        
        if result['success']:
            result['download_url'] = f"/download/{output_filename}"
            result['video_filename'] = safe_filename
            
            # Clean up uploaded video
            video_path.unlink()
            
            logger.info(f"GIF generated successfully: {output_path}")
            return jsonify(result)
        else:
            # Clean up uploaded video on error
            if video_path.exists():
                video_path.unlink()
            
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"Error in upload_video: {str(e)}")
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_gif(filename):
    """Download generated GIF."""
    try:
        safe_filename = secure_filename(filename)
        file_path = OUTPUT_FOLDER / safe_filename
        
        if not file_path.exists():
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(file_path, as_attachment=True, download_name=safe_filename)
        
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        return jsonify({'error': 'Download error'}), 500

@app.route('/preview/<filename>')
def preview_gif(filename):
    """Preview generated GIF."""
    try:
        safe_filename = secure_filename(filename)
        file_path = OUTPUT_FOLDER / safe_filename
        
        if not file_path.exists():
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(file_path, mimetype='image/gif')
        
    except Exception as e:
        logger.error(f"Error previewing file: {str(e)}")
        return jsonify({'error': 'Preview error'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/examples')
def examples():
    """Show example prompts and usage."""
    examples = [
        {
            'prompt': 'Find exciting action moments',
            'description': 'Extracts high-motion, dynamic scenes perfect for action GIFs'
        },
        {
            'prompt': 'Show funny or humorous moments',
            'description': 'Identifies comedic scenes and expressions'
        },
        {
            'prompt': 'Capture emotional reactions',
            'description': 'Finds expressive moments and facial reactions'
        },
        {
            'prompt': 'Find peaceful or calm scenes',
            'description': 'Extracts serene, low-motion moments'
        },
        {
            'prompt': 'Show dramatic or intense moments',
            'description': 'Identifies high-contrast, emotionally charged scenes'
        },
        {
            'prompt': 'Find scenes with bright colors',
            'description': 'Extracts visually vibrant and colorful moments'
        }
    ]
    
    return render_template('examples.html', examples=examples)

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({'error': 'File too large. Maximum size is 100MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Clean up old files on startup
    cleanup_old_files()
    
    # Get configuration from environment
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    port = int(os.getenv('PORT', 5000))
    
    logger.info(f"Starting AI GIF Generator on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)