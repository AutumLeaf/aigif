#!/usr/bin/env python3
"""
Setup script for AI GIF Generator
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required.")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version OK: {sys.version}")
    return True

def check_ffmpeg():
    """Check if FFmpeg is installed."""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("✅ FFmpeg is installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ FFmpeg not found")
        print("   Please install FFmpeg:")
        print("   - Ubuntu/Debian: sudo apt install ffmpeg")
        print("   - macOS: brew install ffmpeg")
        print("   - Windows: Download from https://ffmpeg.org/download.html")
        return False

def install_dependencies():
    """Install Python dependencies."""
    print("\n🔧 Installing Python dependencies...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def setup_environment():
    """Set up environment file."""
    env_example = Path('.env.example')
    env_file = Path('.env')
    
    if env_example.exists() and not env_file.exists():
        print("\n⚙️  Setting up environment file...")
        shutil.copy(env_example, env_file)
        print("✅ Created .env file from template")
        print("   Please edit .env with your OpenAI API key")
        return True
    elif env_file.exists():
        print("✅ Environment file already exists")
        return True
    else:
        print("❌ .env.example file not found")
        return False

def create_directories():
    """Create necessary directories."""
    directories = ['uploads', 'outputs', 'temp']
    
    print("\n📁 Creating directories...")
    for dir_name in directories:
        dir_path = Path(dir_name)
        dir_path.mkdir(exist_ok=True)
        print(f"✅ Created {dir_name}/ directory")

def test_installation():
    """Test if the installation works."""
    print("\n🧪 Testing installation...")
    try:
        from aigif_processor import AIGifProcessor
        processor = AIGifProcessor()
        print("✅ AI GIF Processor can be imported")
        
        # Test Flask app import
        import app
        print("✅ Flask app can be imported")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main setup function."""
    print("🎬 AI GIF Generator Setup\n")
    
    success = True
    
    # Check Python version
    if not check_python_version():
        success = False
    
    # Check FFmpeg
    if not check_ffmpeg():
        success = False
    
    if not success:
        print("\n❌ Setup failed. Please fix the above issues and try again.")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Setup environment
    if not setup_environment():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Test installation
    if not test_installation():
        print("\n❌ Installation test failed")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit .env file with your OpenAI API key (optional)")
    print("2. Run the web server: python app.py")
    print("3. Open http://localhost:5000 in your browser")
    print("4. Or use the CLI: python cli.py --help")
    print("\nHappy GIF creating! 🎬✨")

if __name__ == '__main__':
    main()