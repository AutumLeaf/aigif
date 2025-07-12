# AI GIF Generator 🎬✨

Transform your videos into expressive, captioned GIFs using AI-powered scene detection and analysis.

## Features

- 🤖 **AI-Powered Scene Detection**: Automatically identifies and extracts the most relevant video moments
- 🎯 **Smart Prompt Matching**: Uses OpenAI's GPT to understand your prompts and match them to video content
- 🎨 **Visual Analysis**: Analyzes brightness, motion, colors, and visual complexity
- 📱 **Modern Web Interface**: Beautiful, responsive web UI with drag-and-drop functionality
- 💻 **Command-Line Interface**: Process videos programmatically with full CLI support
- 🔧 **Highly Configurable**: Customize GIF duration, quality, scene count, and more
- 📊 **Detailed Analytics**: Get insights into scene selection and processing decisions

## Quick Start

### Web Interface

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key
   ```

3. **Run the web server:**
   ```bash
   python app.py
   ```

4. **Open your browser:**
   Visit `http://localhost:5000` and start creating GIFs!

### Command Line Interface

```bash
# Generate a GIF from a video
python cli.py generate video.mp4 "Find exciting action moments"

# Analyze a video without generating GIF
python cli.py analyze video.mp4

# See all available commands
python cli.py --help
```

## Installation

### Prerequisites

- Python 3.8+
- OpenAI API key (optional, for enhanced AI features)
- FFmpeg (for video processing)

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd aigif
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install FFmpeg:**
   ```bash
   # Ubuntu/Debian
   sudo apt update && sudo apt install ffmpeg
   
   # macOS
   brew install ffmpeg
   
   # Windows
   # Download from https://ffmpeg.org/download.html
   ```

4. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

## Usage

### Web Interface

The web interface provides the easiest way to create GIFs:

1. **Upload Video**: Drag and drop or select a video file (MP4, AVI, MOV, etc.)
2. **Enter Prompt**: Describe what moments you want to capture
3. **Generate**: Click "Generate AI GIF" and wait for processing
4. **Download**: Preview and download your generated GIF

**Example Prompts:**
- "Find exciting action moments"
- "Show funny reactions and expressions"
- "Capture emotional or dramatic scenes"
- "Find scenes with bright, vibrant colors"
- "Show moments of celebration or joy"

### Command Line Interface

The CLI provides programmatic access to all features:

```bash
# Basic usage
python cli.py generate input.mp4 "Find action scenes"

# Custom output path and scene count
python cli.py generate input.mp4 "Show funny moments" --output funny.gif --max-scenes 5

# Verbose output with detailed information
python cli.py generate input.mp4 "Capture emotions" --verbose

# JSON output for integration
python cli.py generate input.mp4 "Find celebrations" --json-output

# Video analysis only
python cli.py analyze input.mp4 --verbose
```

### Python API

Use the processor directly in your Python code:

```python
from aigif_processor import AIGifProcessor

# Initialize processor
processor = AIGifProcessor(openai_api_key="your-api-key")

# Process video
result = processor.process_video_to_gif(
    video_path="input.mp4",
    user_prompt="Find exciting action moments",
    output_path="output.gif",
    max_scenes=3
)

if result['success']:
    print(f"GIF generated: {result['gif_path']}")
    print(f"Scenes analyzed: {result['total_scenes_analyzed']}")
    print(f"Scenes matched: {result['matched_scenes']}")
else:
    print(f"Error: {result['error']}")
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for enhanced AI features | None |
| `FLASK_DEBUG` | Enable Flask debug mode | False |
| `PORT` | Web server port | 5000 |
| `MAX_FILE_SIZE_MB` | Maximum upload file size | 100 |
| `MAX_GIF_DURATION_SECONDS` | Maximum GIF duration | 10 |
| `TARGET_FPS` | Target GIF frame rate | 15 |
| `MAX_WIDTH` | Maximum GIF width | 640 |

### Processing Parameters

- **Scene Detection**: Automatically detects scene boundaries using frame difference analysis
- **Visual Analysis**: Extracts brightness, contrast, color saturation, and motion intensity
- **AI Matching**: Uses OpenAI's GPT to match scenes to user prompts (requires API key)
- **Fallback Matching**: Basic keyword matching when OpenAI API is not available

## How It Works

1. **Video Analysis**: The system analyzes your video to detect scenes, extract visual features, and calculate motion intensity
2. **Prompt Understanding**: AI interprets your text prompt to understand what type of moments you're looking for
3. **Scene Matching**: Advanced algorithms match video scenes to your prompt based on visual and contextual relevance
4. **GIF Generation**: Selected scenes are combined into a smooth, optimized GIF with optional captions

## Supported Formats

**Input Video Formats:**
- MP4, AVI, MOV, WMV, FLV, WebM, MKV

**Output Format:**
- Optimized GIF with customizable quality and duration

## API Reference

### AIGifProcessor Class

The main class for processing videos and generating GIFs.

#### Methods

- `analyze_video_content(video_path)`: Analyze video and extract scene information
- `match_scenes_to_prompt(video_analysis, user_prompt)`: Match scenes to user prompt
- `generate_gif(video_path, scenes, output_path, max_scenes, add_captions)`: Generate GIF from scenes
- `process_video_to_gif(video_path, user_prompt, output_path, max_scenes)`: Complete processing pipeline

#### Parameters

- `video_path`: Path to input video file
- `user_prompt`: Text description of desired moments
- `output_path`: Path for output GIF file
- `max_scenes`: Maximum number of scenes to include (default: 3)
- `openai_api_key`: OpenAI API key for enhanced AI features

## Examples

### Web Interface Examples

Visit the `/examples` page for interactive prompt examples and usage tips.

### CLI Examples

```bash
# Find action scenes in a sports video
python cli.py generate sports.mp4 "Find exciting action moments and goals"

# Extract funny moments from a comedy video
python cli.py generate comedy.mp4 "Show funny reactions and jokes" --max-scenes 5

# Create a GIF of emotional scenes
python cli.py generate movie.mp4 "Capture dramatic and emotional moments"

# Analyze video content without generating GIF
python cli.py analyze documentary.mp4 --verbose

# Generate with custom settings
python cli.py generate event.mp4 "Find celebration moments" \
  --output celebration.gif \
  --max-scenes 4 \
  --verbose
```

### Python API Examples

```python
from aigif_processor import AIGifProcessor

# Initialize with OpenAI API key
processor = AIGifProcessor(openai_api_key="your-api-key")

# Process different types of content
results = [
    processor.process_video_to_gif("action.mp4", "Find fight scenes"),
    processor.process_video_to_gif("comedy.mp4", "Show funny moments"),
    processor.process_video_to_gif("nature.mp4", "Find beautiful landscapes")
]

for result in results:
    if result['success']:
        print(f"Generated: {result['gif_path']}")
    else:
        print(f"Error: {result['error']}")
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Troubleshooting

### Common Issues

**OpenAI API Key Issues:**
- Ensure your API key is valid and has sufficient credits
- The system will fall back to basic matching without the API key

**Video Processing Errors:**
- Check that FFmpeg is installed and accessible
- Verify video file format is supported
- Ensure sufficient disk space for processing

**Memory Issues:**
- Reduce video file size or duration
- Lower the `MAX_WIDTH` setting
- Process shorter video segments

**Performance Issues:**
- Use GPU acceleration if available
- Reduce `TARGET_FPS` for faster processing
- Limit `MAX_SCENES` for shorter processing time

## License

MIT License - see LICENSE file for details.

## Credits

- **OpenAI**: For AI-powered scene matching
- **OpenCV**: For computer vision and video processing
- **MoviePy**: For video manipulation and GIF generation
- **Flask**: For the web interface
- **Bootstrap**: For responsive UI components

---

**Made with ❤️ by AI enthusiasts**