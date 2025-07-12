#!/usr/bin/env python3
"""
Command-line interface for AI GIF Generator
"""

import click
import os
import sys
from pathlib import Path
from aigif_processor import AIGifProcessor
import json
from datetime import datetime

@click.command()
@click.argument('video_path', type=click.Path(exists=True, readable=True))
@click.argument('prompt', type=str)
@click.option('--output', '-o', type=click.Path(), help='Output GIF file path')
@click.option('--max-scenes', '-s', default=3, type=int, help='Maximum number of scenes to include')
@click.option('--api-key', '-k', type=str, help='OpenAI API key (overrides environment variable)')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--no-captions', is_flag=True, help='Disable captions on GIF')
@click.option('--json-output', '-j', is_flag=True, help='Output results as JSON')
def generate_gif(video_path, prompt, output, max_scenes, api_key, verbose, no_captions, json_output):
    """
    Generate an AI-powered GIF from a video file based on a text prompt.
    
    VIDEO_PATH: Path to the input video file
    PROMPT: Description of what moments to find and extract
    
    Examples:
        python cli.py video.mp4 "Find exciting action moments"
        python cli.py video.mp4 "Show funny reactions" --output funny.gif
        python cli.py video.mp4 "Capture emotional scenes" --max-scenes 5 --verbose
    """
    
    if verbose:
        import logging
        logging.basicConfig(level=logging.INFO)
        click.echo(f"Processing video: {video_path}")
        click.echo(f"Prompt: {prompt}")
    
    # Initialize processor
    processor = AIGifProcessor(openai_api_key=api_key)
    
    # Set output path if not provided
    if not output:
        video_name = Path(video_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = f"{video_name}_aigif_{timestamp}.gif"
    
    try:
        # Process video
        with click.progressbar(length=100, label='Processing video') as bar:
            # This is a simplified progress bar - in reality, the processing happens inside the processor
            result = processor.process_video_to_gif(
                video_path=str(video_path),
                user_prompt=prompt,
                output_path=str(output),
                max_scenes=max_scenes
            )
            bar.update(100)
        
        if result['success']:
            if json_output:
                click.echo(json.dumps(result, indent=2))
            else:
                click.echo(f"\n✅ Success! GIF generated: {result['gif_path']}")
                click.echo(f"📊 Scenes analyzed: {result['total_scenes_analyzed']}")
                click.echo(f"🎯 Scenes matched: {result['matched_scenes']}")
                click.echo(f"🎬 Scenes selected: {result['selected_scenes']}")
                
                if verbose:
                    click.echo("\n📋 Scene Details:")
                    for i, scene in enumerate(result['scene_details']):
                        click.echo(f"  Scene {i+1}: {scene['duration']:.1f}s "
                                 f"(relevance: {scene['relevance_score']:.2f})")
                        click.echo(f"    Reason: {scene['match_reason']}")
                        if scene.get('suggested_caption'):
                            click.echo(f"    Caption: {scene['suggested_caption']}")
        else:
            click.echo(f"❌ Error: {result['error']}", err=True)
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Unexpected error: {str(e)}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

@click.command()
@click.argument('video_path', type=click.Path(exists=True, readable=True))
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--json-output', '-j', is_flag=True, help='Output results as JSON')
def analyze_video(video_path, verbose, json_output):
    """
    Analyze a video file to extract scene information without generating a GIF.
    
    VIDEO_PATH: Path to the input video file
    
    Examples:
        python cli.py analyze video.mp4
        python cli.py analyze video.mp4 --verbose --json-output
    """
    
    if verbose:
        import logging
        logging.basicConfig(level=logging.INFO)
        click.echo(f"Analyzing video: {video_path}")
    
    # Initialize processor
    processor = AIGifProcessor()
    
    try:
        # Analyze video
        with click.progressbar(length=100, label='Analyzing video') as bar:
            analysis = processor.analyze_video_content(str(video_path))
            bar.update(100)
        
        if json_output:
            click.echo(json.dumps(analysis, indent=2))
        else:
            click.echo(f"\n📊 Video Analysis Results:")
            click.echo(f"Duration: {analysis['duration']:.1f} seconds")
            click.echo(f"FPS: {analysis['fps']:.1f}")
            click.echo(f"Total scenes: {analysis['total_scenes']}")
            
            if verbose:
                click.echo("\n🎬 Scene Breakdown:")
                for i, scene in enumerate(analysis['scenes']):
                    click.echo(f"  Scene {i+1}: {scene['start_time']:.1f}s - {scene['end_time']:.1f}s "
                             f"(duration: {scene['duration']:.1f}s)")
                    features = scene['visual_features']
                    click.echo(f"    Brightness: {features.get('brightness', 0):.1f}")
                    click.echo(f"    Motion: {scene['motion_intensity']:.2f}")
                    click.echo(f"    Contrast: {features.get('contrast', 0):.1f}")
            
    except Exception as e:
        click.echo(f"❌ Error analyzing video: {str(e)}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

@click.group()
def cli():
    """
    AI GIF Generator - Command Line Interface
    
    Transform videos into expressive, captioned GIFs using AI-powered scene detection.
    """
    pass

# Add commands to the CLI group
cli.add_command(generate_gif, name='generate')
cli.add_command(analyze_video, name='analyze')

@cli.command()
def version():
    """Show version information."""
    click.echo("AI GIF Generator v1.0.0")
    click.echo("Powered by OpenAI and computer vision")

@cli.command()
def examples():
    """Show example usage commands."""
    click.echo("AI GIF Generator - Example Commands\n")
    
    examples = [
        {
            'title': 'Basic GIF Generation',
            'command': 'python cli.py generate video.mp4 "Find exciting action moments"',
            'description': 'Generate a GIF from action scenes'
        },
        {
            'title': 'Custom Output Path',
            'command': 'python cli.py generate video.mp4 "Show funny reactions" --output funny.gif',
            'description': 'Specify output filename'
        },
        {
            'title': 'More Scenes',
            'command': 'python cli.py generate video.mp4 "Capture emotional scenes" --max-scenes 5',
            'description': 'Include up to 5 scenes'
        },
        {
            'title': 'Verbose Output',
            'command': 'python cli.py generate video.mp4 "Find dramatic moments" --verbose',
            'description': 'Show detailed processing information'
        },
        {
            'title': 'JSON Output',
            'command': 'python cli.py generate video.mp4 "Show celebrations" --json-output',
            'description': 'Output results in JSON format'
        },
        {
            'title': 'Video Analysis Only',
            'command': 'python cli.py analyze video.mp4',
            'description': 'Analyze video without generating GIF'
        }
    ]
    
    for example in examples:
        click.echo(f"📝 {example['title']}")
        click.echo(f"   {example['command']}")
        click.echo(f"   {example['description']}\n")

if __name__ == '__main__':
    cli()