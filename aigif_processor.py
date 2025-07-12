import cv2
import numpy as np
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import openai
from PIL import Image, ImageDraw, ImageFont
import moviepy.editor as mp
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.fx import resize
import imageio
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIGifProcessor:
    """
    AI-powered GIF processor that extracts expressive moments from videos
    based on user prompts and generates captioned GIFs.
    """
    
    def __init__(self, openai_api_key: str = None):
        """Initialize the AI GIF processor."""
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
        else:
            logger.warning("OpenAI API key not provided. AI features will be limited.")
        
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)
        
        # Video processing parameters
        self.min_scene_duration = 0.5  # minimum scene duration in seconds
        self.max_gif_duration = 10.0   # maximum GIF duration in seconds
        self.target_fps = 15           # target FPS for GIF
        self.max_width = 640           # maximum GIF width
        
    def analyze_video_content(self, video_path: str) -> Dict:
        """
        Analyze video content to extract scenes, emotions, and key moments.
        """
        logger.info(f"Analyzing video content: {video_path}")
        
        try:
            # Load video
            video = VideoFileClip(video_path)
            duration = video.duration
            fps = video.fps
            
            # Extract frames for analysis
            frames = []
            timestamps = []
            
            # Sample frames every 0.5 seconds for analysis
            sample_rate = 0.5
            for t in np.arange(0, duration, sample_rate):
                frame = video.get_frame(t)
                frames.append(frame)
                timestamps.append(t)
            
            # Analyze scenes using frame differences
            scene_boundaries = self._detect_scene_boundaries(frames, timestamps)
            
            # Extract features for each scene
            scenes = []
            for i, (start, end) in enumerate(scene_boundaries):
                scene_info = {
                    'id': i,
                    'start_time': start,
                    'end_time': end,
                    'duration': end - start,
                    'frame_count': int((end - start) * fps),
                    'visual_features': self._extract_visual_features(video, start, end),
                    'motion_intensity': self._calculate_motion_intensity(video, start, end)
                }
                scenes.append(scene_info)
            
            video.close()
            
            analysis = {
                'video_path': video_path,
                'duration': duration,
                'fps': fps,
                'total_scenes': len(scenes),
                'scenes': scenes,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Video analysis complete. Found {len(scenes)} scenes.")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing video: {str(e)}")
            raise
    
    def _detect_scene_boundaries(self, frames: List[np.ndarray], timestamps: List[float]) -> List[Tuple[float, float]]:
        """Detect scene boundaries using frame difference analysis."""
        if len(frames) < 2:
            return [(0, timestamps[-1] if timestamps else 0)]
        
        # Calculate frame differences
        differences = []
        for i in range(1, len(frames)):
            diff = np.mean(np.abs(frames[i].astype(float) - frames[i-1].astype(float)))
            differences.append(diff)
        
        # Find scene boundaries using threshold
        threshold = np.mean(differences) + 2 * np.std(differences)
        boundaries = [0]  # Start with first frame
        
        for i, diff in enumerate(differences):
            if diff > threshold:
                boundaries.append(timestamps[i+1])
        
        boundaries.append(timestamps[-1])  # End with last frame
        
        # Create scene intervals
        scenes = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            if end - start >= self.min_scene_duration:
                scenes.append((start, end))
        
        return scenes if scenes else [(0, timestamps[-1])]
    
    def _extract_visual_features(self, video: VideoFileClip, start: float, end: float) -> Dict:
        """Extract visual features from a video segment."""
        try:
            # Get middle frame of the scene
            mid_time = (start + end) / 2
            frame = video.get_frame(mid_time)
            
            # Convert to different color spaces for analysis
            frame_rgb = frame
            frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            frame_hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
            
            # Calculate visual features
            features = {
                'brightness': float(np.mean(frame_gray)),
                'contrast': float(np.std(frame_gray)),
                'color_saturation': float(np.mean(frame_hsv[:, :, 1])),
                'color_dominance': self._get_dominant_colors(frame_rgb),
                'edge_density': self._calculate_edge_density(frame_gray),
                'frame_complexity': self._calculate_frame_complexity(frame_gray)
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting visual features: {str(e)}")
            return {}
    
    def _get_dominant_colors(self, frame: np.ndarray, k: int = 3) -> List[List[int]]:
        """Get dominant colors in the frame."""
        try:
            # Reshape frame for k-means
            pixels = frame.reshape(-1, 3)
            
            # Reduce sample size for performance
            if len(pixels) > 10000:
                indices = np.random.choice(len(pixels), 10000, replace=False)
                pixels = pixels[indices]
            
            # Apply k-means clustering
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get dominant colors
            colors = kmeans.cluster_centers_.astype(int)
            return colors.tolist()
            
        except Exception as e:
            logger.error(f"Error getting dominant colors: {str(e)}")
            return []
    
    def _calculate_edge_density(self, frame_gray: np.ndarray) -> float:
        """Calculate edge density using Canny edge detection."""
        try:
            edges = cv2.Canny(frame_gray, 50, 150)
            return float(np.sum(edges > 0) / edges.size)
        except Exception as e:
            logger.error(f"Error calculating edge density: {str(e)}")
            return 0.0
    
    def _calculate_frame_complexity(self, frame_gray: np.ndarray) -> float:
        """Calculate frame complexity using texture analysis."""
        try:
            # Calculate local binary patterns or use gradient magnitude
            sobelx = cv2.Sobel(frame_gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(frame_gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            return float(np.mean(gradient_magnitude))
        except Exception as e:
            logger.error(f"Error calculating frame complexity: {str(e)}")
            return 0.0
    
    def _calculate_motion_intensity(self, video: VideoFileClip, start: float, end: float) -> float:
        """Calculate motion intensity in the video segment."""
        try:
            # Sample a few frames to calculate motion
            sample_times = np.linspace(start, min(end, start + 2), 5)
            frames = [video.get_frame(t) for t in sample_times]
            
            if len(frames) < 2:
                return 0.0
            
            # Calculate optical flow between consecutive frames
            motion_scores = []
            for i in range(1, len(frames)):
                prev_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_RGB2GRAY)
                curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
                
                # Calculate optical flow
                flow = cv2.calcOpticalFlowPyrLK(
                    prev_gray, curr_gray, 
                    cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7),
                    None
                )
                
                if flow[0] is not None and flow[1] is not None:
                    # Calculate motion magnitude
                    motion_magnitude = np.sqrt(
                        (flow[0] - flow[1])**2).sum(axis=1) if len(flow[0]) > 0 else np.array([0])
                    motion_scores.append(np.mean(motion_magnitude))
            
            return float(np.mean(motion_scores)) if motion_scores else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating motion intensity: {str(e)}")
            return 0.0
    
    def match_scenes_to_prompt(self, video_analysis: Dict, user_prompt: str) -> List[Dict]:
        """
        Use AI to match video scenes to user prompt and rank them by relevance.
        """
        logger.info(f"Matching scenes to prompt: '{user_prompt}'")
        
        if not self.openai_api_key:
            logger.warning("OpenAI API key not available. Using basic matching.")
            return self._basic_scene_matching(video_analysis, user_prompt)
        
        try:
            # Prepare scene descriptions for AI analysis
            scene_descriptions = []
            for scene in video_analysis['scenes']:
                description = self._generate_scene_description(scene)
                scene_descriptions.append({
                    'scene_id': scene['id'],
                    'description': description,
                    'start_time': scene['start_time'],
                    'end_time': scene['end_time'],
                    'duration': scene['duration']
                })
            
            # Use OpenAI to match scenes to prompt
            prompt = f"""
            User Request: "{user_prompt}"
            
            Video Scenes:
            {json.dumps(scene_descriptions, indent=2)}
            
            Please analyze these video scenes and rank them by relevance to the user's request.
            Consider visual elements, motion, timing, and emotional context.
            
            Return a JSON response with the following structure:
            {{
                "matched_scenes": [
                    {{
                        "scene_id": <scene_id>,
                        "relevance_score": <0.0-1.0>,
                        "reason": "<explanation of why this scene matches>",
                        "suggested_caption": "<suggested caption for this scene>",
                        "emotional_tone": "<emotional description>"
                    }}
                ],
                "overall_theme": "<overall theme that matches the prompt>"
            }}
            
            Rank scenes by relevance_score (1.0 = perfect match, 0.0 = no match).
            Only include scenes with relevance_score > 0.3.
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert video analyst who helps match video content to user requests for GIF creation."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            # Parse AI response
            ai_response = json.loads(response.choices[0].message.content)
            
            # Add original scene data to matched scenes
            matched_scenes = []
            for match in ai_response['matched_scenes']:
                scene_id = match['scene_id']
                original_scene = video_analysis['scenes'][scene_id]
                
                enhanced_scene = {
                    **original_scene,
                    'relevance_score': match['relevance_score'],
                    'match_reason': match['reason'],
                    'suggested_caption': match['suggested_caption'],
                    'emotional_tone': match['emotional_tone']
                }
                matched_scenes.append(enhanced_scene)
            
            # Sort by relevance score
            matched_scenes.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            logger.info(f"AI matching complete. Found {len(matched_scenes)} relevant scenes.")
            return matched_scenes
            
        except Exception as e:
            logger.error(f"Error in AI scene matching: {str(e)}")
            return self._basic_scene_matching(video_analysis, user_prompt)
    
    def _generate_scene_description(self, scene: Dict) -> str:
        """Generate a textual description of a scene for AI analysis."""
        features = scene['visual_features']
        
        # Describe brightness
        brightness = features.get('brightness', 0)
        brightness_desc = "bright" if brightness > 150 else "dark" if brightness < 80 else "medium-lit"
        
        # Describe motion
        motion = scene['motion_intensity']
        motion_desc = "high-motion" if motion > 10 else "low-motion" if motion < 2 else "moderate-motion"
        
        # Describe colors
        colors = features.get('color_dominance', [])
        color_desc = "colorful" if len(colors) > 0 else "monochromatic"
        
        # Describe contrast
        contrast = features.get('contrast', 0)
        contrast_desc = "high-contrast" if contrast > 60 else "low-contrast" if contrast < 30 else "medium-contrast"
        
        description = f"A {scene['duration']:.1f}-second {brightness_desc}, {motion_desc}, {color_desc}, {contrast_desc} scene"
        
        return description
    
    def _basic_scene_matching(self, video_analysis: Dict, user_prompt: str) -> List[Dict]:
        """Basic scene matching without AI (fallback method)."""
        logger.info("Using basic scene matching (no AI)")
        
        scenes = video_analysis['scenes']
        
        # Simple keyword matching and scoring
        keywords = user_prompt.lower().split()
        
        scored_scenes = []
        for scene in scenes:
            score = 0.0
            
            # Score based on motion intensity
            if any(word in ['action', 'fast', 'quick', 'moving', 'dynamic'] for word in keywords):
                score += scene['motion_intensity'] * 0.1
            
            # Score based on duration
            if any(word in ['short', 'brief', 'quick'] for word in keywords):
                score += max(0, 1 - scene['duration'] / 5)  # Prefer shorter scenes
            elif any(word in ['long', 'extended', 'slow'] for word in keywords):
                score += min(1, scene['duration'] / 5)  # Prefer longer scenes
            
            # Score based on visual features
            features = scene['visual_features']
            if any(word in ['bright', 'light', 'sunny'] for word in keywords):
                score += features.get('brightness', 0) / 255
            elif any(word in ['dark', 'night', 'shadow'] for word in keywords):
                score += 1 - (features.get('brightness', 0) / 255)
            
            # Add base score for all scenes
            score += 0.5
            
            scored_scenes.append({
                **scene,
                'relevance_score': min(1.0, score),
                'match_reason': 'Basic keyword and feature matching',
                'suggested_caption': f"Scene {scene['id'] + 1}",
                'emotional_tone': 'neutral'
            })
        
        # Sort by relevance score
        scored_scenes.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return scored_scenes
    
    def generate_gif(self, video_path: str, scenes: List[Dict], output_path: str, 
                    max_scenes: int = 3, add_captions: bool = True) -> str:
        """
        Generate a GIF from selected video scenes with optional captions.
        """
        logger.info(f"Generating GIF from {len(scenes)} scenes")
        
        try:
            # Select top scenes
            selected_scenes = scenes[:max_scenes]
            total_duration = sum(scene['duration'] for scene in selected_scenes)
            
            # Adjust scene durations if total is too long
            if total_duration > self.max_gif_duration:
                scale_factor = self.max_gif_duration / total_duration
                for scene in selected_scenes:
                    scene['adjusted_duration'] = scene['duration'] * scale_factor
            else:
                for scene in selected_scenes:
                    scene['adjusted_duration'] = scene['duration']
            
            # Load video
            video = VideoFileClip(video_path)
            
            # Create clips for each scene
            clips = []
            for i, scene in enumerate(selected_scenes):
                # Extract scene clip
                start_time = scene['start_time']
                end_time = scene['end_time']
                
                scene_clip = video.subclip(start_time, end_time)
                
                # Resize to target dimensions
                scene_clip = scene_clip.resize(width=self.max_width)
                
                # Add caption if requested
                if add_captions and scene.get('suggested_caption'):
                    scene_clip = self._add_caption_to_clip(scene_clip, scene['suggested_caption'])
                
                clips.append(scene_clip)
            
            # Concatenate all clips
            if len(clips) > 1:
                final_clip = mp.concatenate_videoclips(clips)
            else:
                final_clip = clips[0]
            
            # Set target FPS
            final_clip = final_clip.set_fps(self.target_fps)
            
            # Write GIF
            final_clip.write_gif(output_path, fps=self.target_fps, opt='OptimizePlus')
            
            # Clean up
            video.close()
            final_clip.close()
            for clip in clips:
                clip.close()
            
            logger.info(f"GIF generated successfully: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating GIF: {str(e)}")
            raise
    
    def _add_caption_to_clip(self, clip: VideoFileClip, caption: str) -> VideoFileClip:
        """Add caption to a video clip."""
        try:
            # Create text clip
            from moviepy.video.tools.drawing import color_gradient
            from moviepy.video.fx import resize
            
            # Simple text overlay (you might want to use a more sophisticated approach)
            # For now, we'll skip complex text rendering and just note the caption
            logger.info(f"Adding caption: {caption}")
            
            # TODO: Implement proper text overlay
            # This would require more complex text rendering with PIL or moviepy's TextClip
            
            return clip
            
        except Exception as e:
            logger.error(f"Error adding caption: {str(e)}")
            return clip
    
    def process_video_to_gif(self, video_path: str, user_prompt: str, 
                           output_path: str = None, max_scenes: int = 3) -> Dict:
        """
        Main method to process a video and generate an AI-powered GIF.
        """
        logger.info(f"Processing video to GIF: {video_path}")
        logger.info(f"User prompt: {user_prompt}")
        
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"output_{timestamp}.gif"
        
        try:
            # Step 1: Analyze video content
            video_analysis = self.analyze_video_content(video_path)
            
            # Step 2: Match scenes to user prompt
            matched_scenes = self.match_scenes_to_prompt(video_analysis, user_prompt)
            
            if not matched_scenes:
                raise ValueError("No relevant scenes found for the given prompt")
            
            # Step 3: Generate GIF
            gif_path = self.generate_gif(video_path, matched_scenes, output_path, max_scenes)
            
            # Step 4: Prepare result
            result = {
                'success': True,
                'gif_path': gif_path,
                'user_prompt': user_prompt,
                'total_scenes_analyzed': len(video_analysis['scenes']),
                'matched_scenes': len(matched_scenes),
                'selected_scenes': min(max_scenes, len(matched_scenes)),
                'scene_details': matched_scenes[:max_scenes],
                'processing_timestamp': datetime.now().isoformat()
            }
            
            logger.info("Video processing complete!")
            return result
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'user_prompt': user_prompt,
                'processing_timestamp': datetime.now().isoformat()
            }


def main():
    """Demo function to test the AI GIF processor."""
    processor = AIGifProcessor()
    
    # Example usage
    video_path = "sample_video.mp4"
    user_prompt = "Find exciting action moments"
    
    if os.path.exists(video_path):
        result = processor.process_video_to_gif(video_path, user_prompt)
        print(json.dumps(result, indent=2))
    else:
        print(f"Sample video not found: {video_path}")
        print("Please provide a video file to test the processor.")


if __name__ == "__main__":
    main()