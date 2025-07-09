import sys
import os
import math
import time
import threading
import pygame
from pygame.locals import *
import numpy as np
from edge_assistant import process_voice_query, load_config, ensure_dirs, load_agent
import re

# Screen settings
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 480
FPS = 30
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 120, 255)
DARK_BLUE = (0, 80, 200)
LIGHT_BLUE = (100, 180, 255)
GRAY = (200, 200, 200)
BACKGROUND_COLOR = (240, 248, 255)  # Light blue background
PURPLE = (150, 100, 250)  # For agent selection animation
ORANGE = (255, 165, 0)    # For welcome screen
GREEN = (100, 200, 100)   # For response animation

# Wave animation parameters
WAVE_COUNT = 8  # Number of waves shown simultaneously
WAVE_SPEED = 2  # Wave spread speed
MAX_RADIUS = 150  # Maximum radius

# Animation colors for different states
RECORDING_COLOR = (255, 100, 100)  # Red for recording
PROCESSING_COLOR = (100, 100, 255)  # Blue for processing
PLAYBACK_COLOR = (100, 255, 100)    # Green for playback
AGENT_SELECT_COLOR = (150, 100, 250)  # Purple for agent selection
WELCOME_COLOR = (255, 165, 0)       # Orange for welcome screen

# Enable verbose logging
VERBOSE_LOGGING = True

# Timeout for returning to welcome screen (ms)
WELCOME_TIMEOUT = 60000  # 60 seconds (was 30 seconds)

# Maximum number of conversation history items to store
MAX_HISTORY_ITEMS = 10

def log(message):
    """Log a message to the console"""
    if VERBOSE_LOGGING:
        print(f"[GUI] {message}")

class WaveAnimation:
    def __init__(self):
        self.waves = []  # Active waves storage
        self.active = False  # Animation active state
        self.color = LIGHT_BLUE  # Default color
        self.animation_type = "normal"  # Default animation type: normal, record, process, playback, agent_select, welcome
        
    def start(self, animation_type="normal"):
        """Start wave animation with specified type"""
        self.active = True
        self.waves = []  # Clear existing waves
        self.animation_type = animation_type
        
        # Set color based on animation type
        if animation_type == "record":
            self.color = RECORDING_COLOR
        elif animation_type == "process":
            self.color = PROCESSING_COLOR
        elif animation_type == "playback":
            self.color = PLAYBACK_COLOR
        elif animation_type == "agent_select":
            self.color = AGENT_SELECT_COLOR
        elif animation_type == "welcome":
            self.color = WELCOME_COLOR
        else:
            self.color = LIGHT_BLUE
            
        log(f"{animation_type} animation started")
        
    def stop(self):
        """Stop wave animation"""
        self.active = False
        log("Wave animation stopped")
    
    def update(self):
        """Update wave states"""
        # If animation is active, randomly add new waves
        if self.active and len(self.waves) < WAVE_COUNT:
            chance = 0.2  # Default chance
            
            # Adjust wave frequency based on animation type
            if self.animation_type == "record":
                chance = 0.3  # More frequent waves when recording
            elif self.animation_type == "process":
                chance = 0.15  # Less frequent waves when processing
            elif self.animation_type == "playback":
                chance = 0.25  # Medium frequency when playing
            elif self.animation_type == "agent_select":
                chance = 0.4  # More frequent waves during agent selection
            elif self.animation_type == "welcome":
                chance = 0.2  # Standard frequency for welcome screen
                
            if np.random.random() < chance:
                # Different wave characteristics for different modes
                if self.animation_type == "record":
                    # Recording mode: fast, short waves
                    self.waves.append({
                        'radius': 5,
                        'alpha': 255,
                        'speed': WAVE_SPEED * (1.0 + 0.5 * np.random.random())
                    })
                elif self.animation_type == "process":
                    # Processing mode: slow, long waves
                    self.waves.append({
                        'radius': 15,
                        'alpha': 255,
                        'speed': WAVE_SPEED * (0.6 + 0.3 * np.random.random())
                    })
                elif self.animation_type == "playback":
                    # Playback mode: randomly sized waves
                    self.waves.append({
                        'radius': 10 + np.random.randint(0, 10),
                        'alpha': 255,
                        'speed': WAVE_SPEED * (0.8 + 0.4 * np.random.random())
                    })
                elif self.animation_type == "agent_select":
                    # Agent selection: pulsating waves
                    self.waves.append({
                        'radius': 5 + np.random.randint(0, 15),
                        'alpha': 255,
                        'speed': WAVE_SPEED * (1.2 + 0.5 * np.random.random())
                    })
                elif self.animation_type == "welcome":
                    # Welcome screen: smooth waves
                    self.waves.append({
                        'radius': 20 + np.random.randint(0, 5),
                        'alpha': 200,
                        'speed': WAVE_SPEED * (0.7 + 0.3 * np.random.random())
                    })
                else:
                    # Default mode
                    self.waves.append({
                        'radius': 10,
                        'alpha': 255,
                        'speed': WAVE_SPEED * (0.8 + 0.4 * np.random.random())
                    })
            
        # Update existing waves
        for wave in self.waves[:]:
            wave['radius'] += wave['speed']
            wave['alpha'] = max(0, 255 * (1 - wave['radius'] / MAX_RADIUS))
            
            # If wave is completely transparent, remove it
            if wave['alpha'] <= 0:
                self.waves.remove(wave)
    
    def draw(self, screen, center_x, center_y):
        """Draw waves"""
        for wave in self.waves:
            # Create a surface with alpha channel
            wave_surface = pygame.Surface((wave['radius']*2, wave['radius']*2), pygame.SRCALPHA)
            
            # Draw semi-transparent circle with current color
            pygame.draw.circle(
                wave_surface, 
                (*self.color, int(wave['alpha'])),
                (wave['radius'], wave['radius']), 
                wave['radius'],
                width=2
            )
            
            # Draw to screen
            screen.blit(
                wave_surface, 
                (center_x - wave['radius'], center_y - wave['radius'])
            )

class Button:
    def __init__(self, x, y, radius, color, hover_color, text=''):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.hover_color = hover_color
        self.text = text
        self.is_hovered = False
        self.icon_type = "mic"  # Default icon is mic, can be "start"
        
    def draw(self, screen, font):
        # Draw button circle
        current_color = self.hover_color if self.is_hovered else self.color
        pygame.draw.circle(screen, current_color, (self.x, self.y), self.radius)
        
        # Draw appropriate icon based on type
        if self.icon_type == "mic":
            # Microphone icon
            pygame.draw.rect(screen, WHITE, (self.x-10, self.y-15, 20, 30), border_radius=5)
            pygame.draw.rect(screen, WHITE, (self.x-15, self.y+15, 30, 5), border_radius=2)
        elif self.icon_type == "start":
            # Play/start icon (triangle)
            points = [
                (self.x-10, self.y-15),
                (self.x+15, self.y),
                (self.x-10, self.y+15)
            ]
            pygame.draw.polygon(screen, WHITE, points)
        
        # If there's text, draw it below the button (not inside)
        if self.text and font:
            text_surface = font.render(self.text, True, BLACK)
            text_rect = text_surface.get_rect(center=(self.x, self.y+self.radius+20))
            screen.blit(text_surface, text_rect)
    
    def is_over(self, pos):
        """Check if mouse is hovering over button"""
        distance = math.sqrt((self.x - pos[0])**2 + (self.y - pos[1])**2)
        return distance <= self.radius

class ConversationHistory:
    def __init__(self, max_items=MAX_HISTORY_ITEMS):
        self.items = []
        self.max_items = max_items
        self.current_agent = ""
    
    def add(self, query, response, agent):
        """Add a new conversation item"""
        self.items.append({
            'query': query,
            'response': response,
            'agent': agent,
            'timestamp': time.strftime("%H:%M:%S")
        })
        # Keep only the most recent items
        if len(self.items) > self.max_items:
            self.items.pop(0)
    
    def get_recent_context(self, count=3):
        """Get the most recent context items for the LLM"""
        recent = self.items[-count:] if len(self.items) > 0 else []
        context = ""
        for item in recent:
            context += f"User: {item['query']}\nAssistant ({item['agent']}): {item['response']}\n\n"
        return context
    
    def clear(self):
        """Clear the conversation history"""
        self.items = []

class EdgeAssistantGUI:
    def __init__(self):
        # Initialize Pygame
        pygame.init()
        pygame.display.set_caption("Edge Voice Assistant")
        
        # Screen setup
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Font setup
        pygame.font.init()
        
        # Load fonts
        try:
            font_path = "./fonts/NotoSansSC-Regular.otf"  # Use Noto Sans SC font
            if os.path.exists(font_path):
                self.font = pygame.font.Font(font_path, 24)
                self.small_font = pygame.font.Font(font_path, 18)
                self.big_font = pygame.font.Font(font_path, 30)
                self.title_font = pygame.font.Font(font_path, 36)
                self.subtitle_font = pygame.font.Font(font_path, 28)
                log("Loaded font")
            else:
                log(f"Font file not found: {font_path}, using system default font")
                self.font = pygame.font.SysFont(None, 24)
                self.small_font = pygame.font.SysFont(None, 18)
                self.big_font = pygame.font.SysFont(None, 30)
                self.title_font = pygame.font.SysFont(None, 36)
                self.subtitle_font = pygame.font.SysFont(None, 28)
        except Exception as e:
            log(f"Failed to load font: {e}, using system default font")
            self.font = pygame.font.SysFont(None, 24)
            self.small_font = pygame.font.SysFont(None, 18)
            self.big_font = pygame.font.SysFont(None, 30)
            self.title_font = pygame.font.SysFont(None, 36)
            self.subtitle_font = pygame.font.SysFont(None, 28)
        
        # Animation setup
        self.wave_animation = WaveAnimation()
        
        # UI state
        self.is_recording = False
        self.is_processing = False
        self.is_selecting_agent = False
        self.show_welcome = True
        self.current_screen = "welcome"  # Initial screen set to welcome
        self.status_message = "Click to start"
        self.detected_query = ""
        self.show_query = False
        self.selected_agent = ""
        self.last_response = ""
        self.show_history = False  # Whether to display history panel
        
        # Button setup
        self.record_button = Button(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 100, 40, BLUE, DARK_BLUE)
        
        # Welcome screen button
        self.start_button = Button(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 80, 50, ORANGE, DARK_BLUE)
        self.start_button.icon_type = "start"
        
        # Conversation history
        self.conversation_history = ConversationHistory(max_items=MAX_HISTORY_ITEMS)
        
        # Fix: Clear any previous conversation history from file
        self.clear_previous_history()
        
        # Timeout tracking
        self.last_interaction_time = pygame.time.get_ticks()
        
        # Processing timeout tracking
        self.processing_start_time = 0  # Processing start time
        self.processing_timeout = 45000  # Reduced to 45 seconds (was 60 seconds)
        self.status_update_time = 0  # Last status update time
        self.status_update_timeout = 8000  # Reduced to 8 seconds (was 10 seconds)
        
        # Version information
        self.version = "v1.1"
        
        # Ensure required directories exist
        ensure_dirs()
        
        log("GUI initialized")
    
    # Fix: Add method to clear previous conversation history
    def clear_previous_history(self):
        """Clear any previous conversation history from file"""
        try:
            # Delete conversation context file if it exists
            if os.path.exists("conversation_context.json"):
                os.remove("conversation_context.json")
                log("Cleared previous conversation history")
        except Exception as e:
            log(f"Failed to clear conversation history: {e}")

    def handle_events(self):
        """Handle events"""
        for event in pygame.event.get():
            if event.type == QUIT:
                return False
            
            elif event.type == MOUSEMOTION:
                # Check mouse hover for current screen
                if self.current_screen == "welcome":
                    self.start_button.is_hovered = self.start_button.is_over(event.pos)
                else:  # "voice" screen
                    self.record_button.is_hovered = self.record_button.is_over(event.pos)
            
            elif event.type == MOUSEBUTTONDOWN:
                # Update interaction time only if not in processing state
                if not self.is_processing:
                    self.last_interaction_time = pygame.time.get_ticks()
                
                # Handle button clicks for current screen
                if self.current_screen == "welcome":
                    if self.start_button.is_over(event.pos):
                        self.switch_to_voice_screen()
                else:  # "voice" screen
                    # Fix: Only respond to click on the record button, not anywhere on screen
                    if self.record_button.is_over(event.pos) and not self.is_processing:
                        self.start_recording()
                    
                    # Toggle history panel only with H key or click in history area
                    if event.pos[0] < 200 and event.pos[1] > 100 and event.pos[1] < 400:
                        self.show_history = not self.show_history
            
            elif event.type == KEYDOWN:
                # Toggle history panel with H key
                if event.key == pygame.K_h:
                    self.show_history = not self.show_history
            
            elif event.type == MOUSEBUTTONUP:
                # End recording if in recording state
                if self.is_recording and self.record_button.is_over(event.pos):
                    # Fix: Only respond if released on the button
                    self.stop_recording()
        
        # IMPROVED: Only check for timeout when not processing or recording
        # This ensures the system won't return to welcome screen while working
        current_time = pygame.time.get_ticks()
        if (self.current_screen == "voice" and 
            not self.is_recording and 
            not self.is_processing and
            not self.is_selecting_agent and
            current_time - self.last_interaction_time > WELCOME_TIMEOUT):
            log(f"Timeout reached ({WELCOME_TIMEOUT/1000}s), returning to welcome screen")
            self.switch_to_welcome_screen()
        
        return True
    
    def switch_to_welcome_screen(self):
        """Switch to welcome screen"""
        self.current_screen = "welcome"
        self.wave_animation.stop()  # No animation on welcome screen
        # Don't clear conversation history when returning to welcome screen
        log("Switched to welcome screen")
    
    def switch_to_voice_screen(self):
        """Switch to voice interface screen"""
        self.current_screen = "voice"
        self.wave_animation.stop()
        self.status_message = "Click the button to ask a question"
        self.last_interaction_time = pygame.time.get_ticks()
        self.show_query = False  # Hide any previous query when switching to voice screen
        log("Switched to voice screen")
    
    def start_recording(self):
        """Start recording"""
        self.is_recording = True
        self.status_message = "Recording... (click again to stop)"
        self.wave_animation.start(animation_type="record")
        log("Recording started")
    
    def stop_recording(self):
        """Stop recording and start processing"""
        self.is_recording = False
        self.is_processing = True
        self.status_message = "Processing..."
        self.wave_animation.start(animation_type="process")
        
        log("Recording stopped, starting processing thread")
        # Start processing thread
        threading.Thread(target=self.process_query, daemon=True).start()
    
    def append_context_to_input_file(self, query):
        """Append conversation context to input file for better context-aware responses"""
        try:
            # Create a new input file with context
            with open("deepseek_input.txt", "r") as f:
                content = f.read()
            
            # Extract the prompt part (everything before User:)
            if "User:" in content:
                prompt_part = content.split("User:")[0].strip()
                
                # Get recent conversation context
                context = self.conversation_history.get_recent_context()
                
                # Create new content with context and current query
                new_content = f"{prompt_part}\n\nPrevious conversation:\n{context}\nUser: {query}\nAssistant:"
                
                # Write back to file
                with open("deepseek_input.txt", "w") as f:
                    f.write(new_content)
                
                log("Added conversation context to input file")
        except Exception as e:
            log(f"Error adding context to input file: {e}")
    
    def process_query(self):
        """Process voice query (run in separate thread)"""
        try:
            # Start timing the entire process
            process_start_time = time.time()
            log("Starting voice processing workflow")
            
            # Phase 1: Preparation before calling process_voice_query
            # Clear previous query and result
            self.detected_query = ""
            self.selected_agent = ""
            self.last_response = ""
            self.show_query = False
            self.processing_start_time = pygame.time.get_ticks()  # Record processing start time
            
            # Create status file to track progress
            with open("processing_status.txt", "w") as f:
                f.write("STARTED\n")
            
            # Update interface to show recording status
            self.status_message = "Recording..."
            self.wave_animation.start(animation_type="record")
            
            # Set up watchdog timer to avoid getting stuck
            watchdog_timer = threading.Timer(
                self.processing_timeout / 1000,  # Convert to seconds
                self.handle_processing_timeout
            )
            watchdog_timer.daemon = True
            watchdog_timer.start()
            
            # Core processing function call - this will handle the entire voice processing workflow
            log("Calling process_voice_query()")
            asr_start_time = time.time()
            result = process_voice_query()
            process_end_time = time.time()
            total_process_time = process_end_time - process_start_time
            log(f"Voice processing completed in {total_process_time:.2f} seconds")
            
            # Cancel the watchdog timer
            watchdog_timer.cancel()
            
            # Read the query text after processing - IMPROVED: Extract only user input
            try:
                with open("deepseek_input.txt", "r") as f:
                    content = f.read()
                    # Extract only the user's query, not the entire prompt
                    if "User:" in content:
                        user_parts = content.split("User:")
                        if len(user_parts) > 1:
                            # Get the last user query and remove any Assistant: parts
                            user_query = user_parts[-1].split("\n")[0].strip()
                            if "Assistant:" in user_query:
                                user_query = user_query.split("Assistant:")[0].strip()
                            self.detected_query = user_query
                            log(f"Extracted user query: {self.detected_query}")
                            self.show_query = True
            except Exception as e:
                log(f"Unable to read query: {e}")
                self.detected_query = "..."
            
            # Get the selected agent
            try:
                with open("debug_deepseek_response.txt", "r") as f:
                    content = f.read()
                    for line in content.split('\n'):
                        if line.startswith("Using agent:") or line.startswith("使用智能体:"):
                            self.selected_agent = line.split(":")[1].strip()
                            log(f"Selected agent: {self.selected_agent}")
                            break
            except Exception as e:
                log(f"Unable to read selected agent: {e}")
                self.selected_agent = "default"
            
            # Load agent information to get display name
            try:
                agent_data = load_agent(self.selected_agent)
                if agent_data:
                    agent_display_name = agent_data.get("name", self.selected_agent)
                    self.selected_agent = agent_display_name
                    log(f"Using agent display name: {agent_display_name}")
            except Exception as e:
                log(f"Unable to load agent data: {e}")
            
            # Get the generated response
            try:
                with open("debug_deepseek_response.txt", "r") as f:
                    content = f.read()
                    response_part = ""
                    if "DeepSeek完整响应:" in content:
                        response_part = content.split("DeepSeek完整响应:")[1].strip()
                    elif "DeepSeek full response:" in content:
                        response_part = content.split("DeepSeek full response:")[1].strip()
                    
                    self.last_response = response_part
                    
                    # Add to conversation history - only if we have both query and response
                    if self.detected_query and response_part:
                        self.conversation_history.add(
                            self.detected_query,
                            response_part,
                            self.selected_agent
                        )
                        log(f"Added to conversation history: Q: {self.detected_query}, A: {response_part[:30]}...")
            except Exception as e:
                log(f"Unable to extract response: {e}")
                self.last_response = "Sorry, I couldn't process your request properly."
            
            # Final phase: Play TTS response
            if result:
                # Check TTS output file to confirm playback duration
                try:
                    import soundfile as sf
                    audio_data, sample_rate = sf.read("tts_output.wav")
                    # Calculate playback duration (seconds)
                    playback_duration = len(audio_data) / sample_rate
                    log(f"TTS output duration: {playback_duration:.2f} seconds")
                    
                    # Show animation during playback
                    self.wave_animation.start(animation_type="playback")
                    self.status_message = f"Agent '{self.selected_agent}' is responding..."
                    
                    # OPTIMIZATION: Reduce waiting time for playback
                    # Playback time at most 8 seconds (was 10 seconds), and dynamically adjusted based on audio length
                    max_wait_time = min(8, playback_duration)  # Reduced from 10 to 8 seconds
                    
                    # More intelligently wait based on audio length
                    if playback_duration < 3:
                        # Short audio, wait 80% of actual duration
                        wait_time = playback_duration * 0.8
                        log(f"Short audio response: waiting {wait_time:.2f} seconds (80% of duration)")
                        time.sleep(wait_time)
                    elif playback_duration < 6:
                        # Medium length, wait 70% of actual duration
                        wait_time = playback_duration * 0.7
                        log(f"Medium audio response: waiting {wait_time:.2f} seconds (70% of duration)")
                        time.sleep(wait_time)
                    else:
                        # Longer audio, use fixed shorter wait time
                        log(f"Long audio response: waiting {max_wait_time:.2f} seconds (fixed maximum)")
                        time.sleep(max_wait_time)
                    
                except Exception as e:
                    log(f"Unable to read TTS file duration: {e}")
                    # Use shorter default wait time
                    self.wave_animation.start(animation_type="playback") 
                    self.status_message = "Playing response..."
                    time.sleep(1.5)  # Reduced from 2 seconds to 1.5 seconds
                
                self.wave_animation.stop()
                self.status_message = "Complete. Click again to ask a question"
                
                # Calculate and log total processing time
                total_time = time.time() - process_start_time
                log(f"Total processing pipeline completed in {total_time:.2f} seconds")
                
                # IMPROVED: Only now reset last interaction time to start the 30-second countdown
                # This ensures timeout only starts counting after the response is complete
                self.last_interaction_time = pygame.time.get_ticks()
                
                # Reset processing timeout timer
                self.processing_start_time = 0
                
                # Show query display after processing is complete
                self.show_query = True
            else:
                self.wave_animation.stop()
                self.status_message = "Processing failed. Please try again"
                log("Voice processing failed")
                # Also reset interaction time on failure
                self.last_interaction_time = pygame.time.get_ticks()
        except Exception as e:
            self.wave_animation.stop()
            self.status_message = "Error: Processing failed"
            log(f"Error details: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Ensure status is correctly reset
            self.is_processing = False
            self.is_selecting_agent = False
            self.is_recording = False
            self.processing_start_time = 0
            # Update processing status
            try:
                with open("processing_status.txt", "w") as f:
                    f.write("PROCESS_FINISHED\n")
            except:
                pass
    
    # Improved: Translate Chinese messages to English for consistency
    def handle_processing_timeout(self):
        """Function called when processing times out"""
        if self.is_processing:
            log("Processing timeout callback triggered")
            self.is_processing = False
            self.is_recording = False
            self.is_selecting_agent = False
            self.status_message = "Processing timeout, please try again"
            self.wave_animation.stop()
            self.processing_start_time = 0
            
            # Try to force update status file
            try:
                with open("processing_status.txt", "w") as f:
                    f.write("WATCHDOG_TIMEOUT\n")
            except:
                pass
    
    def update(self):
        """Update interface state"""
        # Update wave animation
        self.wave_animation.update()
        
        # Check for processing timeout
        if self.is_processing:
            current_time = pygame.time.get_ticks()
            
            # If this is the first time we detect processing state, record start time
            if self.processing_start_time == 0:
                self.processing_start_time = current_time
                self.status_update_time = current_time
                log("Starting processing timeout timer")
                
            # Check if we've exceeded total processing timeout
            if current_time - self.processing_start_time > self.processing_timeout:
                log(f"Processing timeout ({self.processing_timeout/1000}s), resetting state")
                self.is_processing = False
                self.is_recording = False
                self.is_selecting_agent = False
                self.status_message = "Processing timeout, please try again"
                self.wave_animation.stop()
                self.processing_start_time = 0
                
                # Try to force update status file
                try:
                    with open("processing_status.txt", "w") as f:
                        f.write("TIMEOUT\n")
                except:
                    pass
                return
                
            # Check if status update has timed out
            try:
                status_file_modified = os.path.getmtime("processing_status.txt")
                current_time_seconds = time.time()
                
                # If status file hasn't been updated for a long time (over 20 seconds)
                if current_time_seconds - status_file_modified > 20:
                    self.status_update_time = current_time
                    log(f"Status file not updated for a long time ({int(current_time_seconds - status_file_modified)}s)")
                    # Don't interrupt immediately, just log
            except:
                pass
                
        # If processing, check processing status file
        if self.is_processing:
            try:
                with open("processing_status.txt", "r") as f:
                    status_text = f.read().strip()
                    status_lines = status_text.split("\n")
                    current_status = status_lines[0] if status_lines else ""
                    
                    # Update status file access time
                    self.status_update_time = pygame.time.get_ticks()
                    
                    # Use simplified status processing logic, reduce frequent switching
                    # Only update on major status changes
                    
                    # Handle major status transition points
                    if current_status == "RECORDING" and not self.is_recording:
                        # Switch to recording state
                        self.is_recording = True
                        self.wave_animation.start(animation_type="record")
                        self.status_message = "recording"
                    
                    elif current_status == "RECORDING_COMPLETE" and self.is_recording:
                        # Recording complete
                        self.is_recording = False
                        self.status_message = "processing"
                    
                    elif current_status in ["ASR_PROCESSING", "ASR_COMPLETE", "EMPTY_INPUT", "LANGUAGE_ERROR"]:
                        # Speech recognition phase - try to get detected query
                        if current_status.startswith("ASR_COMPLETE"):
                            for line in status_lines:
                                if line.startswith("QUERY:") and not self.detected_query:
                                    self.detected_query = line[6:].strip()
                                    self.show_query = True
                                    break
                    
                    elif current_status == "AGENT_SELECTING" and not self.is_selecting_agent:
                        # Switch to agent selection state
                        self.is_selecting_agent = True
                        self.wave_animation.start(animation_type="agent_select")
                        self.status_message = "selecting_agent"
                    
                    elif current_status.startswith("AGENT_SELECTED") and self.is_selecting_agent:
                        # Agent selection complete
                        self.is_selecting_agent = False
                        self.status_message = "thinking"
                        
                        # Try to get selected agent, only on first selection
                        if not self.selected_agent:
                            # Get from status file
                            for line in status_lines:
                                if line.startswith("AGENT:"):
                                    self.selected_agent = line[6:].strip()
                                    break
                            
                            # If failed, try to get from debug output
                            if not self.selected_agent:
                                try:
                                    with open("debug_deepseek_response.txt", "r") as f:
                                        content = f.read()
                                        # First try to get display name
                                        display_name_match = re.search(r"Agent display name: (.+)", content)
                                        if not display_name_match:
                                            display_name_match = re.search(r"智能体显示名称: (.+)", content)
                                            
                                        if display_name_match:
                                            self.selected_agent = display_name_match.group(1).strip()
                                        else:
                                            # Try to get agent ID
                                            agent_match = re.search(r"Using agent: (.+)", content)
                                            if not agent_match:
                                                agent_match = re.search(r"使用智能体: (.+)", content)
                                                
                                            if agent_match:
                                                agent_id = agent_match.group(1).strip()
                                                self.selected_agent = agent_id
                                                
                                                # Load agent display name
                                                try:
                                                    agent_data = load_agent(agent_id)
                                                    if agent_data and "name" in agent_data:
                                                        self.selected_agent = agent_data["name"]
                                                except:
                                                    pass
                                except:
                                    pass
                            
                            # Ensure default value
                            if not self.selected_agent:
                                self.selected_agent = "Assistant"
                    
                    elif current_status in ["LLM_PROCESSING", "LLM_THINKING", "CACHE_HIT"]:
                        # Thinking/processing phase - maintain consistent state
                        if self.status_message != "thinking":
                            self.status_message = "thinking"
                            self.wave_animation.start(animation_type="process")
                    
                    elif current_status in ["TTS_PROCESSING", "TTS_COMPLETE"]:
                        # Speech synthesis phase
                        if self.status_message != "synthesizing":
                            self.status_message = "synthesizing"
                            
                    elif current_status == "AUDIO_PLAYING":
                        # Playback phase
                        if self.status_message != "speaking":
                            self.status_message = "speaking"
                            self.wave_animation.start(animation_type="playback")
                    
                    elif current_status == "PROCESS_COMPLETE":
                        # Processing complete
                        self.is_processing = False
                        self.status_message = "complete"
                        # Update interaction time
                        self.last_interaction_time = pygame.time.get_ticks()
                        
                        # Try to get response result, add to conversation history
                        if self.detected_query and not self.last_response:
                            try:
                                with open("debug_deepseek_response.txt", "r") as f:
                                    content = f.read()
                                    response_part = ""
                                    if "DeepSeek完整响应:" in content:
                                        response_part = content.split("DeepSeek完整响应:")[1].strip()
                                    elif "DeepSeek full response:" in content:
                                        response_part = content.split("DeepSeek full response:")[1].strip()
                                    
                                    if response_part:
                                        self.last_response = response_part
                                        self.conversation_history.add(
                                            self.detected_query,
                                            response_part,
                                            self.selected_agent
                                        )
                            except:
                                pass
                    
                    elif current_status.startswith("ERROR") or current_status.startswith("CRITICAL"):
                        # Error state
                        self.is_processing = False
                        self.status_message = "error"
                        
            except Exception as e:
                log(f"Error reading processing status file: {e}")
                
                # Check if status file exists, create if not
                if not os.path.exists("processing_status.txt"):
                    try:
                        with open("processing_status.txt", "w") as f:
                            f.write("UNKNOWN\n")
                        log("Recreated processing status file")
                    except:
                        pass
    
    def draw_welcome_screen(self):
        """Draw welcome screen"""
        # Fill background
        self.screen.fill(BACKGROUND_COLOR)
        
        # Draw title
        title = self.title_font.render("Welcome to Edge Voice Assistant", True, DARK_BLUE)
        title_rect = title.get_rect(center=(SCREEN_WIDTH//2, 80))
        self.screen.blit(title, title_rect)
        
        # Draw subtitle
        subtitle = self.subtitle_font.render("Your AI companion powered by Edge Computing", True, BLUE)
        subtitle_rect = subtitle.get_rect(center=(SCREEN_WIDTH//2, 120))
        self.screen.blit(subtitle, subtitle_rect)
        
        # Draw description text
        descriptions = [
            "Edge Voice Assistant helps you with voice commands",
            "Powered by a lightweight local AI model",
            "Intelligently selects the best agent for your question",
            "Works without internet connection"
        ]
        
        for i, desc in enumerate(descriptions):
            desc_text = self.font.render(desc, True, BLACK)
            desc_rect = desc_text.get_rect(center=(SCREEN_WIDTH//2, 170 + i*30))
            self.screen.blit(desc_text, desc_rect)
        
        # Draw start button (positioned to work well with the text layout)
        self.start_button.draw(self.screen, self.font)
        
        # Draw version info
        version_text = self.font.render(self.version, True, GRAY)
        version_rect = version_text.get_rect(bottomright=(SCREEN_WIDTH-10, SCREEN_HEIGHT-10))
        self.screen.blit(version_text, version_rect)
        
        # Draw conversation continuation notice if history exists
        if self.conversation_history.items:
            history_text = self.font.render("Conversation history will be continued", True, DARK_BLUE)
            history_rect = history_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT-40))
            self.screen.blit(history_text, history_rect)
            
            # Show count of history items
            count_text = self.font.render(f"({len(self.conversation_history.items)} previous messages)", True, BLUE)
            count_rect = count_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT-20))
            self.screen.blit(count_text, count_rect)
    
    def draw_history_panel(self):
        """Draw conversation history panel"""
        if not self.show_history or not self.conversation_history.items:
            return
            
        # Draw semi-transparent panel background
        panel_surface = pygame.Surface((200, 300), pygame.SRCALPHA)
        pygame.draw.rect(panel_surface, (240, 240, 255, 200), (0, 0, 200, 300), border_radius=10)
        self.screen.blit(panel_surface, (10, 100))
        
        # Draw history title
        history_title = self.font.render("Conversation History", True, DARK_BLUE)
        history_title_rect = history_title.get_rect(center=(110, 120))
        self.screen.blit(history_title, history_title_rect)
        
        # Draw history items (last 5)
        display_items = self.conversation_history.items[-5:] if len(self.conversation_history.items) > 5 else self.conversation_history.items
        
        for i, item in enumerate(display_items):
            # Truncate long texts
            query = item['query'][:25] + "..." if len(item['query']) > 25 else item['query']
            response = item['response'][:25] + "..." if len(item['response']) > 25 else item['response']
            
            # Draw query
            query_text = self.small_font.render(f"Q: {query}", True, BLUE)
            query_rect = query_text.get_rect(x=20, y=150 + i*30)
            self.screen.blit(query_text, query_rect)
            
            # Draw response
            response_text = self.small_font.render(f"A: {response}", True, DARK_BLUE)
            response_rect = response_text.get_rect(x=20, y=165 + i*30)
            self.screen.blit(response_text, response_rect)
        
        # Draw hint
        hint_text = self.small_font.render("Press 'H' to hide", True, GRAY)
        hint_rect = hint_text.get_rect(center=(110, 390))
        self.screen.blit(hint_text, hint_rect)
    
    def draw_voice_screen(self):
        """Draw voice interface screen"""
        # Fill background
        self.screen.fill(BACKGROUND_COLOR)
        
        # 绘制顶部信息栏背景
        top_bar_rect = pygame.Rect(0, 0, SCREEN_WIDTH, 60)
        pygame.draw.rect(self.screen, (220, 235, 250), top_bar_rect)
        pygame.draw.line(self.screen, BLUE, (0, 60), (SCREEN_WIDTH, 60), 2)
        
        # Draw title
        title = self.title_font.render("Edge Voice Assistant", True, DARK_BLUE)
        title_rect = title.get_rect(center=(SCREEN_WIDTH//2, 30))
        self.screen.blit(title, title_rect)
        
        # 绘制装饰性元素 - 左侧和右侧的垂直指示条
        left_bar_rect = pygame.Rect(20, 80, 5, SCREEN_HEIGHT - 160)
        right_bar_rect = pygame.Rect(SCREEN_WIDTH - 25, 80, 5, SCREEN_HEIGHT - 160)
        pygame.draw.rect(self.screen, LIGHT_BLUE, left_bar_rect, border_radius=2)
        pygame.draw.rect(self.screen, LIGHT_BLUE, right_bar_rect, border_radius=2)
        
        # 绘制中央永久动画区域 - 创新元素：环形脉冲效果
        center_x = SCREEN_WIDTH//2
        center_y = SCREEN_HEIGHT//2 - 30
        
        # Draw wave animation - 始终保持活跃
        if not self.wave_animation.active:
            self.wave_animation.start(animation_type="normal")
        self.wave_animation.draw(self.screen, center_x, center_y)
        
        # 创新元素：添加一个永久性的环形进度指示器
        # 根据处理阶段显示不同颜色
        if self.is_recording:
            circle_color = RECORDING_COLOR
        elif self.is_selecting_agent:
            circle_color = AGENT_SELECT_COLOR
        elif "thinking" in self.status_message.lower():
            circle_color = PROCESSING_COLOR
        elif "speaking" in self.status_message.lower():
            circle_color = PLAYBACK_COLOR
        else:
            circle_color = LIGHT_BLUE
            
        # 动态变化的外圈 - 创造呼吸效果
        current_time = pygame.time.get_ticks() / 1000.0  # 转换为秒
        pulse_size = 5 * math.sin(current_time * 3.0) + 40  # 脉冲效果
        
        # 环形进度条
        pygame.draw.circle(self.screen, circle_color, (center_x, center_y), 45, width=3)
        pygame.draw.circle(self.screen, (*circle_color, 100), (center_x, center_y), int(pulse_size), width=2)
        
        # 内部装饰
        if self.is_recording:
            # 录音状态：红色麦克风图标
            pygame.draw.circle(self.screen, (255, 240, 240), (center_x, center_y), 25)
            mic_rect = pygame.Rect(center_x-5, center_y-15, 10, 20)
            pygame.draw.rect(self.screen, RECORDING_COLOR, mic_rect, border_radius=5)
            pygame.draw.circle(self.screen, RECORDING_COLOR, (center_x, center_y+10), 7)
        elif self.is_selecting_agent:
            # 选择智能体：旋转的紫色箭头
            arrow_angle = (current_time * 120) % 360
            arrow_length = 20
            for i in range(3):
                angle = arrow_angle + i * 120
                rad = math.radians(angle)
                end_x = center_x + arrow_length * math.cos(rad)
                end_y = center_y + arrow_length * math.sin(rad)
                pygame.draw.line(self.screen, PURPLE, (center_x, center_y), (end_x, end_y), 3)
                pygame.draw.circle(self.screen, PURPLE, (int(end_x), int(end_y)), 5)
        elif "speaking" in self.status_message.lower():
            # 播放状态：音波图标
            wave_height = 15
            for i in range(5):
                height = wave_height * math.sin(current_time * 5 + i)
                pygame.draw.line(self.screen, GREEN, 
                                (center_x - 20 + i*10, center_y - height),
                                (center_x - 20 + i*10, center_y + height), 3)
        else:
            # 默认/思考状态：旋转的蓝色思考图标
            for i in range(8):
                angle = current_time * 60 + i * 45
                rad = math.radians(angle)
                dist = 15 + 5 * math.sin(current_time * 3 + i)
                pos_x = center_x + dist * math.cos(rad)
                pos_y = center_y + dist * math.sin(rad)
                pygame.draw.circle(self.screen, BLUE, (int(pos_x), int(pos_y)), 4)
        
        # Display selected agent if available
        if self.selected_agent:
            # Draw agent indicator with background
            agent_bg = pygame.Rect(SCREEN_WIDTH//2 - 100, SCREEN_HEIGHT//2 - 90, 200, 30)
            pygame.draw.rect(self.screen, LIGHT_BLUE, agent_bg, border_radius=15)
            pygame.draw.rect(self.screen, BLUE, agent_bg, 2, border_radius=15)
            
            agent_text = self.font.render(f"Agent: {self.selected_agent}", True, DARK_BLUE)
            agent_rect = agent_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 75))
            self.screen.blit(agent_text, agent_rect)
            
            # 为选定的智能体添加一个专业领域图标/标签
            agent_specialty = ""
            if "weather" in self.selected_agent.lower():
                agent_specialty = "Weather Expert"
            elif "tech" in self.selected_agent.lower():
                agent_specialty = "Tech Expert"
            elif "default" in self.selected_agent.lower():
                agent_specialty = "General Assistant"
                
            if agent_specialty:
                specialty_bg = pygame.Rect(SCREEN_WIDTH//2 - 50, SCREEN_HEIGHT//2 - 115, 100, 20)
                pygame.draw.rect(self.screen, (255, 240, 200), specialty_bg, border_radius=10)
                pygame.draw.rect(self.screen, ORANGE, specialty_bg, 1, border_radius=10)
                
                specialty_text = self.small_font.render(agent_specialty, True, DARK_BLUE)
                specialty_rect = specialty_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 105))
                self.screen.blit(specialty_text, specialty_rect)
        
        # Show detected query if available and we're in a state to display it
        if self.detected_query and self.show_query:
            # 添加一个文本气泡背景
            query_bg = pygame.Rect(SCREEN_WIDTH//2 - 300, SCREEN_HEIGHT//2 + 30, 600, 60)
            pygame.draw.rect(self.screen, (240, 240, 255), query_bg, border_radius=15)
            pygame.draw.rect(self.screen, BLUE, query_bg, 2, border_radius=15)
            
            query_label = self.small_font.render("Your question:", True, DARK_BLUE)
            label_rect = query_label.get_rect(x=SCREEN_WIDTH//2 - 290, y=SCREEN_HEIGHT//2 + 35)
            self.screen.blit(query_label, label_rect)
            
            query_text = self.font.render(self.detected_query[:50], True, BLACK)
            query_rect = query_text.get_rect(x=SCREEN_WIDTH//2 - 290, y=SCREEN_HEIGHT//2 + 55)
            self.screen.blit(query_text, query_rect)
            
            # If query is too long, show continuation on next line
            if len(self.detected_query) > 50:
                query_cont = self.font.render(f"{self.detected_query[50:100]}{'...' if len(self.detected_query) > 100 else ''}", True, BLACK)
                query_cont_rect = query_cont.get_rect(x=SCREEN_WIDTH//2 - 290, y=SCREEN_HEIGHT//2 + 75)
                self.screen.blit(query_cont, query_cont_rect)
        
        # Draw button (only if not selecting agent)
        if not self.is_selecting_agent:
            self.record_button.draw(self.screen, self.font)
        
        # 创建底部状态栏
        status_bar_rect = pygame.Rect(0, SCREEN_HEIGHT - 80, SCREEN_WIDTH, 80)
        pygame.draw.rect(self.screen, (220, 235, 250), status_bar_rect)
        pygame.draw.line(self.screen, BLUE, (0, SCREEN_HEIGHT - 80), (SCREEN_WIDTH, SCREEN_HEIGHT - 80), 2)
        
        # 简化状态消息处理，根据当前状态显示一条概括性消息
        simple_status = "Ready for your question"
        if self.is_recording:
            simple_status = "Listening to your voice input..."
        elif self.is_selecting_agent:
            simple_status = "Finding the right specialist for you..."
        elif self.selected_agent and not self.is_recording:
            if "thinking" in self.status_message.lower() or "processing" in self.status_message.lower():
                simple_status = f"Agent '{self.selected_agent}' is analyzing your request..."
            elif "speaking" in self.status_message.lower() or "playback" in self.status_message.lower():
                simple_status = f"Agent '{self.selected_agent}' is responding..."
            elif "complete" in self.status_message.lower() or "done" in self.status_message.lower():
                simple_status = "Response complete. Ready for next question."
        elif self.detected_query and not self.selected_agent:
            simple_status = "Processing your voice input..."
            
        # Draw status message with improved styling
        status_bg = pygame.Rect(SCREEN_WIDTH//2 - 250, SCREEN_HEIGHT - 65, 500, 30)
        pygame.draw.rect(self.screen, WHITE, status_bg, border_radius=15)
        pygame.draw.rect(self.screen, BLUE, status_bg, 1, border_radius=15)
        
        status_text = self.font.render(simple_status, True, DARK_BLUE)
        status_rect = status_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT - 50))
        self.screen.blit(status_text, status_rect)
        
        # 添加一个小图标表示当前状态
        status_icon_x = SCREEN_WIDTH//2 - 230
        status_icon_y = SCREEN_HEIGHT - 50
        if self.is_recording:
            # 录音图标 - 红色圆点
            pygame.draw.circle(self.screen, (255, 50, 50), (status_icon_x, status_icon_y), 8)
        elif self.is_selecting_agent:
            # 选择智能体图标 - 紫色三角形
            points = [(status_icon_x-8, status_icon_y+5), (status_icon_x, status_icon_y-8), (status_icon_x+8, status_icon_y+5)]
            pygame.draw.polygon(self.screen, PURPLE, points)
        elif "thinking" in self.status_message.lower():
            # 思考图标 - 蓝色齿轮
            pygame.draw.circle(self.screen, BLUE, (status_icon_x, status_icon_y), 8, width=2)
            for i in range(4):
                angle = i * 90
                rad_angle = math.radians(angle)
                x = status_icon_x + 8 * math.cos(rad_angle)
                y = status_icon_y + 8 * math.sin(rad_angle)
                pygame.draw.line(self.screen, BLUE, (status_icon_x, status_icon_y), (x, y), 2)
        elif "speaking" in self.status_message.lower():
            # 播放图标 - 绿色三角形
            points = [(status_icon_x-5, status_icon_y-7), (status_icon_x-5, status_icon_y+7), (status_icon_x+7, status_icon_y)]
            pygame.draw.polygon(self.screen, GREEN, points)
        
        # 创新元素：底部小型可视化反馈区
        viz_rect = pygame.Rect(10, SCREEN_HEIGHT - 40, 200, 20)
        pygame.draw.rect(self.screen, (230, 230, 240), viz_rect, border_radius=10)
        
        # 动态变化的可视化元素
        current_time = pygame.time.get_ticks() / 1000.0
        for i in range(20):
            height = 5 + 5 * abs(math.sin(current_time * 2 + i * 0.3))
            bar_rect = pygame.Rect(15 + i * 10, SCREEN_HEIGHT - 40 + (10 - height), 6, height)
            if self.is_recording:
                color = (255, 100 + i * 5, 100)
            elif self.is_selecting_agent:
                color = (150, 100, 250 - i * 5)
            elif "thinking" in self.status_message.lower():
                color = (100, 100 + i * 5, 255)
            elif "speaking" in self.status_message.lower():
                color = (100, 200 + i * 2, 100)
            else:
                color = (100 + i * 5, 180, 255)
            pygame.draw.rect(self.screen, color, bar_rect, border_radius=3)
        
        # 精简底部信息
        if self.conversation_history.items:
            history_count = self.small_font.render(f"History: {len(self.conversation_history.items)} items", True, BLUE)
            history_rect = history_count.get_rect(x=220, y=SCREEN_HEIGHT - 35)
            self.screen.blit(history_count, history_rect)
        
        # Draw history panel if enabled
        self.draw_history_panel()
        
        # Draw version info
        version_text = self.small_font.render(self.version, True, GRAY)
        version_rect = version_text.get_rect(bottomright=(SCREEN_WIDTH-10, SCREEN_HEIGHT-10))
        self.screen.blit(version_text, version_rect)
    
    def draw(self):
        """Draw interface based on current screen"""
        if self.current_screen == "welcome":
            self.draw_welcome_screen()
        else:  # "voice" screen
            self.draw_voice_screen()
        
        # Update display
        pygame.display.flip()
    
    def run(self):
        """Run the application"""
        running = True
        
        while running:
            # 处理事件
            running = self.handle_events()
            
            # 更新状态
            self.update()
            
            # 绘制界面
            self.draw()
            
            # 更新显示
            pygame.display.flip()
            
            # 控制帧率
            self.clock.tick(FPS)
        
        pygame.quit()
        sys.exit()

# Entry point when run as standalone program
if __name__ == "__main__":
    log("Starting Edge Voice Assistant GUI")
    app = EdgeAssistantGUI()
    app.run() 