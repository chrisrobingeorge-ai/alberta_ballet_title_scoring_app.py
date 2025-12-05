import os
import textwrap
import subprocess

try:
    from moviepy.editor import (
        TextClip, CompositeVideoClip, AudioFileClip, 
        ColorClip, concatenate_videoclips
    )
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    TextClip = None
    CompositeVideoClip = None
    AudioFileClip = None
    ColorClip = None
    concatenate_videoclips = None

def generate_tts_audio(text, output_file):
    """
    Generate TTS audio using espeak-ng command line tool.
    This works offline without needing internet access.
    Returns a WAV file that moviepy can directly use.
    """
    # Generate WAV with espeak-ng
    cmd = [
        'espeak-ng',
        '-v', 'en',  # English voice
        '-s', '150',  # Speed (words per minute)
        '-w', output_file,  # Output WAV file
        text
    ]
    result = subprocess.run(cmd, check=True, capture_output=True)
    if result.stderr:
        print(f"  Warning from espeak-ng: {result.stderr.decode()}")
    
    return output_file

def create_scene_clip(scene_num, title, visual_desc, narration_text, output_audio_name):
    """
    Creates a video clip for a single scene with:
    - TTS Audio of the narration.
    - A visual slide containing the Scene Title and Visual Description.
    """
    if not MOVIEPY_AVAILABLE:
        raise ImportError(
            "moviepy is not installed. Install it with: pip install moviepy"
        )
    print(f"  > Processing Scene {scene_num}: {title}...")

    # 1. Generate TTS Audio using espeak-ng (offline)
    audio_file = generate_tts_audio(narration_text, output_audio_name)
    
    # 2. Create Audio Clip
    audio_clip = AudioFileClip(audio_file)
    duration = audio_clip.duration + 1.0  # Add 1s padding
    
    # 3. Create Visual Background (Dark Blue for professional look)
    # Size 1280x720 (720p)
    w, h = 1280, 720
    bg_color = (10, 20, 40) # Dark Navy Blue
    bg_clip = ColorClip(size=(w, h), color=bg_color).set_duration(duration)
    
    # 4. Create Text Overlays
    # Title Text (Top)
    title_str = f"Scene {scene_num}: {title}"
    title_clip = TextClip(
        title_str, 
        fontsize=50, 
        color='gold', 
        font='DejaVu-Sans-Bold',
        size=(w - 100, None), 
        method='caption',
        align='center'
    ).set_position(('center', 50)).set_duration(duration)

    # Visual Description Text (Middle - Italicized representation)
    visual_str = f"[Visual: {visual_desc}]"
    visual_clip = TextClip(
        visual_str, 
        fontsize=30, 
        color='lightgrey', 
        font='DejaVu-Sans',
        size=(w - 150, None), 
        method='caption',
        align='center'
    ).set_position(('center', 'center')).set_duration(duration)

    # Subtitles/Script Snippet (Bottom - showing first few lines of narration)
    # Truncate narration for visual clarity
    short_narration = textwrap.shorten(narration_text, width=200, placeholder="...")
    script_clip = TextClip(
        short_narration, 
        fontsize=24, 
        color='white', 
        font='DejaVu-Sans',
        size=(w - 100, None), 
        method='caption',
        align='center'
    ).set_position(('center', h - 150)).set_duration(duration)

    # 5. Composite
    video = CompositeVideoClip([bg_clip, title_clip, visual_clip, script_clip])
    video = video.set_audio(audio_clip)
    
    return video, audio_file

def main():
    if not MOVIEPY_AVAILABLE:
        print("Error: moviepy is not installed.")
        print("To use video_maker.py, install the video dependencies:")
        print("  pip install -r video_requirements.txt")
        return
    
    # --- SCRIPT DATA ---
    # Format: (Scene Number, Title, Visual Description, Narration)
    script_data = [
        (
            1, "Cold Open: The Problem",
            "Wide shot of a theatre filling up; overlay of a season planning calendar, sticky notes with show titles.",
            "Planning a ballet season is a little bit like choreographing in the dark. Months or even years before opening night, you have to decide which titles to program, in which city, and for how long - long before you know how audiences will respond. For leaders at Alberta Ballet, that's a high-stakes guessing game. Will this title fill the Jubilee in Calgary? Is Edmonton more interested in a classic story ballet or a contemporary work this season? How much marketing do we need to get there? The Alberta Ballet Ticket Estimator was built to shine some light into that dark. It doesn't replace artistic judgment, but it gives decision-makers a disciplined, data-driven view of future single-ticket demand, city by city."
        ),
        (
            2, "What the Tool Actually Is",
            "Simple diagram: User -> App -> Data & Models -> Ticket Estimates (Calgary / Edmonton).",
            'At its core, this is a Streamlit app - a browser-based tool where planners can explore "what if" scenarios for future shows. The app estimates single-ticket sales for each title, split between Calgary and Edmonton. It combines four big ingredients: Online visibility signals like Wikipedia views, Google search interest, YouTube activity, and Spotify popularity. Historical ticket sales from Alberta Ballet\'s own box office history. Economic sentiment, using indicators from the Bank of Canada and Alberta\'s economy. Live analytics, which capture how engaged people are with different types of work right now. Under the hood, it blends clear formulas, a machine-learning model trained on past shows, and a layer of rules and safety checks that keep estimates realistic. In other words: it doesn\'t just say "this feels like a four-thousand-ticket title." It shows why.'
        ),
        (
            3, "Where the Data Comes From",
            "Icons: Wikipedia, Google Trends, YouTube, Spotify, a bar chart of past tickets, and an economic chart.",
            "Let's start with the data sources this app uses. First, it looks at online awareness and interest: Wikipedia views - how often people look up the title or subject. Google Trends - how frequently people search for it, relative to other topics. YouTube views - how many people are watching related videos. Spotify popularity - an index of how popular related music is on the platform. Each of those is turned into a signal index: A Wiki index that grows with daily page views. A Trends index scaled from Google's 0 to 100 search interest. A YouTube index based on median video views, with safeguards so one viral clip doesn't completely dominate. A Spotify index based on how a title's music ranks in popularity. Next, the app uses historical ticket data from Alberta Ballet's own productions. That includes: Past single-ticket sales by show and city. A derived measure called Ticket Index DeSeason - essentially what the typical ticket demand looks like once you strip out the seasonal effect of the month. On top of that, it layers in: Category and segment information - for example, is this a classic story ballet, a contemporary work, a special event? Economic data - indicators from the Bank of Canada and Alberta's economy, combined into an economic sentiment factor. And live analytics from recent audience behaviour, which tell us how engaged people are with different categories of work right now. All of this is wired together through configuration files and feature inventories, so the app knows exactly where each signal comes from and how it should be used."
        ),
        (
            4, "From Raw Signals to Familiarity & Motivation",
            "Split-screen: left side raw signals (views, searches); right side gauges labelled Familiarity and Motivation.",
            "On their own, those raw numbers don't mean much. Is 2,000 YouTube views good or bad? Is 10,000 monthly Wikipedia views a lot? The app solves this by transforming raw data into two core concepts: Familiarity and Motivation. Familiarity asks: How well-known is this title in the wider world? It gives more weight to: Wikipedia views, plus Google Trends, plus Spotify popularity. Motivation asks: How excited are people to engage with it right now? Here, YouTube tends to matter more, because actually watching a video is a stronger sign of interest than just looking something up. The app blends these underlying indices into: a Familiarity score, a Motivation score, and a combined Signal Only score, which is basically the average of the two. Behind the scenes, that's done through explicit formulas. For example, Familiarity puts a higher weight on Wikipedia than Spotify, and Motivation gives YouTube the biggest say. These formulas are documented in the TICKET ESTIMATOR FORMULAS specification and implemented directly in the code. The result is a clean set of numbers - typically centred around 100 for a benchmark title - that tell you how strong the signals are for any show, even before you look at your own history."
        ),
        (
            5, "History Meets Machine Learning",
            "A timeline of past shows with ticket bars; then a simple model icon.",
            "Of course, Alberta Ballet has something even more powerful than online buzz: its own history. For titles that have been produced before, the app calculates a de-seasonalized ticket index. Think of this as: If we strip away the effect of the month or holiday, how strong was the audience response to this title? That gives us a stable, historical benchmark for each show. But what about brand-new titles, or works that are only loosely related to anything in the archive? That's where machine learning comes in. The app trains a regression model on past shows, with: Ticket Index DeSeason as the target, and Inputs such as the four signal indices, category, month, remount status, and more. When a new title has little or no history, the model estimates what its ticket index should be, based on shows with similar profiles. In some situations, the app can also use nearest-neighbour logic - essentially finding past titles that look similar in terms of signals and characteristics, and borrowing their performance as a guide. Every step is governed by safety checks: Predictions are clipped so they can't explode to unrealistic highs or drop to absurd lows. There's a post-COVID factor that dampens overall demand to reflect lingering uncertainty and slower recovery. And the app is careful to avoid leaky features that would accidentally let it use future information to predict the past. The goal is not perfection. The goal is a grounded, explainable guess - one that can stand up to scrutiny in a boardroom conversation."
        ),
        (
            6, "Adjustment Layers",
            "Layered graphic: Base Ticket Index at the bottom, then stacked layers labeled Seasonality, Remount, Economic Sentiment, Engagement, City Split.",
            "Once the app has a base ticket index - either from history or from the model - it applies a series of adjustment layers to make the estimate more realistic. First: Seasonality. Different months behave very differently for ballet. December isn't October. The app uses category-by-month seasonality factors: it looks at how each category performs in each month compared to its baseline. These factors are shrunk and clipped so that months with very little data don't distort everything. Second: Remount decay. If a title was performed six years ago, audience demand for a remount likely won't be identical. The app applies a remount decay factor based on the number of years since the last run, gently reducing expected demand. Third: City split. The app needs to decide how much of that demand belongs in Calgary and how much in Edmonton. It follows a hierarchy: If it has title-specific splits from history, it uses those. If not, it falls back to category-level averages. If even that is missing, it uses a default split defined in configuration. The final splits are clipped so neither city gets an implausibly extreme share. Fourth: Economic sentiment. Indicators from the Bank of Canada and Alberta's economy are combined into a single economic sentiment factor. These indicators are standardized, weighted, and then damped so that they nudge ticket estimates up or down within a reasonable range, rather than swinging them wildly. Fifth: Live engagement. Recent live analytics, such as engagement levels by category, are turned into an engagement factor. This factor tells the model, Right now, audiences seem more or less responsive to works like this. It is also clipped and damped to prevent recent blips from over-correcting long-term patterns. All of these layers combine into an Effective Ticket Index and eventually into: Estimated Tickets - the core demand estimate, and Estimated Tickets Final - the estimate after all safety and adjustment rules. Finally, the app applies marketing spend assumptions: default or title-specific marketing spend per ticket, multiplied by the estimated tickets in each city, to propose a total marketing budget."
        ),
        (
            7, "What the App Can and Can't Do",
            "Two columns: Can do vs Cannot do, with icons.",
            "So what, exactly, can this tool do for Alberta Ballet - and what can't it do? Here's what it can do: Provide single-ticket sales estimates for future titles in Calgary and Edmonton. Compare a new title to historic hits and flops. Show how online signals, history, economics, and engagement combine to support the estimate. Suggest marketing spend levels based on a cost-per-ticket assumption. Let planners explore what if scenarios by changing dates, categories, benchmarks, and other assumptions. And here's what it cannot do: It does not model subscriptions, school shows, or donor-driven tickets. It does not currently forecast revenue; it forecasts ticket volume. It is limited to Calgary and Edmonton. It isn't designed for multi-year forecasts or for dynamic pricing strategies. It doesn't automatically understand unusual, one-off events that fall far outside your historical categories. In short, it's a powerful lens on single-ticket demand - but it still needs human judgment, context, and artistic vision around it."
        ),
        (
            8, "A Walkthrough Example: Winter's Grace",
            "Screen mock-up of the app with fields filled for a fictional title Winter's Grace.",
            "Let's walk through a simplified example of how a planner might use the app for a hypothetical title: Winter's Grace, a new contemporary ballet planned for March 2026. The planner launches the app and opens the ticket estimator screen. They enter the title Winter's Grace and select the category Original Contemporary. They choose performance dates in March, and indicate that this is a new work with no direct historical run. Behind the scenes, the app: Collects Wikipedia, Trends, YouTube, and Spotify signals for the title and its composer or subject. Converts these into Familiarity, Motivation, and a Signal Only score. Because there's no direct history for Winter's Grace, the app uses its machine-learning model to estimate what the ticket index should be, based on comparable works. It applies March seasonality for contemporary works, adjusts for no remount decay - this is a first run, calculates the Calgary Edmonton split from category patterns, and layers in the current economic sentiment and engagement factor for contemporary titles. The planner then sees something like: An estimated single-ticket volume for Calgary and Edmonton, A breakdown of the key drivers: signals, history model, seasonality, remount status, economics, engagement, And a proposed marketing budget for each city, based on the configured cost-per-ticket. They can tweak assumptions - for example, shifting dates to April, or changing the category - and instantly see how the estimates respond. The app doesn't tell them whether to program Winter's Grace. But it does give them a transparent, data-backed forecast to inform that decision."
        ),
        (
            9, "Limitations and Conclusion",
            "Balanced scales: Data on one side, Judgment on the other. Then icons for future features.",
            "Like any forecasting tool, the Alberta Ballet Ticket Estimator has limitations. It relies heavily on benchmarks and categories; if a new work doesn't resemble anything in the archive, the model has to stretch. It is sensitive to how you classify a show - calling something a family classic versus a special event matters. And while the app makes a serious effort to incorporate economic sentiment and live engagement, those are still proxies. They can move quickly, especially in times of disruption. That's why this tool is designed as a partner to human judgment, not a replacement. It gives leaders a clear, defensible starting point - a way to say, Here's what the data suggests, and here's where our artistic and strategic judgment needs to weigh in. Looking ahead, the architecture leaves room for: Revenue forecasting, not just tickets. More detailed audience segment behaviour. Rolling out to more cities or touring markets. And eventually, integration with real-time analytics and dynamic pricing. But even in its current form, the Ticket Estimator is a major step forward: it brings together signals, history, economics, and engagement into a single, coherent view of future demand. In a world where every seat - and every dollar - matters, that kind of clarity is invaluable."
        )
    ]

    clips = []
    temp_files = []

    print("--- Starting Video Generation ---")
    
    for i, (scene_num, title, visual, narration) in enumerate(script_data):
        temp_audio = f"temp_scene_{scene_num}.wav"
        
        try:
            clip, audio_file = create_scene_clip(scene_num, title, visual, narration, temp_audio)
            clips.append(clip)
            temp_files.append(audio_file)
        except Exception as e:
            print(f"Error processing scene {scene_num}: {e}")
            import traceback
            traceback.print_exc()

    if clips:
        print("Concatenating clips...")
        final_video = concatenate_videoclips(clips, method="compose")
        
        output_filename = "AlbertaBallet_Estimator_Video.mp4"
        print(f"Writing final video to {output_filename}...")
        final_video.write_videofile(output_filename, fps=24)
        print("Done!")
        
        # Cleanup
        print("Cleaning up temporary audio files...")
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)
    else:
        print("No clips generated.")

if __name__ == "__main__":
    main()
