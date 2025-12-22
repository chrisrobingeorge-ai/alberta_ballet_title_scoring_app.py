# Alberta Ballet Title Scoring App - Video Script (5 minutes)

**Target Length:** ~850 words | **Speaking Pace:** ~170 words/minute | **Duration:** ~5 minutes

---

## [Opening - 0:00-0:30]

**[Scene: App interface with logo]**

What if you could predict audience demand for a ballet performance before you even book the theater?

The Alberta Ballet Title Scoring App is a machine-learning-powered decision support system that helps artistic directors, programmers, and marketing teams make evidence-based choices about which ballets to stage, when to schedule them, and where audiences are most likely to respond.

Let's explore how this system transforms uncertainty into actionable insights.

---

## [The Challenge - 0:30-1:15]

**[Scene: Calendar with question marks, empty theaters]**

Ballet companies face a fundamental challenge: programming seasons up to 18 months in advance with limited data about future audience demand. Will a classic like Swan Lake sell better in December or March? Should we remount Cinderella or premiere a contemporary work? Will Calgary audiences respond differently than Edmonton?

Traditionally, these decisions relied on intuition, anecdotal experience, and past box office numbers. But historical ticket sales alone don't capture the full picture. Public interest shifts. Digital platforms like YouTube and Wikipedia reveal what audiences are searching for right now. Seasonal patterns matter. The calendar matters.

The Title Scoring App brings all these factors together into a single, data-driven forecast.

---

## [How It Works: Digital Signals - 1:15-2:15]

**[Scene: Wikipedia logo, Google Trends graph, YouTube icon, Chartmetric visualization]**

At the heart of the system are four digital signals that measure public interest in real time:

**Wikipedia page views** show baseline familiarity. How many people are researching this ballet? Is it a household name or a hidden gem?

**Google Trends data** captures search volume. Are audiences actively looking for performances of this title?

**YouTube engagement** measures motivation. Are people watching videos, sharing clips, and engaging with content related to this ballet?

**Chartmetric streaming data** tracks artist popularity across platforms like Spotify, Apple Music, and TikTok. For ballets with recognizable composers or choreographers, this reveals cultural momentum.

The app combines these four signals into a composite "online visibility score" — a single metric that captures both awareness and active interest. A title might be well-known but lack current excitement, or it might be trending online despite being unfamiliar to traditional audiences. The system detects both patterns.

---

## [Machine Learning Engine - 2:15-3:00]

**[Scene: Graph showing historical data points, regression line]**

But online signals alone don't guarantee ticket sales. That's where machine learning comes in.

The app trains regression models dynamically using your own historical performance data. It learns patterns from past productions: which titles sold well in Calgary versus Edmonton, how seasonality affects different categories of ballet, and how online visibility translates into actual box office results for your specific audience.

The system uses Ridge regression with empirical constraints. It knows that a title with zero online presence still has a realistic floor for ticket sales, and it anchors predictions to benchmark titles you trust. As you add more historical data, the models become more accurate and personalized to Alberta Ballet's unique audience.

If a title has no local performance history, the model falls back to pattern matching based on similar productions with known outcomes.

---

## [Seasonality and Context - 3:00-3:30]

**[Scene: Calendar highlighting months, winter holiday imagery]**

Timing matters. A family-friendly ballet like The Nutcracker performs differently in December than it would in March. The app learns category-specific seasonality from historical data and applies those patterns to future scheduling decisions.

Month-by-month adjustments ensure predictions reflect real-world demand fluctuations. The system knows that February is shoulder season, that holiday programming gets a boost, and that certain categories thrive in specific windows.

---

## [Real Example: Giselle in September - 3:30-4:15]

**[Scene: Giselle performance imagery, app interface showing calculation]**

Let's see how this works with a real example: Giselle scheduled for September.

The app starts with digital signals. Giselle has strong Wikipedia traffic — it's a recognized classic with steady page views. Google Trends shows moderate but consistent search volume. YouTube engagement is high, with performance videos from major companies getting millions of views. Chartmetric picks up streaming activity from the ballet's iconic Adam score.

These signals combine into a Familiarity score of 110 and a Motivation score of 95. Both above the benchmark of 100, indicating this is a well-known title with active audience interest.

Next, the machine learning model kicks in. Historical data shows Giselle performed well in past Alberta Ballet seasons. The model learned that romantic story ballets resonate strongly with the Core Classical audience segment. It applies those learned patterns to this prediction.

Then seasonality adjustments. September is early fall — not peak holiday season, but also not summer shoulder period. The model applies a neutral seasonal factor of 1.0, meaning no adjustment up or down.

The final prediction: a Ticket Index of 115, translating to approximately 3,800 tickets in Calgary and 2,500 in Edmonton. The system forecasts strongest appeal to Core Classical enthusiasts, followed by General Population audiences.

Marketing now knows exactly where to focus: target classical ballet fans, emphasize the romantic tragedy, and allocate budget proportionally between cities.

---

## [City and Audience Segmentation - 4:15-4:30]

**[Scene: Map showing Calgary and Edmonton, audience demographic icons]**

As you saw with Giselle, every prediction is decomposed into actionable details:

**Calgary versus Edmonton splits** are learned from historical city-level performance. Some titles resonate more strongly in one market. The system captures those patterns automatically.

**Audience segment predictions** break forecasts into four groups: General Population, Core Classical enthusiasts, Family audiences, and Emerging Adults. Each segment gets a tailored estimate based on the title's characteristics and historical attendance patterns.

This granularity helps marketing teams allocate resources efficiently and target the right audiences with the right messaging.

---

## [Explainability and Trust - 4:30-4:50]

**[Scene: PDF report with narrative text, SHAP visualization]**

Transparency is critical. The app doesn't just give you a number — it explains why.

Using SHAP analysis, the system identifies which factors are driving each prediction. Is it strong Wikipedia traffic? Favorable seasonality? Historical remount performance? Category patterns? The narrative engine translates these technical factors into plain-language explanations that anyone on your team can understand.

Every forecast comes with a detailed report showing exactly how the model arrived at its recommendation. No black boxes. No mystery. Just clear, interpretable insights.

---

## [Closing - 4:50-5:00]

**[Scene: App dashboard with results, tickets being purchased]**

The Alberta Ballet Title Scoring App transforms season planning from guesswork into data-driven strategy.

It combines real-time digital signals, machine learning trained on your own history, seasonality intelligence, and transparent explanations into a single platform that answers the questions that matter: Which titles will resonate? When should we schedule them? Where will demand be strongest?

From artistic directors to marketing teams to board members, everyone gets the insights they need to make confident, evidence-based decisions about Alberta Ballet's future seasons.

**Smart programming starts with better data. Let the Title Scoring App be your guide.**

---

**[End Card: Alberta Ballet logo, App URL]**

---

**Total Word Count:** ~850 words  
**Estimated Duration:** 5 minutes at natural speaking pace
