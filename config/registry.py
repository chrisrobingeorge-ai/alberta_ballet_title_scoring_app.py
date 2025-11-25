Data Source Name,Description,Owner,Access Method,Update Frequency,Sample File / Table
Historical Show-Level Sales CSV,Aggregated show-level single and subscription tickets by city for Alberta Ballet productions.,Finance / Analytics,"CSV on shared drive / data lake (e.g., history_city_sales.csv).",Ad hoc / after each season load.,history_city_sales.csv
Ticketing Transaction Database,"Performance- and ticket-level data: ticket type, price, channel, timestamps, promo codes, comps.",Ticketing / IT,Direct DB connection or API; secured credentials.,Daily / near-real-time.,"fact_tickets, dim_price_type, dim_performance."
Ticketing Event & Production Metadata,"Setup tables for productions, seasons, venues, and capacities.",Ticketing / Artistic Admin,DB/API or scheduled CSV exports.,Updated as shows are created/edited.,"dim_production, dim_season, dim_venue."
CRM / Customer Database,"Patron profiles, purchase history, segmentation, email permissions.",Marketing / Development,CRM UI + API or scheduled exports.,Daily/Weekly.,"customers, households, interactions."
Email Marketing Platform,"Email campaigns, sends, opens, clicks, unsubscribes.",Marketing,"Platform APIs (e.g., Mailchimp/WordFly) or CSV exports.",Per campaign / weekly.,"email_campaigns, email_stats."
Digital Ads Platforms,"Paid media performance: spend, impressions, clicks, conversions per campaign.",Marketing / Digital Agency,Meta/Google Ads APIs or agency reports.,Daily/Weekly.,ad_campaign_performance.
Marketing Budget Spreadsheets,Planned and actual spend by production and city.,Marketing / Finance,Excel/Google Sheets in shared drive.,Monthly / seasonal planning.,Marketing_Budget_YYYY-YY.xlsx.
Production Planning Sheets/DB,"Artistic information: premiere vs revival, run_type, guest stars, collaborators.",Artistic Director / Artistic Admin,Spreadsheets or internal database.,Seasonal.,SeasonYYYY_Productions.xlsx.
Venue Capacity Reference,Seat maps and capacity by price zone and configuration.,Production / Ticketing,Static CSV or database table; occasionally updated.,Rarely.,venue_capacity.csv.
External Economic Data,"Alberta unemployment, CPI, GDP and related macro indicators.",Analytics,Download from Statistics Canada / Alberta Govt portals.,Monthly/Quarterly.,alberta_macro.csv.
External Weather Data,"Daily temperature, precipitation, extreme weather flags for Calgary and Edmonton.",Analytics,Environment Canada API or CSV downloads.,Daily / historical backfill.,weather_daily_city.csv.
External Events & Sports Calendars,"Major events (Stampede, concerts, NHL games) that may compete with performances.",Marketing / Analytics,Manual spreadsheets or scraped event listings.,Seasonal.,city_events_calendar.csv.
Tourism & Population Data,Population estimates and visitor volumes for Calgary and Edmonton.,Analytics,StatsCan and Travel Alberta downloads.,Annual/Seasonal.,city_population_tourism.csv.
Google Trends Export,Search interest indices for production titles and ballet-related terms.,Analytics,Google Trends UI/API wrapper; CSV exports.,Per campaign / ad hoc.,google_trends_titles.csv.

Theme,Feature Name,Description,Data Type,Status,Notes / Source System
Production Attributes,show_title,"Marketing / ticketing title of the production (e.g., Nutcracker, Swan Lake, Alice in Wonderland).",categorical (string),available,From history_city_sales.csv [Show Title]; canonicalise against ticketing system.
Production Attributes,show_title_id,Internal unique ID for each production title (across seasons).,categorical (ID),to be collected,"Create title dimension, map to ticketing 'production' or 'event' IDs."
Production Attributes,production_season,"Artistic season label for the production (e.g., 2024–25).",categorical (string),to be collected,From season planning sheets / ticketing season codes.
Production Attributes,city,"City of performance (Calgary, Edmonton).",categorical,available,Implied by separate columns in history_city_sales.csv; normalise into a City dimension.
Production Attributes,venue_name,Name of venue where production is performed.,categorical,to be collected,"From ticketing performance setup (e.g., Southern Alberta Jubilee Auditorium)."
Production Attributes,venue_capacity,Total sellable capacity for venue and configuration (per performance).,numeric (int),to be collected,From venue/production setup in ticketing; needed for % capacity features.
Production Attributes,performance_count_city,Number of performances of this title in a given city/run.,numeric (int),to be collected,Aggregate from performance-level table in ticketing system.
Production Attributes,performance_count_total,Total number of performances across both cities for the run.,numeric (int),to be collected,Sum performance_count_city across cities.
Production Attributes,run_type,"Production type category (e.g., full-length classical, mixed bill, contemporary narrative, family).",categorical,in development,"Define taxonomy with Artistic & Marketing, tag all historical titles."
Production Attributes,musical_forces,"Live orchestra, recorded track, guest band, etc.",categorical,in development,From production notes; influences perceived value and pricing.
Production Attributes,score_familiarity_index,Index for how familiar the music is to a general Alberta audience.,numeric (0–1 or 1–5),in development,"Combine expert scoring with external consumption metrics (Spotify, YouTube)."
Production Attributes,story_familiarity_index,"Index of narrative familiarity (e.g., Nutcracker high, new creation lower).",numeric,in development,Expert ratings + Google Trends search interest.
Production Attributes,production_branding_cluster,"Cluster label grouping titles with similar brand positioning (holiday, romantic classic, edgy contemporary, etc.).",categorical,in development,From prior clustering work and marketing segmentation.
Production Attributes,family_friendliness_flag,Whether production is marketed as suitable for children/families.,boolean,in development,"From marketing copy and series labelling (e.g., family series)."
Production Attributes,run_length_days,Number of days between first and last performance in a city.,numeric (int),to be collected,Derived from performance schedule.
Production Attributes,pricing_tier_structure,"Representation of price zones and spread (e.g., number of tiers, premium differential).",categorical / numeric vector,to be collected,From ticketing price tables.
Production Attributes,average_base_ticket_price,Average list price for standard adult single tickets (before fees/discounts).,numeric (float),to be collected,From ticketing price tables; summarise per city and title.
Production Attributes,discounting_intensity_index,Share of tickets sold on promo/discount vs full price.,numeric (0–1),in development,Requires promo flag at ticket level; aggregate per title/city.
Production Attributes,subscription_eligibility_flag,Whether production is included in subscription packages.,boolean,to be collected,From season/subscription design tables.
Production Attributes,guest_star_flag,"Whether production features a named guest star (dancer, choreographer, band).",boolean,in development,From artistic contracts and marketing materials.
Production Attributes,world_premiere_flag,Whether this is a world premiere / new creation.,boolean,in development,From artistic archives.
Production Attributes,co_production_flag,Whether production is a co-production with another company.,boolean,in development,From contracts / artistic notes.
Historical Sales Trends,single_tickets_calgary,Total single tickets sold for a title in Calgary for a given run.,numeric (int),available,history_city_sales.csv [Single Tickets - Calgary].
Historical Sales Trends,single_tickets_edmonton,Total single tickets sold for a title in Edmonton for a given run.,numeric (int),available,history_city_sales.csv [Single Tickets - Edmonton].
Historical Sales Trends,subscription_tickets_calgary,Total subscription tickets allocated to the title in Calgary.,numeric (int),available,history_city_sales.csv [Subscription Tickets - Calgary].
Historical Sales Trends,subscription_tickets_edmonton,Total subscription tickets allocated to the title in Edmonton.,numeric (int),available,history_city_sales.csv [Subscription Tickets - Edmonton].
Historical Sales Trends,total_single_tickets,Total single tickets sold across both cities.,numeric (int),available/derived,history_city_sales.csv [Total Single Tickets] or compute from city columns.
Historical Sales Trends,total_subscription_tickets,Total subscription tickets across both cities.,numeric (int),derived,Compute from Calgary/Edmonton subscription columns.
Historical Sales Trends,total_tickets_all,Total tickets (singles + subscriptions) across both cities.,numeric (int),derived,Key demand target candidate; combine single and subscription totals.
Historical Sales Trends,avg_tickets_per_performance,Average tickets sold per performance in a city.,numeric (float),in development,Requires performance_count_city; tickets / performance_count_city.
Historical Sales Trends,load_factor,Percentage of capacity sold (tickets / theoretical capacity).,numeric (0–1),in development,Needs capacity and performance counts; useful for normalisation.
Historical Sales Trends,prior_season_title_sales,Lagged total tickets for the same title in prior seasons.,numeric (int),in development,Join by show_title_id and season; used for recurring titles like Nutcracker.
Historical Sales Trends,title_sales_trend,Trend in sales for recurring titles across multiple seasons.,numeric (float),in development,Compute slope via regression over historical seasons.
Historical Sales Trends,first_run_vs_revival_flag,Flag for first presentation vs revival of a title.,boolean,in development,From artistic archives; interacts with prior sales features.
Historical Sales Trends,weeks_to_80pct_sold,Weeks before opening when 80% of capacity was sold.,numeric (float),to be collected,Requires dated transaction and on-sale data.
Historical Sales Trends,pre_on_sale_demand_proxy,Tickets sold in first 48–72 hours after on-sale as % of run capacity.,numeric (float),to be collected,From transaction timestamps; a leading indicator of demand.
Historical Sales Trends,late_sales_share,Share of tickets sold in final 7 days before opening.,numeric (0–1),to be collected,From transactions; useful for behavioural segmentation.
Historical Sales Trends,channel_mix_distribution,"Distribution of sales by channel (web, phone, box office, group).",numeric vector,to be collected,From ticketing channel codes; summarise as percentages.
Historical Sales Trends,group_sales_share,Share of tickets sold via group bookings.,numeric (0–1),to be collected,From group booking flag in ticketing.
Historical Sales Trends,comp_ticket_share,Share of complimentary tickets vs paid.,numeric (0–1),to be collected,From comp flag; important for interpreting raw ticket counts.
Historical Sales Trends,refund_cancellation_rate,Percentage of tickets refunded or cancelled for the run.,numeric (float),to be collected,From refund transactions; may correlate with weather or health events.
Historical Sales Trends,subscription_mix_ratio,Subscription tickets / total tickets.,numeric (0–1),derived,Compute from subscription and total tickets.
Marketing & Promotions,marketing_campaign_id,Unique ID for the main marketing campaign associated with the production.,categorical,to be collected,From CRM / email or marketing automation tools.
Marketing & Promotions,marketing_budget_city,Paid marketing spend allocated to the production per city.,numeric (float),to be collected,From marketing budget spreadsheets / finance.
Marketing & Promotions,digital_ad_spend,"Total paid digital ad spend (Meta, Google, programmatic) for the run.",numeric (float),to be collected,From ad platform exports or agency reports.
Marketing & Promotions,email_campaign_count,Number of distinct email campaigns sent promoting the production.,numeric (int),to be collected,From email marketing platform logs.
Marketing & Promotions,email_impressions,Total email sends for campaigns linked to the production.,numeric (int),to be collected,From email platform stats.
Marketing & Promotions,email_open_rate,Average open rate across relevant email campaigns.,numeric (0–1),to be collected,From email performance reports.
Marketing & Promotions,email_click_rate,Average click-through rate across relevant email campaigns.,numeric (0–1),to be collected,From email performance reports.
Marketing & Promotions,social_posts_count,Number of social posts tagged to the production.,numeric (int),to be collected,From social scheduling/analytics tools.
Marketing & Promotions,social_engagement_rate,Engagement (likes/comments/shares) per impression across production posts.,numeric (float),to be collected,From social media analytics.
Marketing & Promotions,video_views_production_trailer,Total views of trailers / promo videos.,numeric (int),to be collected,"From YouTube, Instagram, etc."
Marketing & Promotions,out_of_home_flag,"Whether out-of-home media (billboards, transit ads) were used.",boolean,to be collected,From media plans / invoices.
Marketing & Promotions,media_partnership_flag,"Presence of media partners (radio/TV promotions, contests).",boolean,to be collected,From marketing partnership records.
Marketing & Promotions,pricing_promotion_flag,"Any price-based promotion (early-bird, dynamic discounts, rush tickets).",boolean,to be collected,From promo setup in ticketing and campaign briefs.
Marketing & Promotions,special_offer_volume,Number of tickets sold under promo codes / special offers.,numeric (int),to be collected,From ticketing promo code reports.
Marketing & Promotions,campaign_start_lead_time_days,Days between campaign launch date and first performance.,numeric (int),in development,Requires campaign start dates from marketing calendar.
Marketing & Promotions,email_to_sale_conversion_rate,Percentage of email recipients who purchase after campaign exposure.,numeric (float),in development,Link CRM exposure to sales; may require matchback process.
Marketing & Promotions,remarketing_flag,Use of retargeting/dynamic remarketing for the production.,boolean,to be collected,From digital agency / ads manager.
Timing & Schedule Factors,opening_date,Date of first performance in the run (city-level).,date,to be collected,From performance schedule in ticketing.
Timing & Schedule Factors,closing_date,Date of final performance in the run (city-level).,date,to be collected,From performance schedule in ticketing.
Timing & Schedule Factors,performance_day_of_week_distribution,Share of performances by day of week (Thu/Fri/Sat/Sun).,numeric vector,in development,Derived from performance_date for all performances in run.
Timing & Schedule Factors,holiday_period_flag,"Run overlaps with major holidays (Christmas, Easter, Thanksgiving, Family Day, etc.).",boolean,in development,Compare run window to Canadian/Alberta holiday calendar.
Timing & Schedule Factors,school_break_overlap_flag,Run overlaps with school breaks (winter/spring) for Calgary/Edmonton school boards.,boolean,in development,Requires school calendar data.
Timing & Schedule Factors,competing_major_event_flag,"Major events in city (Stampede, major concerts, playoff games) overlap run.",boolean,in development,From city events calendars and NHL schedule.
Timing & Schedule Factors,month_of_opening,Month of opening (1–12).,numeric (int),in development,Derived from opening_date.
Timing & Schedule Factors,season_quarter,Fiscal or calendar quarter of the run.,categorical,in development,Derived from opening_date and internal fiscal calendar.
Timing & Schedule Factors,days_on_sale_before_opening,Days between on-sale date and opening night.,numeric (int),to be collected,Requires on-sale date from ticketing.
Timing & Schedule Factors,on_sale_month,Month in which tickets first went on sale.,categorical,in development,Derived from on-sale date.
Timing & Schedule Factors,typical_performance_time,Share of performances categorised as evening vs matinee.,numeric (float),in development,From performance start times.
Timing & Schedule Factors,weekday_vs_weekend_mix,Percent of performances on Fri–Sun vs Mon–Thu.,numeric (float),in development,Derived from schedule.
Timing & Schedule Factors,overlapping_internal_events_flag,Other Alberta Ballet productions overlap in same city/run window.,boolean,in development,Compare run windows across all productions.
External Factors,alberta_unemployment_rate,Unemployment rate in Alberta during run period.,numeric (float),to be collected,From Statistics Canada Labour Force Survey; use monthly or quarterly value aligned to run.
External Factors,alberta_cpi_index,Consumer Price Index for Alberta (inflation proxy) at time of run.,numeric (float),to be collected,From StatsCan CPI for Alberta; align to month of opening.
External Factors,alberta_real_gdp_growth_rate,Real GDP growth rate for Alberta in the season year.,numeric (float),to be collected,From Alberta Treasury Board and Finance or StatsCan.
External Factors,wti_oil_price_avg,Average WTI oil price during the run or season.,numeric (float),to be collected,From Bank of Canada or EIA historical oil price series.
External Factors,weather_severity_index_city,"Index summarising extreme weather during run dates in each city (e.g., cold snaps, major snow).",numeric (float),in development,From Environment Canada daily weather data.
External Factors,covid_restriction_level,Categorical index of COVID-19 restrictions in place during run.,categorical,in development,Derive from Alberta public health order timelines.
External Factors,exchange_rate_cad_usd,Average CAD–USD exchange rate during season.,numeric (float),to be collected,From Bank of Canada FX rates.
External Factors,population_city,Population of Calgary/Edmonton in given year.,numeric (int),to be collected,From StatsCan or municipal estimates.
External Factors,median_household_income_city,Median household income in Calgary/Edmonton.,numeric (float),to be collected,From StatsCan Census / income tables.
External Factors,tourism_visitation_index,"Proxy index for visitor volume (hotel occupancy, airport arrivals).",numeric (float),to be collected,From Travel Alberta or city tourism data.
External Factors,google_trends_title_interest,Normalised Google Trends index for production title/theme during campaign window.,numeric (float),in development,From Google Trends for terms like 'Nutcracker ballet Calgary'.
External Factors,arts_sector_confidence_index,Indicator of arts sector health (attendance/financial trend proxy).,numeric (float),in development,Would require data from peer organisations or sector surveys.
Join Key,Role / Purpose,Connected Data Sources,Format & Constraints,Notes
show_title_id,Primary key for unique production title across seasons.,"Ticketing (productions), history_city_sales, production metadata, marketing campaigns.",string/ID; must be unique and stable.,Create mapping from ticketing 'production' table and backfill for historical CSV.
show_title,Human-readable label for production; secondary join key.,"history_city_sales, marketing assets, website copy.","string; may vary slightly by year (e.g., 'The Nutcracker 2024').",Use mainly for display; map to show_title_id before modelling.
season,"Season identifier grouping productions by year (e.g., 2024–25).","Production plan, ticketing, financial reporting, marketing budgets.",string 'YYYY–YY'.,Standardise via dim_season table; attach to each production.
city,City for performance and sales (Calgary/Edmonton).,"history_city_sales (implicitly), ticketing performances, marketing spend by city.","enum {Calgary, Edmonton}.",Normalise city labels; avoid free text.
venue_id,Unique venue identifier for Jubilee and any other locations.,"Ticketing venue table, capacity reference, performance schedule.",string/ID.,"Link to dim_venue with capacity, city, and configuration."
performance_id,Unique identifier per performance instance.,"Ticketing transactions, performance schedule, seat inventory.",string/ID.,Needed if moving from show-level to performance-level modelling.
performance_date,Calendar date of each performance.,"Performance schedule, weather data, city events calendars.",date YYYY-MM-DD.,Key for joining to daily weather and external events.
on_sale_date,Date tickets first go on sale for the run.,"Ticketing system, marketing calendar.",date.,Used to calculate days_on_sale and align marketing start.
marketing_campaign_id,Key linking production to marketing campaigns and metrics.,"Email platform, ad platforms, CRM.",string.,Maintain campaign metadata table mapping to show_title_id/season/city.
fiscal_period,Accounting period key for financial and macro joins.,"Finance actuals, budgets, macroeconomic data.","string (e.g., '2025-04').",Derive from performance_date or season; align with internal fiscal calendar.
external_geo_key,Geo key for external data (province/city).,"Economic data, population, tourism, weather.","string (e.g., 'Alberta', 'Calgary').",Standardise labels to match StatsCan and other official sources.

Feature Name,Leakage Risk (Y/N),Allowed at Forecast Time (Y/N),Mitigation Notes
show_title_id,N,Y,Known at planning time; safe as a structural ID.
run_type,N,Y,Defined in artistic planning; does not depend on realised sales.
city,N,Y,Known in advance; safe.
venue_capacity,N,Y,Capacity known once venue is booked; safe.
performance_count_city,N,Y,Schedule defined pre-forecast; safe.
single_tickets_calgary (historical),N,Y,Use only for past seasons as lagged features; never from the forecast season.
prior_season_title_sales,N,Y,Construct using only seasons prior to the forecast season.
weeks_to_80pct_sold,Y,N,Requires future sales trajectory; exclude from initial forecasting models.
late_sales_share,Y,N,Outcome-like feature; only use for ex-post analysis or in-run models.
comp_ticket_share (current run),Y,N,Based on post hoc allocations; avoid for pre-run forecasts.
refund_cancellation_rate (current run),Y,N,Depends on realised behaviour; leakage if used in training as a feature.
marketing_budget_city (planned),N,Y,"If budget is fixed pre-forecast, use planned amount as scenario input."
digital_ad_spend (actual),Y,Only if using planned estimates,Do not feed actual spend that is decided after forecast time into training features.
email_open_rate (current run),Y,N,Only known post-campaign; safe only for back-testing analyses.
email_open_rate (historical averages),N,Y,Historical aggregate stats can be used as input features.
pre_on_sale_demand_proxy,Y,N for initial model,Can be used for in-run forecasting but not for pre-on-sale forecasts.
external_economic_indicators,N,Y,Use values available at or before forecast date; avoid future macro revisions.
weather_severity_index_city (current run),Y,N,Future weather unknown; use only in scenario testing or back-tests.
covid_restriction_level,Depends,Y if restrictions announced,Use only restrictions known or announced at the time of forecasting.
google_trends_title_interest (near-term),Depends,Y if measured pre-forecast,"If forecasting very early, exclude; if forecasting close to on-sale, may include recent trends."

Task ID,Task Description,Tools / Methods,Owner,Status,Priority
M1,"Finalise feature list and taxonomies (run_type, family, etc.) with Artistic & Marketing.",Workshops; Google Sheets; tagging in dim_production.,Analytics + Artistic + Marketing,in progress,high
M2,"Design unified data model (dims/facts) for ticketing, productions, venues, seasons.",SQL; dbt or equivalent; warehouse schema design.,Analytics / IT,not started,high
M3,Ingest and clean historical performance-level ticketing data.,SQL; Python (pandas); ETL tools.,Analytics,in progress,high
M4,Connect CRM and marketing platforms for campaign metrics.,Platform APIs; CSV imports; Python integration scripts.,Marketing + Analytics,not started,medium
M5,"Assemble external macro, weather, and events datasets for Calgary/Edmonton.",Python scripts; API downloads; manual curation where needed.,Analytics,not started,medium
M6,Implement show-level feature engineering pipeline.,Python; dbt; possible feature store.,Analytics,not started,high
M7,"Define target variables and evaluation metrics (MAE, RMSE, R²) and document them.",Analytical design; internal documentation.,Analytics + Executive,in progress,high
M8,"Develop baseline models for demand forecasting (GLM, random forest, gradient boosting, PyCaret).","Python (scikit-learn, XGBoost, PyCaret).",Analytics,not started,medium
M9,Implement back-testing framework across past seasons.,Time-series-aware CV; MLflow or similar tracking.,Analytics,not started,high
M10,"Perform model interpretability analysis (feature importance, SHAP).",SHAP; permutation importance; partial dependence plots.,Analytics,not started,medium
M11,Build or extend internal forecasting UI (Streamlit app) for title scoring and scenario analysis.,Python; Streamlit; existing Alberta Ballet apps.,Analytics,in progress,high
M12,"Create and maintain project documentation (data dictionary, feature glossary, leakage controls).",Confluence or similar wiki; version-controlled docs.,Analytics,not started,medium
M13,Set governance and review cadence for model results with leadership.,Standing meetings; dashboard reviews.,CEO / Analytics,not started,medium
M14,Establish monitoring for forecast vs actual performance and define retraining triggers.,Dashboards; automated KPI checks; retrain schedule.,Analytics,not started,medium

Pipeline Type,Pipeline Name,Description,Inputs,Outputs,Frequency,Owner,Status
Data,Sales Data Ingestion,Ingest and clean show- and performance-level ticketing data into analytics warehouse.,"Ticketing DB, history_city_sales.csv","fact_sales_show, fact_sales_performance",Weekly / after runs,Analytics,in development
Data,Production Metadata Ingestion,"Load production, season, and venue metadata into dimension tables.","Ticketing event setup, production planning sheets","dim_production, dim_season, dim_venue",Seasonal / on change,Analytics / Artistic Admin,in development
Data,Marketing Data Ingestion,"Pull email, digital ads, and budget data into analytics environment.","Email platform, ad platforms, marketing budget spreadsheets","fact_marketing_campaign, dim_campaign",Weekly,Marketing / Analytics,to be designed
Data,External Data Ingestion,"Load macroeconomic, weather, and events data for Calgary and Edmonton.","StatsCan, Alberta Govt, Environment Canada, event calendars","dim_macro, fact_weather, fact_city_event",Monthly/Seasonal,Analytics,to be designed
Data,Feature Engineering – Show-Level,"Build show-level feature set combining production, sales, marketing, schedule, and external features.",facts and dims above,ml_features_show_level,Ad hoc / per retrain,Analytics,to be designed
Data,Target Construction,"Define and compute target variables (e.g., total tickets, load factor) for each production/city.","fact_sales_show, dim_production, dim_season",ml_targets_show_level,Ad hoc / per retrain,Analytics,in development
Data,Train/Test Dataset Build,"Assemble final modelling dataset with features, targets, and time-aware splits.","ml_features_show_level, ml_targets_show_level",dataset_modelling_show_level,Ad hoc / per retrain,Analytics,to be designed
ML,Model Training – Title Demand,Train models predicting show-level ticket demand by city for planned productions.,dataset_modelling_show_level,"model artefacts (e.g., sklearn/XGBoost/PyCaret pipelines)",Quarterly / pre-season,Analytics,in development
ML,Model Validation & Monitoring,"Evaluate model accuracy (MAE, RMSE, R²) and monitor performance over time.","Predictions vs actuals, history_city_sales.csv","Validation reports, monitoring dashboards",After each major run,Analytics,to be designed
ML,Forecast Scoring – Season Planning,Score proposed titles early in season design to inform repertoire decisions.,Planned productions metadata + external baseline data,Demand forecasts per title/city,Seasonal,Artistic Director & Analytics,to be designed
ML,Forecast Scoring – On-Sale,Score confirmed productions before marketing launch using final schedule and capacity.,"Confirmed production metadata, schedule, macro data",Updated demand and revenue forecasts,Per on-sale,Analytics / Marketing,to be designed
ML,Forecast Refresh – In-Run (Optional),Refresh forecasts mid-campaign based on early sales and marketing performance.,"Transaction-level sales, campaign metrics to date",Revised run forecasts and risk flags,Weekly during key runs,Analytics,future / optional
