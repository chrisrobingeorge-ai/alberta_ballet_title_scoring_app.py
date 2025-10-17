# streamlit_app.py
with colD:
gen_pdf = st.checkbox("Prepare PDF after scoring", value=False)
with colE:
st.caption("v8 uses **priors × score adapter** for forecasts; download CSV for full columns.")


priors_df = None
if uploaded is not None:
try:
raw_df = pd.read_excel(uploaded) if uploaded.name.endswith((".xlsx",".xls")) else pd.read_csv(uploaded)
hist_df = build_tickets_columns(raw_df)
priors_df_AB = build_priors(hist_df, cutoff_year, "AB")
priors_df_YYC = build_priors(hist_df, cutoff_year, "Calgary")
priors_df_YEG = build_priors(hist_df, cutoff_year, "Edmonton")
priors_df = pd.concat([priors_df_AB, priors_df_YYC, priors_df_YEG], ignore_index=True)
# Keep only selected market's priors for lookup
priors_df = priors_df[priors_df["market"]==market_key].copy()
st.success(f"Priors ready for {market}: {len(priors_df)} cohorts.")
st.dataframe(priors_df.sort_values(["bucket","gender_focus","n_obs"], ascending=[True,True,False]), use_container_width=True)
except Exception as e:
st.error(f"Failed to load or build priors: {e}")


if run_btn:
with st.spinner("Scoring titles…"):
df = score_titles(
titles,
yt_key.strip() or None,
sp_id.strip() or None,
sp_secret.strip() or None,
use_trends,
market_key,
segment_name,
attrs_df,
priors_df if (use_priors and priors_df is not None) else None,
use_adjusted=use_adjusted,
)


# Choose which columns to sort & visualize
sort_x, sort_y = ("FamiliarityAdj","MotivationAdj") if use_adjusted else ("Familiarity","Motivation")
df = df.sort_values(by=[sort_y, sort_x], ascending=False)


if do_benchmark and benchmark_title:
df = apply_benchmark(df, benchmark_title, use_adjusted)


st.success("Done.")
st.dataframe(df, use_container_width=True)


# Diagnostics — dispersion
st.subheader("Diagnostics")
diag_cols = [sort_x, sort_y, "GoogleTrends","WikipediaRawAvgDaily","WikipediaFamiliarity","YouTubeN","SpotifyN"]
if "ForecastTickets" in df.columns:
diag_cols += ["PriorMeanTickets","PriorCV","PriorN","ForecastTickets"]
present = [c for c in diag_cols if c in df.columns]
diag = pd.DataFrame({
"metric": present,
"min": [float(df[c].min()) for c in present],
"max": [float(df[c].max()) for c in present],
"std": [float(df[c].std(ddof=0)) for c in present],
})
st.dataframe(diag, use_container_width=True)


# Charts
st.subheader("Charts")
figQ = quadrant_plot(df, sort_x, sort_y, f"{sort_x} vs {sort_y} (Quadrant Map)")
st.pyplot(figQ)
if "ForecastTickets" in df.columns and df["ForecastTickets"].notna().any():
figF = bar_chart(df.fillna({"ForecastTickets":0}), "ForecastTickets", "Forecast Tickets (rule-based)")
st.pyplot(figF)


# CSV / PDF
st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode("utf-8"), "title_scores_v8.csv", "text/csv")
if gen_pdf:
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
pdf_path = f"title_scores_brief_{ts}.pdf"
generate_pdf_brief(df, use_adjusted, pdf_path)
st.success("PDF created.")
st.download_button("⬇️ Download PDF Brief", data=open(pdf_path, "rb").read(
