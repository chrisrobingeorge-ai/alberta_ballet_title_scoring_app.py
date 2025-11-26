import streamlit as st
from config.registry import load_feature_inventory
from data.loader import load_history_sales

st.set_page_config(page_title="Data Quality", layout="wide")
st.title("Data Quality – Registry vs Dataset")

st.markdown("""
### What is this page?
The **Data Quality** page validates alignment between:
- 📋 **Feature Registry** (what features are documented)
- 📊 **Dataset** (what columns actually exist in your data)

This helps you identify:
- ⚠️ **Undocumented columns**: Data in your files that needs to be added to the registry
- 🔍 **Missing features**: Documented features that haven't been collected yet

This ensures data governance and helps prioritize data collection efforts.
""")

df_reg = load_feature_inventory()
known = set(df_reg["Feature Name"].dropna().tolist())
df_raw = load_history_sales()

cols = set(df_raw.columns)
undefined_in_registry = sorted(cols - known)
unused_in_dataset = sorted(known - cols)

col1, col2 = st.columns(2)

with col1:
    st.subheader("⚠️ Columns in Dataset but NOT in Registry")
    st.caption("These columns exist in your data but aren't documented yet")
    if undefined_in_registry:
        st.warning(f"Found {len(undefined_in_registry)} undocumented columns")
        st.code("\n".join(undefined_in_registry))
    else:
        st.success("All dataset columns are documented ✓")

with col2:
    st.subheader("🔍 Features in Registry but NOT in Dataset")
    st.caption("These features are planned but not yet collected")
    if unused_in_dataset:
        st.info(f"Found {len(unused_in_dataset)} features to collect")
        # Show just first 30 to avoid overwhelming
        st.code("\n".join(unused_in_dataset[:30]))
        if len(unused_in_dataset) > 30:
            st.caption(f"...and {len(unused_in_dataset) - 30} more")
    else:
        st.success("All registry features are available ✓")
