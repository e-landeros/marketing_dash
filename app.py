import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Page Configuration
st.set_page_config(page_title="Marketing Analytics Dashboard", layout="wide")

# ==========================================
# 1. DATA LOADING & FEATURE ENGINEERING
# ==========================================
@st.cache_data
def load_data():
    try:
        # Robust path handling
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, "../data/raw/synthetic_leads.csv")
        if not os.path.exists(data_path):
             data_path = "data/raw/synthetic_leads.csv"

        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # --- NEW: Feature Engineering (From Notebook) ---
        # 1. Profitability Metrics
        df['net_profit'] = df['payout_amount'] - df['cost_per_click']
        
        # 2. Age Binning
        df['Age_Group'] = pd.cut(df['age'], bins=[18, 25, 35, 45, 55, 65, 100], 
                                 labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop()


# ==========================================
# 2. GLOBAL FILTERS (Top of Page)
# ==========================================
st.title("Marketing Analytics Dashboard")

st.subheader("Global Filters")
col_f1, col_f2 = st.columns(2)

# Date Range
min_date = df['timestamp'].min().date()
max_date = df['timestamp'].max().date()
default_start = max_date - timedelta(days=30)
with col_f1:
    start_date, end_date = st.date_input("Date Range", [default_start, max_date])

# Traffic Source
sources = df['traffic_source'].unique().tolist()
with col_f2:
    selected_sources = st.multiselect("Traffic Source", sources, default=sources)

# Apply Filters
mask = (
    (df['timestamp'].dt.date >= start_date) & 
    (df['timestamp'].dt.date <= end_date) &
    (df['traffic_source'].isin(selected_sources))
)
filtered_df = df[mask]

if filtered_df.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

# ==========================================
# 3. TABS
# ==========================================
tab1, tab2, tab_campaign, tab_geo, tab3, tab4, tab_detailed, tab_ts, tab5, tab6, tab7 = st.tabs([
    "Overview", 
    "Traffic", 
    "Campaigns",
    "Geographic",
    "Funnel Analysis",
    "Segments & Targeting", 
    "Detailed Data",
    "Time Series Analysis",
    "Sus Traffic",
    "Linear Regression",
    "Clustering"
])

# --- TAB 1: EXECUTIVE SUMMARY ---
with tab1:
    st.header("Executive Summary")
    
    # 1. High-Level KPIs
    total_spend = filtered_df['cost_per_click'].sum()
    total_revenue = filtered_df['payout_amount'].sum()
    net_profit = filtered_df['net_profit'].sum()
    roas = total_revenue / total_spend if total_spend > 0 else 0
    conversion_rate = filtered_df['policy_sold'].mean()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Spend", f"${total_spend:,.0f}")
    col2.metric("Total Revenue", f"${total_revenue:,.0f}")
    col3.metric("Net Profit", f"${net_profit:,.0f}", delta_color="normal")
    col4.metric("ROAS", f"{roas:.2f}x", delta=f"{roas-1.0:.2f} vs Break-even")
    col5.metric("Conv. Rate", f"{conversion_rate:.2%}")
    
    st.divider()
    
    # 2. Financial Performance Curve (Smoothed)
    st.subheader("Financial Trends (7-Day Rolling Average)")
    st.caption("We use a 7-day rolling average to smooth out daily volatility and reveal the true profit trend.")
    
    # Resample to daily sums first, then apply rolling window
    # We use 'D' frequency to ensure missing days are accounted for (as zeros)
    daily_financials = filtered_df.set_index('timestamp').resample('D').agg({
        'cost_per_click': 'sum', 
        'payout_amount': 'sum', 
        'net_profit': 'sum'
    }).fillna(0)
    
    # Calculate Rolling Averages
    rolling_df = daily_financials.rolling(window=7, min_periods=1).mean().reset_index()
    
    fig_daily = go.Figure()
    
    # Revenue Line (Green)
    fig_daily.add_trace(go.Scatter(
        x=rolling_df['timestamp'], 
        y=rolling_df['payout_amount'], 
        mode='lines', 
        name='Revenue (7d Avg)', 
        line=dict(color='#00CC96', width=3)
    ))
    
    # Spend Line (Red)
    fig_daily.add_trace(go.Scatter(
        x=rolling_df['timestamp'], 
        y=rolling_df['cost_per_click'], 
        mode='lines', 
        name='Spend (7d Avg)', 
        line=dict(color='#EF553B', width=3)
    ))
    
    # Net Profit Area (Gray)
    fig_daily.add_trace(go.Scatter(
        x=rolling_df['timestamp'], 
        y=rolling_df['net_profit'], 
        mode='lines', 
        name='Net Profit (7d Avg)', 
        fill='tozeroy', # Fill to zero line
        line=dict(color='gray', width=0),
        opacity=0.2
    ))
    
    fig_daily.update_layout(
        hovermode="x unified", 
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_daily, width='stretch')

    st.divider()

    # 3. Traffic & Conversions Over Time (By Source)
    # 3. Traffic & Conversions Over Time (By Source)
    st.subheader("Volume Analysis by Source")
    
    # Aggregate data by Date and Traffic Source
    daily_source = filtered_df.groupby([pd.Grouper(key='timestamp', freq='D'), 'traffic_source']).agg({
        'user_id': 'count',     # Traffic Volume
        'policy_sold': 'sum'    # Conversion Volume
    }).reset_index()
    
    # Chart 1: Traffic Volume (Line)
    fig_traffic = px.line(
        daily_source, 
        x='timestamp', 
        y='user_id', 
        color='traffic_source',
        title="Daily Traffic Volume by Source",
        labels={'user_id': 'Visits', 'timestamp': 'Date'},
        template='plotly_white'
    )
    fig_traffic.update_layout(hovermode="x unified")
    st.plotly_chart(fig_traffic, width='stretch')

    st.divider()

    # Chart 2: Conversions (Line)
    fig_conversions = px.line(
        daily_source, 
        x='timestamp', 
        y='policy_sold', 
        color='traffic_source',
        title="Daily Conversions by Source",
        labels={'policy_sold': 'Sales', 'timestamp': 'Date'},
        template='plotly_white'
    )
    fig_conversions.update_layout(hovermode="x unified")
    st.plotly_chart(fig_conversions, width='stretch')

# --- TAB 2: TRAFFIC ---
with tab2:
    st.header("Traffic Comparison")
    
    # 1. Aggregation
    source_stats = filtered_df.groupby('traffic_source').agg(
        Visits=('user_id', 'count'),
        Spend=('cost_per_click', 'sum'),
        Revenue=('payout_amount', 'sum'),
        Net_Profit=('net_profit', 'sum'),
        Conversions=('policy_sold', 'sum')
    ).reset_index()
    
    # 2. Calculate Key Metrics
    source_stats['ROAS'] = source_stats['Revenue'] / source_stats['Spend']
    source_stats['CPA'] = source_stats['Spend'] / source_stats['Conversions']
    source_stats['Conv_Rate'] = source_stats['Conversions'] / source_stats['Visits']
    source_stats['Margin'] = (source_stats['Net_Profit'] / source_stats['Revenue']) * 100
    
    # 3. Source Data Table (Moved to Top)
    st.subheader("Source Performance Data")
    display_cols = ['traffic_source', 'Visits', 'Spend', 'Revenue', 'Net_Profit', 'ROAS', 'CPA', 'Conv_Rate', 'Margin']
    
    st.dataframe(
        source_stats[display_cols].style.format({
            'Spend': '${:,.0f}', 
            'Revenue': '${:,.0f}', 
            'Net_Profit': '${:,.0f}',
            'ROAS': '{:.2f}x', 
            'CPA': '${:.2f}', 
            'Conv_Rate': '{:.2%}', 
            'Margin': '{:.1f}%'
        }).background_gradient(subset=['ROAS', 'Net_Profit'], cmap='RdYlGn', vmin=0.5, vmax=1.5),
        width='stretch'
    )

    st.divider()

    # 4. The "Quadrant" Scatter Plot (Performance Matrix)
    st.subheader("Efficiency Matrix: CPA vs. Conversion Rate")
    st.caption("Identify 'Stars' (Top-Left) vs. 'Waste' (Bottom-Right). Bubble size represents Total Spend.")
    
    # Calculate averages for the "Crosshairs"
    avg_cpa = source_stats['CPA'].mean()
    avg_cv = source_stats['Conv_Rate'].mean()
    
    fig_quad = px.scatter(
        source_stats,
        x="CPA",
        y="Conv_Rate",
        size="Spend",            # Bubble size = Budget allocation
        color="ROAS",            # Bubble color = Profitability
        hover_name="traffic_source",
        text="traffic_source",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=1.0, # Green if profitable (>1), Red if loss (<1)
        hover_data={
            'Spend': ':$,.0f',
            'Net_Profit': ':$,.0f',
            'ROAS': ':.2f',
            'Visits': ':,',
            'Conversions': ':,',
            'CPA': ':.2f',
            'Conv_Rate': False # Hide default, we format it in the axis
        }
    )
    
    # Add Quadrant Lines (Average CPA and Average Conversion Rate)
    fig_quad.add_hline(y=avg_cv, line_dash="dash", line_color="gray", annotation_text="Avg Conv. Rate")
    fig_quad.add_vline(x=avg_cpa, line_dash="dash", line_color="gray", annotation_text="Avg CPA")
    
    # Formatting
    fig_quad.update_traces(textposition='top center', marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    fig_quad.update_layout(
        xaxis_title="CPA (Cost Per Acquisition) - Lower is Better",
        yaxis_title="Conversion Rate - Higher is Better",
        yaxis_tickformat=".1%",
        xaxis_tickformat="$",
        height=600
    )
    
    st.plotly_chart(fig_quad, width='stretch')

# --- TAB (NEW): CAMPAIGNS ---
with tab_campaign:
    st.header("Campaign Performance")
    
    # Check if campaign_id exists
    if 'campaign_id' not in filtered_df.columns:
        st.error("Campaign ID column not found in data.")
    else:
        # 1. Aggregation
        camp_stats = filtered_df.groupby('campaign_id').agg(
            Visits=('user_id', 'count'),
            Spend=('cost_per_click', 'sum'),
            Revenue=('payout_amount', 'sum'),
            Net_Profit=('net_profit', 'sum'),
            Conversions=('policy_sold', 'sum')
        ).reset_index()
        
        # 2. Calculate Key Metrics
        camp_stats['ROAS'] = camp_stats['Revenue'] / camp_stats['Spend']
        camp_stats['CPA'] = camp_stats['Spend'] / camp_stats['Conversions']
        camp_stats['Conv_Rate'] = camp_stats['Conversions'] / camp_stats['Visits']
        camp_stats['Margin'] = (camp_stats['Net_Profit'] / camp_stats['Revenue']) * 100
        
        # 3. Campaign Data Table
        st.subheader("Campaign Performance Data")
        display_cols_camp = ['campaign_id', 'Visits', 'Spend', 'Revenue', 'Net_Profit', 'ROAS', 'CPA', 'Conv_Rate', 'Margin']
        
        st.dataframe(
            camp_stats[display_cols_camp].style.format({
                'Spend': '${:,.0f}', 
                'Revenue': '${:,.0f}', 
                'Net_Profit': '${:,.0f}',
                'ROAS': '{:.2f}x', 
                'CPA': '${:.2f}', 
                'Conv_Rate': '{:.2%}', 
                'Margin': '{:.1f}%'
            }).background_gradient(subset=['ROAS', 'Net_Profit'], cmap='RdYlGn', vmin=0.5, vmax=1.5),
            width='stretch'
        )

        st.divider()

        # 4. Campaign Scatter Plot
        st.subheader("Campaign Efficiency: CPA vs. Conversion Rate")
        
        if not camp_stats.empty:
            avg_cpa_camp = camp_stats['CPA'].mean()
            avg_cv_camp = camp_stats['Conv_Rate'].mean()
            
            fig_camp_quad = px.scatter(
                camp_stats,
                x="CPA",
                y="Conv_Rate",
                size="Spend",
                color="ROAS",
                hover_name="campaign_id",
                text="campaign_id",
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=1.0,
                hover_data={
                    'Spend': ':$,.0f',
                    'Net_Profit': ':$,.0f',
                    'ROAS': ':.2f',
                    'Visits': ',',
                    'Conversions': ',',
                    'CPA': ':.2f',
                    'Conv_Rate': False
                }
            )
            
            fig_camp_quad.add_hline(y=avg_cv_camp, line_dash="dash", line_color="gray", annotation_text="Avg Conv. Rate")
            fig_camp_quad.add_vline(x=avg_cpa_camp, line_dash="dash", line_color="gray", annotation_text="Avg CPA")
            
            fig_camp_quad.update_traces(textposition='top center', marker=dict(line=dict(width=1, color='DarkSlateGrey')))
            fig_camp_quad.update_layout(
                xaxis_title="CPA (Cost Per Acquisition)",
                yaxis_title="Conversion Rate",
                yaxis_tickformat=".1%",
                xaxis_tickformat="$",
                height=600
            )
            
            st.plotly_chart(fig_camp_quad, width='stretch')


# --- TAB_GEO: GEOGRAPHIC ANALYSIS ---
with tab_geo:
    st.header("Geographic Performance")
    st.caption("Conversion Rate by State (USA). Use filters to drill down.")

    # 1. Local Filters (Campaign & Creative)
    col_g1, col_g2 = st.columns(2)
    geo_df = filtered_df.copy()

    with col_g1:
        if 'campaign_id' in geo_df.columns:
            g_campaigns = geo_df['campaign_id'].unique().tolist()
            sel_g_campaigns = st.multiselect("Filter by Campaign", g_campaigns, key="geo_camp")
            if sel_g_campaigns:
                geo_df = geo_df[geo_df['campaign_id'].isin(sel_g_campaigns)]

    with col_g2:
        if 'creative_id' in geo_df.columns:
            g_creatives = geo_df['creative_id'].unique().tolist()
            sel_g_creatives = st.multiselect("Filter by Creative", g_creatives, key="geo_creat")
            if sel_g_creatives:
                geo_df = geo_df[geo_df['creative_id'].isin(sel_g_creatives)]
    
    st.divider()

    # 2. Data Processing for Map
    if 'state' in geo_df.columns:
        state_stats = geo_df.groupby('state').agg(
            Visits=('user_id', 'count'),
            Conversions=('policy_sold', 'sum'),
            CPA=('cost_per_click', 'mean') # Approx CPA for map hover
        ).reset_index()
        
        # Calculate Conversion Rate
        state_stats['Conversion Rate'] = state_stats['Conversions'] / state_stats['Visits']
        
        # 3. Visualization: Choropleth Map
        st.subheader("Conversion Rate by State")
        
        fig_map = px.choropleth(
            state_stats,
            locations='state',
            locationmode="USA-states",
            color='Conversion Rate',
            scope="usa",
            color_continuous_scale="RdYlGn",
            hover_data={'state': True, 'Visits': True, 'Conversions': True, 'CPA': ':.2f', 'Conversion Rate': ':.1%'},
            labels={'Conversion Rate': 'Conv. Rate'}
        )
        fig_map.update_layout(height=600, margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_map, width='stretch')
        
        # 4. Top States Table
        st.subheader("Top Performing States")
        # Filter for meaningful sample size > 50 visits
        top_states = state_stats[state_stats['Visits'] > 50].sort_values('Conversion Rate', ascending=False).head(10)
        st.dataframe(
            top_states.style.format({
                'Conversion Rate': '{:.1%}',
                'CPA': '${:.2f}'
            }), 
            width='stretch'
        )

    else:
        st.warning("State data not found in dataset.")

# --- TAB 3: FUNNEL ANALYSIS---
with tab3:
    st.header("Conversion Funnel Analysis")
    
    funnel_steps = ['landed', 'started_quote', 'completed_quote', 'submitted_to_carrier', 'carrier_accepted', 'policy_sold']
    
    # 1. Funnel Chart (Volume)
    st.subheader("Funnel Volume (Absolute Counts)")
    funnel_data = filtered_df.groupby('traffic_source')[funnel_steps].sum().reset_index()
    
    # Melt for Plotly Funnel
    funnel_melted = funnel_data.melt(
        id_vars='traffic_source', 
        value_vars=funnel_steps,
        var_name='Step', 
        value_name='Users'
    )
    
    fig_funnel = px.funnel(
        funnel_melted, 
        x='Users', 
        y='Step', 
        color='traffic_source',
        title="Funnel Volume by Source",
        template='plotly_white'
    )
    st.plotly_chart(fig_funnel, width='stretch')
    
    st.divider()

    # 2. Step-to-Step Progression
    st.subheader("Step Progression Health")
    st.caption("How many users successfully move to the NEXT step? (Lower is worse)")
    
    # Calculate step-to-step conversion
    progression_data = []
    
    # Group by source first
    grouped_source = filtered_df.groupby('traffic_source')[funnel_steps].sum()
    
    # Iterate through steps to calculate N / N-1
    for i in range(len(funnel_steps) - 1):
        prev_step = funnel_steps[i]
        curr_step = funnel_steps[i+1]
        
        # Calculate conversion rate: curr / prev (handle division by zero if necessary)
        step_conv = (grouped_source[curr_step] / grouped_source[prev_step]).fillna(0)
        
        for source, rate in step_conv.items():
            progression_data.append({
                'traffic_source': source,
                'Transition': f"{prev_step} → {curr_step}",
                'Conversion_Rate': rate
            })
            
    progression_df = pd.DataFrame(progression_data)
    
    fig_progression = px.bar(
        progression_df,
        x='Transition',
        y='Conversion_Rate',
        color='traffic_source', 
        facet_col='traffic_source',
        facet_col_wrap=3,
        title="Micro-Conversion Rates by Source (Drop-off Analysis)",
        labels={'Conversion_Rate': 'Conversion Rate (%)'},
        template='plotly_white'
    )
    fig_progression.update_yaxes(tickformat=".1%")
    # Clean up facet titles
    fig_progression.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    st.plotly_chart(fig_progression, width='stretch')

    st.divider()

    # 3. Drop-off Analysis (Grouped Bar)
    st.subheader("Drop-off Rate Comparison")
    st.caption("Percentage of users lost at each step (Higher is worse). Compare sources side-by-side.")
    
    # Calculate Drop-off Rate
    progression_df['Drop_off_Rate'] = 1 - progression_df['Conversion_Rate']
    
    fig_dropoff = px.bar(
        progression_df,
        x='Transition',
        y='Drop_off_Rate',
        color='traffic_source',
        barmode='group',
        title="Drop-off Rate by Step & Source",
        labels={'Drop_off_Rate': 'Drop-off Rate (%)'},
        template='plotly_white'
    )
    fig_dropoff.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig_dropoff, width='stretch')

# --- TAB 4: SEGMENTS & TARGETING (NEW!) ---
with tab4:
    st.header("Demographic & Technical Segments")
    st.caption("Insights into Device, OS, Browser, and Gender performance. (Filtered by Source above)")
    
    # Optional Campaign & Creative Filters for Drill-down
    col_seg1, col_seg2 = st.columns(2)
    
    seg_df = filtered_df.copy()
    
    with col_seg1:
        if 'campaign_id' in seg_df.columns:
            campaigns = seg_df['campaign_id'].unique().tolist()
            sel_campaigns = st.multiselect("Filter by Campaign (Optional)", campaigns, key="seg_camp")
            if sel_campaigns:
                seg_df = seg_df[seg_df['campaign_id'].isin(sel_campaigns)]
                
    with col_seg2:
        if 'creative_id' in seg_df.columns:
            creatives = seg_df['creative_id'].unique().tolist()
            sel_creatives = st.multiselect("Filter by Creative (Optional)", creatives, key="seg_creat")
            if sel_creatives:
                seg_df = seg_df[seg_df['creative_id'].isin(sel_creatives)]
    
    # List of dimensions to plot
    dimensions = [
        ('creative_id', 'Creative'),
        ('device_type', 'Device Type'),
        ('os', 'Operating System'),
        ('browser', 'Browser'),
        ('gender', 'Gender'),
        ('hour_of_day', 'Hour of Day'),
        ('day_of_week', 'Day of Week'),
        ('vehicles_count', 'Vehicles Count'),
        ('years_driving', 'Years Driving'),
        ('accidents_last_3yr', 'Accidents (Last 3 Years)'),
        ('current_insurer', 'Current Insurer'),
        ('coverage_type', 'Coverage Type')
    ]
    
    # Filter dimensions that exist in the dataframe
    valid_dimensions = [d for d in dimensions if d[0] in seg_df.columns]
    
    # Create grid layout
    cols = st.columns(2) # 2 columns grid
    
    for i, (col_name, title) in enumerate(valid_dimensions):
        with cols[i % 2]:
            st.subheader(title)
            
            # Aggregation
            agg = seg_df.groupby(col_name).agg({
                'user_id': 'count', 
                'policy_sold': 'sum'
            }).reset_index()
            
            agg['Conv_Rate'] = agg['policy_sold'] / agg['user_id']
            
            fig = px.bar(
                agg, 
                x=col_name, 
                y='user_id',
                title=f"Users by {title}", # Clean Title
                labels={'user_id': 'Users'},
                color='Conv_Rate',
                color_continuous_scale='RdYlGn', # Performance Color Scale
                color_continuous_midpoint=agg['Conv_Rate'].mean() if not agg.empty else None
            )
            st.plotly_chart(fig, width='stretch')

# --- TAB: DETAILED DATA (NEW) ---
with tab_detailed:
    st.header("Detailed Data & Performance Analysis")
    st.caption("Granular breakdown by Traffic Source and Campaign ID. Use this to identify specific winners and losers.")
    
    # 1. Aggregation (Grouping by Source + Campaign + Creative)
    # Check if necessary columns exist
    group_cols = ['traffic_source']
    if 'campaign_id' in filtered_df.columns:
        group_cols.append('campaign_id')
    if 'creative_id' in filtered_df.columns:
        group_cols.append('creative_id')
    
    detailed_stats = filtered_df.groupby(group_cols).agg(
        Visits=('user_id', 'count'),
        Spend=('cost_per_click', 'sum'),
        Revenue=('payout_amount', 'sum'),
        Net_Profit=('net_profit', 'sum'),
        Conversions=('policy_sold', 'sum')
    ).reset_index()
    
    # 2. Key Metrics
    detailed_stats['ROAS'] = detailed_stats['Revenue'] / detailed_stats['Spend']
    detailed_stats['CPA'] = detailed_stats['Spend'] / detailed_stats['Conversions']
    detailed_stats['Conv_Rate'] = detailed_stats['Conversions'] / detailed_stats['Visits']
    detailed_stats['Margin'] = (detailed_stats['Net_Profit'] / detailed_stats['Revenue']) * 100
    
    # 3. Styling & Display
    # Columns to show
    show_cols = group_cols + ['Visits', 'Spend', 'Revenue', 'Net_Profit', 'ROAS', 'CPA', 'Conv_Rate', 'Margin']
    
    st.dataframe(
        detailed_stats[show_cols].style.format({
            'Spend': '${:,.0f}', 
            'Revenue': '${:,.0f}', 
            'Net_Profit': '${:,.0f}',
            'ROAS': '{:.2f}x', 
            'CPA': '${:.2f}', 
            'Conv_Rate': '{:.2%}', 
            'Margin': '{:.1f}%'
        }).background_gradient(subset=['ROAS', 'Net_Profit'], cmap='RdYlGn', vmin=0.5, vmax=1.5),
        width='stretch'
    )

# --- TAB: TIME SERIES ANALYSIS (NEW) ---
with tab_ts:
    st.header("Time Series Analysis")
    st.caption("Analyze trends over time with custom filters and metrics.")
    
    # 1. Local Drill-down Filters
    col_ts1, col_ts2 = st.columns(2)
    
    ts_df = filtered_df.copy()
    
    with col_ts1:
        if 'campaign_id' in ts_df.columns:
            ts_campaigns = st.multiselect("Filter by Campaign (Time Series Only)", ts_df['campaign_id'].unique(), key="ts_camp")
            if ts_campaigns:
                ts_df = ts_df[ts_df['campaign_id'].isin(ts_campaigns)]
                
    with col_ts2:
        if 'creative_id' in ts_df.columns:
            ts_creatives = st.multiselect("Filter by Creative (Time Series Only)", ts_df['creative_id'].unique(), key="ts_creat")
            if ts_creatives:
                ts_df = ts_df[ts_df['creative_id'].isin(ts_creatives)]
                
    st.divider()
    
    # 2. Daily Aggregation
    daily_ts = ts_df.groupby(pd.Grouper(key='timestamp', freq='D')).agg({
        'user_id': 'count',
        'cost_per_click': 'sum',
        'payout_amount': 'sum',
        'net_profit': 'sum',
        'policy_sold': 'sum'
    }).reset_index()
    
    # Calculate derived metrics
    daily_ts.rename(columns={'user_id': 'Visits', 'cost_per_click': 'Spend', 'payout_amount': 'Revenue', 'net_profit': 'Net_Profit', 'policy_sold': 'Conversions'}, inplace=True)
    daily_ts['ROAS'] = daily_ts['Revenue'] / daily_ts['Spend']
    daily_ts['CPA'] = daily_ts['Spend'] / daily_ts['Conversions']
    daily_ts['Conv_Rate'] = daily_ts['Conversions'] / daily_ts['Visits']
    daily_ts['Margin'] = (daily_ts['Net_Profit'] / daily_ts['Revenue']) * 100
    
    # 3. Plotting
    all_metrics = ['Visits', 'Spend', 'Revenue', 'Net_Profit', 'Conversions', 'ROAS', 'CPA', 'Conv_Rate', 'Margin']
    default_metrics = ['Revenue', 'Spend']
    
    selected_metrics = st.multiselect("Select Metrics to Plot", all_metrics, default=default_metrics)
    
    if not selected_metrics:
        st.warning("Please select at least one metric to plot.")
    else:
        fig_ts = px.line(
            daily_ts,
            x='timestamp',
            y=selected_metrics,
            title="Performance Trends Over Time",
            markers=True,
            template='plotly_white'
        )
        fig_ts.update_layout(hovermode="x unified", yaxis_title="Value")
        st.plotly_chart(fig_ts, width='stretch')

# --- TAB 5: SUS TRAFFIC (RENAMED) ---
with tab5:
    st.header("Sus Traffic Inspection")
    st.caption("Identify suspicious activity by auditing Traffic Source, Campaign, Device, and OS.")
    
    # 1. Local Drill-down Filters
    sus_df = filtered_df.copy()
    col_s1, col_s2, col_s3 = st.columns(3)
    
    with col_s1:
        if 'utm_medium' in sus_df.columns:
            sus_mediums = st.multiselect("Filter by UTM Medium", sus_df['utm_medium'].unique(), key="sus_med")
            if sus_mediums:
                sus_df = sus_df[sus_df['utm_medium'].isin(sus_mediums)]
    
    with col_s2:
        if 'device_type' in sus_df.columns:
            sus_devices = st.multiselect("Filter by Device", sus_df['device_type'].unique(), key="sus_dev")
            if sus_devices:
                sus_df = sus_df[sus_df['device_type'].isin(sus_devices)]
                
    with col_s3:
        if 'os' in sus_df.columns:
            sus_os = st.multiselect("Filter by OS", sus_df['os'].unique(), key="sus_os")
            if sus_os:
                sus_df = sus_df[sus_df['os'].isin(sus_os)]
    
    st.divider()

    # 2. Risk Analysis Scatter Plot (Volume vs Fraud Rate)
    st.subheader("Risk Analysis: Volume vs. Fraud Rate")
    st.caption("Identify high-volume/high-fraud outliers. Bubble size = Spend.")
    
    # Group by Source and Campaign
    if 'campaign_id' in sus_df.columns:
        fraud_scatter = sus_df.groupby(['traffic_source', 'campaign_id']).agg(
            Visits=('user_id', 'count'),
            Spend=('cost_per_click', 'sum'),
            Fraud_Rate=('_is_fraud_synthetic', 'mean')
        ).reset_index()
        
        if not fraud_scatter.empty:
            fig_sus = px.scatter(
                fraud_scatter,
                x='Visits',
                y='Fraud_Rate',
                size='Spend',
                color='traffic_source',
                hover_name='campaign_id',
                title="Fraud Rate by Campaign (Size=Spend)",
                labels={'Fraud_Rate': 'Fraud Rate (0-1)'},
                color_discrete_sequence=px.colors.qualitative.Bold,
                height=700
            )
            # Add threshold line (e.g., 5% fraud)
            fig_sus.add_hline(y=0.05, line_dash="dash", line_color="red", annotation_text="5% Fraud Threshold")
            
            st.plotly_chart(fig_sus, width='stretch')
        else:
            st.info("No data available for scatter plot.")
    else:
        st.warning("Campaign ID not found for scatter plot.")

    st.divider()
    
    # 3. Existing Charts (Updated to use sus_df)
    st.subheader("Fraud Rate by Source")
    fraud_by_source = sus_df.groupby('traffic_source')['_is_fraud_synthetic'].mean().reset_index()
    fraud_by_source.columns = ['Source', 'Fraud Rate']
    
    fig_fraud = px.bar(fraud_by_source, x='Source', y='Fraud Rate', color='Fraud Rate', title='Fraud Rate by Source', color_continuous_scale='Reds')
    st.plotly_chart(fig_fraud, width='stretch')
    
    st.subheader("Suspicious Publishers")
    pub_fraud = sus_df.groupby(['traffic_source', 'publisher_id']).agg(
        Visits=('user_id', 'count'),
        Fraud_Rate=('_is_fraud_synthetic', 'mean')
    ).reset_index()
    
    bad_pubs = pub_fraud[pub_fraud['Visits'] > 50].sort_values('Fraud_Rate', ascending=False).head(10)
    st.table(bad_pubs.style.format({'Fraud_Rate': '{:.1%}'}))

# --- TAB 6: LINEAR REGRESSION (NEW) ---
with tab6:
    st.header("Linear Regression Analysis")
    st.caption("Predict key metrics based on user features. Use the sidebar to adjust parameters.")
    
    # 1. Configuration (Sidebar-like, but in main page for better flow)
    st.subheader("Model Configuration")
    
    lr_col1, lr_col2, lr_col3 = st.columns(3)
    
    # Available numeric columns for features
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    # Remove obvious target leaks or IDs
    exclude_cols = ['user_id', 'timestamp', 'campaign_id', 'publisher_id', 'utm_source', 'utm_medium', 'creative_id', 'device_type', 'os', 'browser', 'screen_resolution', 'state', 'user_agent_hash', 'zip_code', 'landed', 'started_quote', 'completed_quote', 'submitted_to_carrier', 'carrier_accepted', 'policy_sold', '_is_fraud_synthetic']
    feature_candidates = [c for c in numeric_cols if c not in exclude_cols]
    
    # Default Selection
    default_features = ['age', 'cost_per_click', 'session_duration_sec', 'pages_viewed']
    default_features = [f for f in default_features if f in feature_candidates]
    
    with lr_col1:
        target_col = st.selectbox("Select Target Variable (Y)", ['payout_amount', 'net_profit', 'session_duration_sec'], index=0)
    
    with lr_col2:
        feature_cols = st.multiselect("Select Feature Variables (X)", feature_candidates, default=default_features)
    
    with lr_col3:
        test_size = st.slider("Test Set Size (%)", 10, 50, 20) / 100
        
    st.divider()
    
    # 2. Model Training
    if feature_cols and target_col:
        # Prepare Data
        # Filter out NaNs if any (though synthetic data is clean)
        model_df = df[feature_cols + [target_col]].dropna()
        
        X = model_df[feature_cols]
        y = model_df[target_col]
        
        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Train
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        # 3. Model Performance Display
        st.subheader("Model Performance")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("R² Score", f"{r2:.4f}")
        m_col2.metric("Mean Squared Error", f"{mse:,.2f}")
        m_col3.metric("Training Samples", f"{len(X_train):,}")
        
        # 4. Visualizations
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            st.subheader("Actual vs. Predicted")
            # Sample for scatter plot if too large
            plot_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
            if len(plot_df) > 1000:
                plot_df = plot_df.sample(1000)
                
            fig_perf = px.scatter(
                plot_df, x='Actual', y='Predicted', 
                title=f"Actual vs Predicted {target_col}",
                opacity=0.6, template='plotly_white'
            )
            # Add perfect prediction line
            min_val = min(plot_df['Actual'].min(), plot_df['Predicted'].min())
            max_val = max(plot_df['Actual'].max(), plot_df['Predicted'].max())
            fig_perf.add_shape(
                type="line", line=dict(dash='dash', color='red'),
                x0=min_val, y0=min_val, x1=max_val, y1=max_val
            )
            st.plotly_chart(fig_perf, width='stretch')
            
        with viz_col2:
            st.subheader("Feature Importance (Coefficients)")
            coef_df = pd.DataFrame({
                'Feature': feature_cols,
                'Coefficient': model.coef_
            }).sort_values('Coefficient', key=abs, ascending=True)
            
            fig_coef = px.bar(
                coef_df, x='Coefficient', y='Feature', orientation='h',
                title="Linear Coefficients",
                template='plotly_white',
                color='Coefficient', color_continuous_scale='RdBu'
            )
            st.plotly_chart(fig_coef, width='stretch')
            
    else:
        st.warning("Please select at least one feature and a target variable.")

# --- TAB 7: CLUSTERING (NEW) ---
with tab7:
    st.header("User Segmentation (Clustering)")
    st.caption("Identify distinct user groups using K-Means Clustering. Visualized using PCA (2D projection).")
    
    # 1. Configuration
    st.subheader("Clustering Configuration")
    
    cl_col1, cl_col2 = st.columns(2)
    
    # Available numeric columns
    numeric_cols_cl = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    exclude_cols_cl = ['user_id', 'timestamp', 'campaign_id', 'publisher_id', 'utm_source', 'utm_medium', 'creative_id', 'device_type', 'os', 'browser', 'screen_resolution', 'state', 'user_agent_hash', 'zip_code', 'landed', 'started_quote', 'completed_quote', 'submitted_to_carrier', 'carrier_accepted', 'policy_sold', '_is_fraud_synthetic']
    feature_candidates_cl = [c for c in numeric_cols_cl if c not in exclude_cols_cl]
    
    # Default Selection
    default_features_cl = ['age', 'session_duration_sec', 'pages_viewed', 'scroll_depth_pct', 'mouse_movement_score']
    default_features_cl = [f for f in default_features_cl if f in feature_candidates_cl]
    
    with cl_col1:
        cluster_features = st.multiselect("Select Features for Clustering", feature_candidates_cl, default=default_features_cl)
        
    with cl_col2:
        n_clusters = st.slider("Number of Clusters (K)", 2, 8, 3)
        
    st.divider()
    
    if cluster_features:
        # 2. Data Preparation
        # Drop NaNs
        cluster_df = df[cluster_features].dropna()
        
        # Scaling is crucial for K-Means
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_df)
        
        # 3. K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled_data)
        
        # Add labels back to original (subset) df for interpretation
        cluster_df['Cluster'] = labels.astype(str)
        
        # 4. Visualization (PCA)
        st.subheader("Cluster Visualization (PCA 2D)")
        
        # Reduce to 2 components for plotting
        pca = PCA(n_components=2)
        pca_components = pca.fit_transform(scaled_data)
        
        pca_df = pd.DataFrame(data=pca_components, columns=['PCA1', 'PCA2'])
        pca_df['Cluster'] = labels.astype(str)
        
        # Sample for performance if needed
        if len(pca_df) > 2000:
            pca_df = pca_df.sample(2000, random_state=42)
            
        fig_pca = px.scatter(
            pca_df, x='PCA1', y='PCA2', 
            color='Cluster',
            title=f"User Segments (K={n_clusters})",
            template='plotly_white',
            opacity=0.7,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        st.plotly_chart(fig_pca, width='stretch')
        
        st.divider()
        
        # 5. Cluster Interpretation
        st.subheader("Cluster Characteristics (Centroids)")
        
        # Calculate mean of original features for each cluster
        # We re-attach the labels to the original unscaled data (subset match)
        # Note: We need careful index alignment if we dropped NaNs. 
        # Here cluster_df is already aligned with labels.
        
        centroid_stats = cluster_df.groupby('Cluster')[cluster_features].mean().reset_index()
        
        # Add Count
        counts = cluster_df['Cluster'].value_counts().reset_index()
        counts.columns = ['Cluster', 'Count']
        centroid_stats = centroid_stats.merge(counts, on='Cluster')
        
        # Reorder columns
        cols = ['Cluster', 'Count'] + cluster_features
        
        st.dataframe(
            centroid_stats[cols].style.background_gradient(cmap='Blues', subset=cluster_features),
            width='stretch'
        )
        
    else:
        st.warning("Please select at least one feature for clustering.")

