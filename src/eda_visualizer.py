from typing import Optional

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Shared style constants ─────────────────────────────────────────────────────
_PALETTE = ['#3498db','#e74c3c','#2ecc71','#f39c12','#9b59b6']
_WHO_COLORS = {'Quiet':'#2ecc71','Moderate':'#f1c40f','Loud':'#e67e22','Very Loud':'#e74c3c'}
_MONTH_NAMES = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
_DAY_NAMES   = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

def _apply_style(
    fig,
    height=400,
    *,
    bottom_legend: bool = True,
    margin: Optional[dict] = None,
    show_legend: bool = True,
    polar: bool = False,
):
    """Uniform margins: legend below the plot area to avoid overlapping the title."""
    m = dict(t=58, b=118 if (bottom_legend and show_legend) else 56, l=56, r=28)
    if margin:
        m.update(margin)
    leg = None
    if show_legend:
        leg = dict(
            orientation='h',
            yanchor='top',
            y=-0.22,
            xanchor='center',
            x=0.5,
            font=dict(size=11),
            bgcolor='rgba(255,255,255,0.75)',
        )
    layout = dict(
        height=height,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif', size=12),
        margin=m,
        legend=leg,
        title=dict(x=0, xanchor='left', pad=dict(t=4)),
    )
    if not polar:
        layout['xaxis'] = dict(showgrid=True, gridcolor='rgba(128,128,128,0.15)', zeroline=False)
        layout['yaxis'] = dict(showgrid=True, gridcolor='rgba(128,128,128,0.15)', zeroline=False)
    fig.update_layout(**layout)
    if not show_legend:
        fig.update_layout(showlegend=False, legend=None)
    return fig

def _who_bands(fig):
    """WHO threshold bands + dashed reference lines (no side annotations — avoids legend/title clash)."""
    fig.add_hrect(y0=0,  y1=55, fillcolor='#2ecc71', opacity=0.06, line_width=0, layer='below')
    fig.add_hrect(y0=55, y1=65, fillcolor='#f1c40f', opacity=0.06, line_width=0, layer='below')
    fig.add_hrect(y0=65, y1=75, fillcolor='#e67e22', opacity=0.06, line_width=0, layer='below')
    fig.add_hrect(y0=75, y1=100, fillcolor='#e74c3c', opacity=0.06, line_width=0, layer='below')
    for y, color in [(55, '#2ecc71'), (65, '#f39c12'), (75, '#e74c3c')]:
        fig.add_hline(
            y=y, line_dash='dot', line_color=color, line_width=1.5,
            annotation_text='', layer='below',
        )
    return fig

def _noise_zone(db):
    if db < 55:  return 'Quiet'
    if db < 65:  return 'Moderate'
    if db < 75:  return 'Loud'
    return 'Very Loud'


class EDAVisualizer:
    def __init__(self, data_path='data/processed_data.csv'):
        self._raw = pd.read_csv(data_path)
        self._raw['DateTime'] = pd.to_datetime(self._raw['DateTime'])
        self._raw['noise_zone'] = self._raw['Noise_Level_dB'].apply(_noise_zone)
        self.cities = sorted(self._raw['City'].unique().tolist())

    @property
    def data(self):
        return self._raw

    # ── Stats ──────────────────────────────────────────────────────────────────
    def get_stats(self, data=None):
        d = data if data is not None else self._raw
        return {
            "total_rows":         len(d),
            "cities":             self.cities,
            "noise_mean":         round(d['Noise_Level_dB'].mean(), 2),
            "noise_max":          d['Noise_Level_dB'].max(),
            "noise_min":          d['Noise_Level_dB'].min(),
            "noise_std":          round(d['Noise_Level_dB'].std(), 2),
            "traffic_mean":       round(d['Traffic_Count'].mean(), 0),
            "corr_noise_traffic": round(d['Noise_Level_dB'].corr(d['Traffic_Count']), 3),
            "corr_noise_pm25":    round(d['Noise_Level_dB'].corr(d['pm25']), 3),
            "corr_noise_humidity":round(d['Noise_Level_dB'].corr(d['humidity']), 3),
            "pct_above_65":       round(100*(d['Noise_Level_dB']>65).mean(), 1),
            "pct_above_75":       round(100*(d['Noise_Level_dB']>75).mean(), 1),
        }
    # ── Tab 1: Time Patterns ───────────────────────────────────────────────────
    def _fig_noise_by_hour(self, data):
        hourly = data.groupby(['hour','City'])['Noise_Level_dB'].agg(['mean','std']).reset_index()
        hourly.columns = ['hour','City','mean','std']
        fig = go.Figure()
        for i, city in enumerate(sorted(data['City'].unique())):
            d = hourly[hourly['City']==city]
            fig.add_trace(go.Scatter(x=d['hour'], y=d['mean'], name=city,
                mode='lines+markers', line=dict(width=2.5, color=_PALETTE[i % len(_PALETTE)]),
                marker=dict(size=6),
                error_y=dict(type='data', array=d['std'], visible=True, thickness=1, width=3)))
        _who_bands(fig)
        fig.update_layout(title='Average Noise Level by Hour of Day',
                          xaxis_title='Hour of Day', yaxis_title='Avg Noise (dB)',
                          xaxis=dict(tickmode='linear', dtick=1))
        return _apply_style(fig, 440)

    def _fig_heatmap_hour_day(self, data):
        pivot = data.groupby(['day_of_week','hour'])['Noise_Level_dB'].mean().unstack()
        fig = go.Figure(go.Heatmap(
            z=pivot.values, x=list(range(24)), y=_DAY_NAMES,
            colorscale='RdYlGn_r',
            colorbar=dict(title=dict(text='dB'), thickness=12, len=0.72),
            hovertemplate='%{y} %{x}:00 -> %{z:.1f} dB<extra></extra>',
            zmin=data['Noise_Level_dB'].quantile(0.05),
            zmax=data['Noise_Level_dB'].quantile(0.95),
        ))
        fig.update_layout(title='Noise Heatmap: Day of Week x Hour',
                          xaxis_title='Hour of Day', yaxis_title='')
        return _apply_style(fig, 360, bottom_legend=False, show_legend=False,
                          margin=dict(r=92, b=52, l=56))

    def _fig_noise_by_day(self, data):
        """Average noise by day of week — more meaningful than monthly for 6-month data."""
        daily = data.groupby(['day_of_week', 'City'])['Noise_Level_dB'].mean().reset_index()
        daily['Day'] = daily['day_of_week'].map(dict(enumerate(_DAY_NAMES)))
        fig = px.bar(daily, x='Day', y='Noise_Level_dB', color='City',
                     barmode='group',
                     color_discrete_sequence=_PALETTE,
                     title='Average Noise Level by Day of Week',
                     labels={'Noise_Level_dB': 'Avg Noise (dB)', 'Day': 'Day of Week'},
                     category_orders={'Day': _DAY_NAMES})
        _who_bands(fig)
        return _apply_style(fig, 420)

    def _fig_noise_by_hour_stacked(self, data):
        """Stacked area showing noise zone proportions by hour."""
        zones = ['Quiet','Moderate','Loud','Very Loud']
        rows = []
        for h in range(24):
            hd = data[data['hour']==h]
            total = max(len(hd), 1)
            for z in zones:
                rows.append({'Hour': h, 'Zone': z,
                             'Pct': round(100*(hd['noise_zone']==z).sum()/total, 1)})
        df_z = pd.DataFrame(rows)
        fig = px.area(df_z, x='Hour', y='Pct', color='Zone',
                      color_discrete_map=_WHO_COLORS,
                      title='Noise Zone Distribution by Hour (%)',
                      labels={'Pct': '% of Readings', 'Hour': 'Hour of Day'})
        fig.update_layout(xaxis=dict(tickmode='linear', dtick=1))
        return _apply_style(fig, 380)

    # ── Tab 2: Correlations ────────────────────────────────────────────────────
    def _fig_traffic_vs_noise(self, data):
        sample = data.sample(min(3000, len(data)), random_state=1)
        fig = px.scatter(sample, x='Traffic_Count', y='Noise_Level_dB',
                         color='City', opacity=0.45, trendline='ols',
                         color_discrete_sequence=_PALETTE,
                         title='Traffic Count vs Noise Level',
                         labels={'Traffic_Count':'Vehicles/hr','Noise_Level_dB':'Noise (dB)'})
        _who_bands(fig)
        return _apply_style(fig, 440)

    def _fig_humidity_vs_noise(self, data):
        sample = data.sample(min(3000, len(data)), random_state=4)
        fig = px.scatter(sample, x='humidity', y='Noise_Level_dB',
                         color='City', opacity=0.4, trendline='ols',
                         color_discrete_sequence=_PALETTE,
                         title='Humidity vs Noise Level',
                         labels={'humidity':'Relative Humidity (%)','Noise_Level_dB':'Noise (dB)'})
        return _apply_style(fig, 400)

    def _fig_precipitation_effect(self, data):
        d = data.copy()
        d['Rain'] = d['precipitation'].apply(lambda x: 'Raining' if x > 0.1 else 'Dry')
        fig = px.box(d, x='City', y='Noise_Level_dB', color='Rain',
                     color_discrete_map={'Raining':'#3498db','Dry':'#e67e22'},
                     title='Effect of Precipitation on Noise Level',
                     labels={'Noise_Level_dB':'Noise (dB)'})
        return _apply_style(fig, 400)

    def _fig_correlation_heatmap(self, data):
        cols = ['Noise_Level_dB','Traffic_Count','temperature','wind_speed',
                'pm25','humidity','precipitation','is_weekend','is_rush_hour','is_night']
        corr = data[cols].corr().round(2)
        labels = ['Noise','Traffic','Temp','Wind','PM2.5','Humidity','Precip','Weekend','Rush','Night']
        fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                        x=labels, y=labels,
                        title='Feature Correlation Matrix', zmin=-1, zmax=1)
        fig.update_traces(textfont_size=10)
        fig.update_xaxes(tickangle=-35, side='bottom')
        fig.update_layout(coloraxis_colorbar=dict(thickness=14, len=0.72, title=dict(text='r')))
        return _apply_style(fig, 500, bottom_legend=False, show_legend=False,
                          margin=dict(r=108, b=96, l=72, t=58))

    # ── Tab 3: City Comparison ─────────────────────────────────────────────────
    def _fig_noise_by_city(self, data):
        fig = px.box(data, x='City', y='Noise_Level_dB', color='City',
                     color_discrete_sequence=_PALETTE,
                     title='Noise Level Distribution by City',
                     labels={'Noise_Level_dB':'Noise (dB)'}, points='outliers')
        _who_bands(fig)
        return _apply_style(fig, 420)

    def _fig_weekend_vs_weekday(self, data):
        d = data.copy()
        d['Day Type'] = d['is_weekend'].map({1:'Weekend', 0:'Weekday'})
        fig = px.violin(d, x='City', y='Noise_Level_dB', color='Day Type',
                        box=True, points=False,
                        color_discrete_map={'Weekend':'#9b59b6','Weekday':'#3498db'},
                        title='Weekday vs Weekend Noise Distribution by City')
        _who_bands(fig)
        return _apply_style(fig, 420)

    def _fig_rush_vs_offpeak(self, data):
        d = data.copy()
        d['Period'] = d['is_rush_hour'].map({1:'Rush Hour', 0:'Off-Peak'})
        fig = px.box(d, x='City', y='Noise_Level_dB', color='Period',
                     color_discrete_map={'Rush Hour':'#e74c3c','Off-Peak':'#3498db'},
                     title='Rush Hour vs Off-Peak Noise by City', points='outliers')
        _who_bands(fig)
        return _apply_style(fig, 420)

    def _fig_city_radar(self, data):
        """Radar chart comparing cities across multiple noise metrics."""
        metrics = ['Avg dB','Max dB','Rush dB','Night dB','Weekend dB']
        fig = go.Figure()
        for i, city in enumerate(sorted(data['City'].unique())):
            d = data[data['City']==city]
            vals = [
                round(d['Noise_Level_dB'].mean(), 1),
                round(d['Noise_Level_dB'].max(), 1),
                round(d[d['is_rush_hour']==1]['Noise_Level_dB'].mean(), 1),
                round(d[d['is_night']==1]['Noise_Level_dB'].mean(), 1),
                round(d[d['is_weekend']==1]['Noise_Level_dB'].mean(), 1),
            ]
            fig.add_trace(go.Scatterpolar(r=vals+[vals[0]], theta=metrics+[metrics[0]],
                fill='toself', name=city, opacity=0.6,
                line=dict(color=_PALETTE[i % len(_PALETTE)])))
        fig.update_layout(
            title='City Noise Profile Radar',
            polar=dict(radialaxis=dict(visible=True, range=[50, 85])),
        )
        return _apply_style(fig, 460, bottom_legend=True, margin=dict(b=128, t=58), polar=True)

    # ── Tab 4: Distribution ────────────────────────────────────────────────────
    def _fig_noise_distribution(self, data):
        fig = px.histogram(data, x='Noise_Level_dB', color='City',
                           nbins=50, barmode='overlay', opacity=0.65,
                           color_discrete_sequence=_PALETTE,
                           title='Noise Level Distribution by City',
                           labels={'Noise_Level_dB':'Noise (dB)'})
        for y, _, color in [(55, 'Quiet', '#2ecc71'), (65, 'Moderate', '#f39c12'), (75, 'Loud', '#e74c3c')]:
            fig.add_vline(x=y, line_dash='dot', line_color=color, line_width=2)
        return _apply_style(fig, 420)

    def _fig_noise_by_category(self, data):
        counts = data['noise_zone'].value_counts().reset_index()
        counts.columns = ['Category','Count']
        counts['Pct'] = (counts['Count']/counts['Count'].sum()*100).round(1)
        fig = px.bar(counts, x='Category', y='Count', color='Category',
                     color_discrete_map=_WHO_COLORS, text='Pct',
                     title='WHO Noise Category Breakdown',
                     labels={'Count':'Number of Readings'})
        fig.update_traces(texttemplate='%{text}%', textposition='inside', insidetextanchor='middle')
        fig.update_layout(showlegend=False)
        return _apply_style(fig, 380, bottom_legend=False, show_legend=False)

    def _fig_cdf(self, data):
        """Cumulative distribution of noise levels per city."""
        fig = go.Figure()
        for i, city in enumerate(sorted(data['City'].unique())):
            vals = np.sort(data[data['City']==city]['Noise_Level_dB'].values)
            cdf  = np.arange(1, len(vals)+1) / len(vals) * 100
            fig.add_trace(go.Scatter(x=vals, y=cdf, name=city, mode='lines',
                line=dict(width=2.5, color=_PALETTE[i % len(_PALETTE)])))
        for y, _, color in [(55, '55', '#2ecc71'), (65, '65', '#f39c12'), (75, '75', '#e74c3c')]:
            fig.add_vline(x=y, line_dash='dot', line_color=color, line_width=1.5)
        fig.update_layout(title='Cumulative Distribution of Noise Levels',
                          xaxis_title='Noise Level (dB)', yaxis_title='Cumulative % of Readings')
        return _apply_style(fig, 400)

    def _fig_pm25_vs_noise(self, data):
        sample = data.sample(min(2000, len(data)), random_state=2)
        fig = px.scatter(sample, x='pm25', y='Noise_Level_dB', color='City',
                         opacity=0.45, trendline='ols',
                         color_discrete_sequence=_PALETTE,
                         title='PM2.5 Air Quality vs Noise Level',
                         labels={'pm25':'PM2.5 (ug/m3)','Noise_Level_dB':'Noise (dB)'})
        return _apply_style(fig, 400)

    # ── Tab 5: Statistical Summary ─────────────────────────────────────────────
    def _city_stats_table(self, data):
        rows = []
        for city in sorted(data['City'].unique()):
            d = data[data['City']==city]
            rows.append({
                'City':        city,
                'Avg dB':      round(d['Noise_Level_dB'].mean(), 1),
                'Std dB':      round(d['Noise_Level_dB'].std(), 1),
                'Min dB':      round(d['Noise_Level_dB'].min(), 1),
                'Max dB':      round(d['Noise_Level_dB'].max(), 1),
                'Rush Avg':    round(d[d['is_rush_hour']==1]['Noise_Level_dB'].mean(), 1),
                'Night Avg':   round(d[d['is_night']==1]['Noise_Level_dB'].mean(), 1),
                'Weekend Avg': round(d[d['is_weekend']==1]['Noise_Level_dB'].mean(), 1),
                '% > 65 dB':   round(100*(d['Noise_Level_dB']>65).mean(), 1),
                '% > 75 dB':   round(100*(d['Noise_Level_dB']>75).mean(), 1),
            })
        return pd.DataFrame(rows)

    # ── Main Streamlit display ─────────────────────────────────────────────────
    def display_eda_in_streamlit(self):
        st.subheader("Interactive charts")
        st.caption(
            "Filter by city below. WHO background bands: green <55, yellow 55–65, orange 65–75, red >75 dB "
            "(reference lines at 55 / 65 / 75 — see legend under each chart)."
        )

        stats = self.get_stats()

        # ── KPI cards (two rows — avoids cramped metrics on narrow screens) ───
        st.markdown("#### Dataset overview")
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        r1c1.metric("Records", f"{stats['total_rows']:,}")
        r1c2.metric("Cities", len(stats['cities']))
        r1c3.metric("Avg noise", f"{stats['noise_mean']} dB")
        r1c4.metric("Std dev", f"{stats['noise_std']} dB")
        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        r2c1.metric("Max noise", f"{stats['noise_max']} dB")
        r2c2.metric("Traffic ↔ noise (r)", stats['corr_noise_traffic'])
        r2c3.metric("% readings > 65 dB", f"{stats['pct_above_65']}%")
        r2c4.metric("% readings > 75 dB", f"{stats['pct_above_75']}%")

        st.divider()

        # ── Filters ────────────────────────────────────────────────────────────
        col_f, col_dl = st.columns([4, 1])
        with col_f:
            selected = st.multiselect("Filter by City", self.cities, default=self.cities,
                                      help="Select one or more cities to filter all charts below.")
        with col_dl:
            st.write("")
            csv = self._raw[self._raw['City'].isin(selected)].to_csv(index=False)
            st.download_button("Download CSV", csv, "noise_data.csv", "text/csv",
                               use_container_width=True)

        if not selected:
            st.warning("Please select at least one city.")
            return

        filtered = self._raw[self._raw['City'].isin(selected)].copy()

        # ── Tabs ───────────────────────────────────────────────────────────────
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Time Patterns", "Correlations", "City Comparison",
            "Distribution", "Statistical Summary"
        ])

        with tab1:
            st.plotly_chart(self._fig_noise_by_hour(filtered), use_container_width=True)
            st.caption("Error bars show standard deviation. WHO threshold lines indicate safe/caution/danger zones.")
            st.divider()
            st.plotly_chart(self._fig_heatmap_hour_day(filtered), use_container_width=True)
            st.caption("Darker red = louder. Rush hours (7-9, 16-19) and weekdays show consistently higher noise.")
            st.divider()
            st.plotly_chart(self._fig_noise_by_hour_stacked(filtered), use_container_width=True)
            st.caption("Proportion of readings in each WHO noise zone per hour. Shows how risk exposure shifts throughout the day.")
            st.divider()
            st.plotly_chart(self._fig_noise_by_day(filtered), use_container_width=True)
            st.caption("Weekdays (Mon–Fri) are consistently louder than weekends due to commuter traffic. Friday shows the highest peak in most cities.")

        with tab2:
            st.plotly_chart(self._fig_traffic_vs_noise(filtered), use_container_width=True)
            st.caption("OLS trendlines show the linear relationship between traffic volume and noise. Traffic is the dominant driver.")
            st.divider()
            st.plotly_chart(self._fig_correlation_heatmap(filtered), use_container_width=True)
            st.caption("Pearson correlation matrix. Traffic and rush hour show the strongest positive correlation with noise.")
            st.divider()
            col_a, col_b = st.columns(2)
            with col_a:
                st.plotly_chart(self._fig_humidity_vs_noise(filtered), use_container_width=True)
                st.caption("Higher humidity slightly increases PM2.5 readings, which correlates with noise.")
            with col_b:
                st.plotly_chart(self._fig_precipitation_effect(filtered), use_container_width=True)
                st.caption("Rain reduces traffic and thus noise, but adds ambient precipitation sound.")
            st.divider()
            st.plotly_chart(self._fig_pm25_vs_noise(filtered), use_container_width=True)
            st.caption("PM2.5 is estimated from traffic and humidity — its correlation with noise reflects shared traffic origin.")

        with tab3:
            st.plotly_chart(self._fig_noise_by_city(filtered), use_container_width=True)
            st.caption("Box plots show median, IQR, and outliers. Cairo shows the highest median and spread.")
            st.divider()
            st.plotly_chart(self._fig_city_radar(filtered), use_container_width=True)
            st.caption("Radar chart compares cities across 5 noise dimensions. Larger area = noisier overall profile.")
            st.divider()
            col_c, col_d = st.columns(2)
            with col_c:
                st.plotly_chart(self._fig_weekend_vs_weekday(filtered), use_container_width=True)
                st.caption("Weekdays are consistently louder due to commuter traffic.")
            with col_d:
                st.plotly_chart(self._fig_rush_vs_offpeak(filtered), use_container_width=True)
                st.caption("Rush hours add 5-8 dB on average compared to off-peak periods.")

        with tab4:
            col_e, col_f2 = st.columns(2)
            with col_e:
                st.plotly_chart(self._fig_noise_distribution(filtered), use_container_width=True)
                st.caption("Overlapping histograms show each city's noise distribution. Cairo skews right (louder).")
            with col_f2:
                st.plotly_chart(self._fig_noise_by_category(filtered), use_container_width=True)
                st.caption("WHO category breakdown. Most readings fall in the Loud (65-75 dB) range.")
            st.divider()
            st.plotly_chart(self._fig_cdf(filtered), use_container_width=True)
            st.caption("CDF shows what % of time each city stays below a given noise level. Tokyo has the best profile.")

        with tab5:
            st.markdown("#### City-by-City Statistical Summary")
            stats_df = self._city_stats_table(filtered)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
            st.divider()
            st.markdown("#### Correlation with Noise Level")
            corr_data = {
                'Feature': ['Traffic Count','Rush Hour','Humidity','PM2.5','Temperature','Wind Speed','Precipitation','Weekend','Night'],
                'Correlation': [
                    round(filtered['Noise_Level_dB'].corr(filtered['Traffic_Count']), 3),
                    round(filtered['Noise_Level_dB'].corr(filtered['is_rush_hour']), 3),
                    round(filtered['Noise_Level_dB'].corr(filtered['humidity']), 3),
                    round(filtered['Noise_Level_dB'].corr(filtered['pm25']), 3),
                    round(filtered['Noise_Level_dB'].corr(filtered['temperature']), 3),
                    round(filtered['Noise_Level_dB'].corr(filtered['wind_speed']), 3),
                    round(filtered['Noise_Level_dB'].corr(filtered['precipitation']), 3),
                    round(filtered['Noise_Level_dB'].corr(filtered['is_weekend']), 3),
                    round(filtered['Noise_Level_dB'].corr(filtered['is_night']), 3),
                ]
            }
            corr_df = pd.DataFrame(corr_data).sort_values('Correlation', key=abs, ascending=False)
            fig_corr = px.bar(corr_df, x='Feature', y='Correlation',
                              color='Correlation', color_continuous_scale='RdBu_r',
                              title='Feature Correlation with Noise Level',
                              range_color=[-1, 1])
            fig_corr.add_hline(y=0, line_color='black', line_width=1)
            fig_corr.update_layout(
                coloraxis_showscale=False,
                height=380,
                margin=dict(t=56, b=88, l=56, r=28),
                title=dict(x=0, xanchor='left'),
                xaxis=dict(tickangle=-30),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            st.divider()
            with st.expander("Raw Data Sample (200 rows)"):
                st.dataframe(filtered.sample(min(200, len(filtered))).reset_index(drop=True),
                             use_container_width=True)
