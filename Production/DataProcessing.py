import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent 
DATA_FOLDER = PROJECT_ROOT / 'Data'
OUTPUT_FOLDER = PROJECT_ROOT / 'Output' / 'data_proc_eng/'
OUTPUT_FILE = OUTPUT_FOLDER / 'se_data_processed_prod.csv'

INPUT_FILE = DATA_FOLDER / 'mini_sparkify_event_data.json'
PERSONA_COLS = ['userId','gender', 'location', 'registration']

def load_data(file_path=INPUT_FILE):
    try:
        se_data = pd.read_json(file_path, lines=True)
        se_data = se_data[se_data['userId']!=""]
        se_data['ts'] = pd.to_datetime(se_data['ts'], unit='ms')
        se_data['registration'] = pd.to_datetime(se_data['registration'], unit='ms')
        return se_data
    except Exception as e:
        print(f"[load_data]Error loading data: {e}")
        return None
    
def cohort_building(se_data):
    # Identify churned users
    churn_user_date_df = se_data[se_data['page'] == 'Cancellation Confirmation'][['userId', 'ts']].rename(columns={'ts':'cancel_date'})
    churn_user_date_df['churn'] = True
    churn_user_date_df['include'] = churn_user_date_df['cancel_date'].dt.date >= pd.to_datetime('2018-10-08').date()
    churn_user_date_df['start_date'] = None
    churn_user_date_df.loc[churn_user_date_df['include'], 'start_date'] = (churn_user_date_df.loc[churn_user_date_df['include'], 'cancel_date'] - pd.Timedelta(days=7)).dt.floor('d')
    exclude_churn_users = churn_user_date_df[~churn_user_date_df['include']]['userId'].unique()
    churn_user_date_df['end_date'] = churn_user_date_df['cancel_date']

    se_users = se_data[['userId']].drop_duplicates()
    se_users = se_users.merge(churn_user_date_df,how='left')
    se_users.loc[se_users['include'].isna(), 'include'] = True
    se_users.loc[se_users['churn'].isna(), 'churn'] = False
    se_users['include'] = se_users['include'].astype(bool)
    se_users['churn'] = se_users['churn'].astype(bool)
    start_dates = churn_user_date_df[churn_user_date_df['include']].start_date.tolist()
    mask = (se_users['include'])&(se_users.start_date.isna())
    n_start_dates_to_assign = se_users[mask].shape[0]
    # random sample with replacement from start_dates to assign to users without churn date
    se_users.loc[mask, 'start_date'] = pd.Series(start_dates).sample(n=n_start_dates_to_assign, replace=True, random_state=42).values
    se_users['start_date'] = pd.to_datetime(se_users['start_date'])
    se_users['end_date'] = se_users['start_date'] + pd.Timedelta(days=7)

    # Create sub_activity data to determine has_activity
    se_data_sub_activity = se_data[se_data['userId']!=""].merge(se_users[se_users['include']], on='userId', how='inner')
    se_data_sub_activity = se_data_sub_activity[(se_data_sub_activity['ts'] >= se_data_sub_activity['start_date']) & (se_data_sub_activity['ts'] <= se_data_sub_activity['end_date'])]
    se_users['has_activity'] = se_users['userId'].isin(se_data_sub_activity['userId'])

    return se_users

def create_persona_data(se_data, se_users):
    se_persona = se_users[se_users['include']][['userId','end_date']].merge(
        se_data[(se_data['userId'] != '') & (se_data['userId'].isin(se_users[se_users['include']]['userId']))][PERSONA_COLS]\
        .drop_duplicates(), how='inner', on='userId'
    )
    se_persona['day_since_reg'] = (se_persona['end_date'] - se_persona['registration']).dt.days
    print(f"persona df shape: {se_persona.shape}")
    se_persona[['city','state']] = se_persona['location'].str.split(',', expand=True)
    se_persona['city'] = se_persona['city'].str.strip().str.title()
    se_persona['state'] = se_persona['state'].str.strip().str.upper()
    se_persona = se_persona[['userId','gender','city','state','day_since_reg']]\
        .rename(columns={'gender':'psn_gender','city':'psn_city','state':'psn_state','day_since_reg':'acct_day_since_reg'})
    se_persona['psn_multi_city'] = se_persona['psn_city'].str.contains('-')
    se_persona['psn_multi_state'] = se_persona['psn_state'].str.contains('-')
    return se_persona

def create_account_data(se_data, se_users):
    se_data_in_scope_user = se_data[se_data['userId'].isin(se_users[se_users['include']]['userId'])].copy()
    act_before_end_df = se_data_in_scope_user.merge(se_users,how='left',on='userId').query('end_date > ts')
    se_last_level = (
        act_before_end_df[~act_before_end_df['page'].str.lower().str.contains('cancel')]
        .sort_values(['userId', 'ts'])
        .groupby('userId')['level']
        .agg(last_level='last', multiple_levels=lambda x: x.nunique() > 1)
        .reset_index()
    )
    # For users without any activity before end_date, find the nearest level record
    rest_level_df = se_data_in_scope_user[~se_data_in_scope_user['userId'].isin(se_last_level.userId)][['userId','ts','level']] \
        .merge(se_users[['userId','end_date']],how='left',on='userId').copy()
    rest_level_df['time_diff'] = (rest_level_df['end_date'] - rest_level_df['ts']).abs()
    rest_level_nearest = rest_level_df.sort_values(['userId','time_diff']).groupby('userId').first().reset_index()[['userId','level']].rename(columns={'level':'last_level'})
    rest_level_nearest['multiple_levels'] = False
    # Combine both dataframes
    se_last_level = pd.concat([se_last_level, rest_level_nearest], ignore_index=True)
    se_last_level = se_last_level.rename(columns={'last_level':'acct_last_level','multiple_levels':'acct_multiple_levels'})
    return se_last_level

def create_se_data_sub_activity(se_data, se_users):
    se_data_sub_activity = se_data[se_data['userId']!=""].merge(se_users[se_users['include']], on='userId', how='inner')
    se_data_sub_activity = se_data_sub_activity[(se_data_sub_activity['ts'] >= se_data_sub_activity['start_date']) & (se_data_sub_activity['ts'] <= se_data_sub_activity['end_date'])]
    return se_data_sub_activity

def create_activity_page_data(se_cohort_df,se_data_sub_activity):
    se_page = se_data_sub_activity.loc[~se_data_sub_activity['page'].str.lower().str.contains('cancel'), ['userId','page']]\
        .assign(page=lambda x: 'act_page_' + x['page'].str.replace(' ', '_').str.lower())\
        .pivot_table(index='userId', columns='page', aggfunc='size', fill_value=0)\
        .reset_index()
    se_page_agg = se_data_sub_activity.loc[~se_data_sub_activity['page'].str.lower().str.contains('cancel'), ['userId','page']]\
        .groupby('userId')['page'].agg(['count','nunique']).reset_index().rename(columns={'count':'total_page_visits', 'nunique':'unique_page_types'})
    se_page = se_page.merge(se_page_agg, how='left', on='userId')
    se_page = se_cohort_df[['userId']].merge(se_page, how='left', on='userId').fillna(0)
    for col in se_page.columns[1:]:
        se_page[col] = se_page[col].astype(int)
    return se_page

def create_activity_session_data(se_cohort_df,se_data_sub_activity):
    se_session = se_data_sub_activity[['userId','sessionId']]\
        .groupby('userId')['sessionId']\
        .agg(['nunique','mean'])\
        .reset_index()\
        .rename(columns={'nunique':'session_count', 'mean':'session_avg_pages'})
    se_session = se_session.merge(se_cohort_df[['userId']], how='right', on='userId').fillna(0)
    return se_session

def create_activity_active_days(se_cohort_df,se_data_sub_activity):
    se_days_active = se_data_sub_activity[['userId','ts']]\
        .assign(activity_day=lambda x: x['ts'].dt.floor('d'))\
        .groupby('userId')['activity_day'].nunique().reset_index().rename(columns={'activity_day':'act_days_active'})
    print(se_days_active.shape)
    se_days_active = se_days_active.merge(se_cohort_df[['userId']], how='right', on='userId').fillna(0)
    return se_days_active

def create_song_artist_metadata(se_data):
    se_data_song_artist = se_data[['artist','song']].dropna().copy()
    se_data_song_artist['song'] = se_data_song_artist['song'].str.strip().str.lower()
    se_data_song_artist['artist'] = se_data_song_artist['artist'].str.strip().str.lower()
    se_data_song_artist['song_artist'] = se_data_song_artist['artist'] + " - " + se_data_song_artist['song']
    
    song_artist_counts = se_data_song_artist['song_artist'].value_counts().reset_index().sort_values('count', ascending=False).reset_index(drop=True)
    song_artist_counts['prf_top100song'] = song_artist_counts['song_artist'].isin(song_artist_counts['song_artist'][:100])
    song_artist_counts['prf_top1000song'] = song_artist_counts['song_artist'].isin(song_artist_counts['song_artist'][:1000])
    song_artist_counts['prf_unique_song'] = song_artist_counts['count'] == 1

    artist_counts = se_data_song_artist['artist'].value_counts().reset_index().sort_values('count', ascending=False).reset_index(drop=True)
    artist_counts['prf_top20artist'] = artist_counts['artist'].isin(artist_counts['artist'][:20])
    artist_counts['prf_top100artist'] = artist_counts['artist'].isin(artist_counts['artist'][:100])
    artist_counts['prf_unique_artist'] = artist_counts['count'] == 1
    song_artist_meta_data = se_data_song_artist.drop_duplicates()\
        .merge(artist_counts.drop(columns=['count'])).merge(song_artist_counts.drop(columns=['count']))
    
    return song_artist_meta_data


def create_preference_data(se_cohort_df,se_data_sub_activity,song_artist_meta_data):
    se_prf_raw = se_data_sub_activity[['userId','song','artist']].dropna()\
        .assign(
            song=lambda x: x['song'].str.strip().str.lower(),
            artist=lambda x: x['artist'].str.strip().str.lower()
        )\
        .merge(song_artist_meta_data, how='left', on=['artist','song'])
    se_prf_counts = se_prf_raw[['userId','song_artist']].groupby('userId').nunique().reset_index().rename(columns={'song_artist':'prf_unique_songs_count'})
    se_prf_top = se_prf_raw[['userId','prf_top20artist','prf_top100artist','prf_unique_artist','prf_top100song','prf_top1000song','prf_unique_song']]\
        .groupby('userId').agg(['sum','any']).reset_index()
    se_prf_top.columns = [
        '_'.join(col).rstrip('_') if isinstance(col, tuple) else col
        for col in se_prf_top.columns
    ]
    se_prf_top = se_prf_top.merge(se_cohort_df[['userId']], how='right', on='userId')
    se_prf_top.columns = [col.replace('_sum','_count') for col in se_prf_top.columns]
    any_cols = [col for col in se_prf_top.columns if col.endswith('_any')]
    for col in any_cols:
        se_prf_top.loc[se_prf_top[col].isna(), col] = False
        se_prf_top[col] = se_prf_top[col].astype(bool)
    cnt_cols = [col for col in se_prf_top.columns if col.endswith('_count')]
    se_prf_top[cnt_cols] = se_prf_top[cnt_cols].fillna(0).astype(int)
    return se_prf_top


if __name__ == "__main__":
    se_data = load_data()
    print(f"[load_data]Data shape:\n{se_data.shape}")
    se_users = cohort_building(se_data)
    print(se_users.columns)
    print(f"[cohort_building]Cohort shape:\n{se_users.shape}")
    se_cohort_df = se_users[se_users['include']].copy().drop(columns=['include'])
    print(f"[cohort_building]Cohort after filtering shape:\n{se_cohort_df.shape}")
    se_persona = create_persona_data(se_data, se_users)
    print(f"[create_persona_data]Persona df shape:\n{se_persona.shape}")
    se_last_level = create_account_data(se_data, se_users)
    print(f"[create_account_data]Last level df shape:\n{se_last_level.shape}")
    se_data_sub_activity = create_se_data_sub_activity(se_data, se_users)
    print(f"[create_se_data_sub_activity]Sub activity df shape:\n{se_data_sub_activity.shape}")
    se_page = create_activity_page_data(se_cohort_df,se_data_sub_activity)
    print(f"[create_activity_page_data]Page df shape:\n{se_page.shape}")
    se_session = create_activity_session_data(se_cohort_df,se_data_sub_activity)
    print(f"[create_activity_session_data]Session df shape:\n{se_session.shape}")
    se_days_active = create_activity_active_days(se_cohort_df,se_data_sub_activity)
    print(f"[create_activity_active_days]Days active df shape:\n{se_days_active.shape}")
    song_artist_meta_data = create_song_artist_metadata(se_data)
    se_prf_top = create_preference_data(se_cohort_df,se_data_sub_activity,song_artist_meta_data)
    print(f"[create_preference_data]Preference top df shape:\n{se_prf_top.shape}")

    # Merge all data
    se_data_processed = se_cohort_df\
        .merge(se_persona, how='left', on='userId')\
        .merge(se_last_level, how='left', on='userId')\
        .merge(se_page, how='left', on='userId')\
        .merge(se_session, how='left', on='userId')\
        .merge(se_days_active, how='left', on='userId')\
        .merge(se_prf_top, how='left', on='userId')
    print(f"[Merge all data]Processed data shape:\n{se_data_processed.shape}")
    se_data_processed.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    