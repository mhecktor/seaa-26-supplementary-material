__all__ = ["display_topic_summary"]

def display_topic_summary(df_dominant_topics):
    """Displays the topics, keywords, titles, and abstracts in a readable format.

    Args:
        df_dominant_topics (pd.DataFrame): DataFrame containing the dominant topics and document details.
    """
    for _, row in df_dominant_topics.sort_values(['Topic_Num']).iterrows():
        print(f"Topic\n\t {row['Topic_Num']}: {row['Keywords']}")
        print(f"Title:\n\t {row['Title']}")
        print(f"Abstract:\n\t {row['Abstract']}")
        print()
